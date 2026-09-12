"""Optimizador de corte de varillas de acero para Streamlit Community Cloud."""
from __future__ import annotations

from collections import defaultdict, deque
from io import BytesIO, StringIO
import math
import re
from typing import Any, Iterable

import ezdxf
import pandas as pd
import pdfplumber
import streamlit as st
from ortools.linear_solver import pywraplp


PESOS_NOMINALES = {
    6: 0.222, 8: 0.395, 10: 0.617, 12: 0.888,
    16: 1.578, 20: 2.466, 25: 3.853, 32: 6.313,
}
COLUMNAS = ["Posición", "Diámetro (mm)", "Cantidad", "Longitud (m)"]
TOLERANCIA_MM = 10  # ± 1 cm en todo el modelo de optimización.


def tabla_vacia(filas: int = 5) -> pd.DataFrame:
    """Devuelve filas editables para comenzar una carga manual."""
    return pd.DataFrame({"Posición": [""] * filas, "Diámetro (mm)": [0] * filas,
                         "Cantidad": [0] * filas, "Longitud (m)": [0.0] * filas})


def limpiar_numero(val: Any) -> float:
    """Extrae el primer número de cualquier texto de plano o del editor."""
    if val is None or pd.isna(val):
        return 0.0
    texto = str(val).lower().strip().replace(" ", "")
    # Maneja 1.200,50 y 1,200.50 antes de usar la expresión regular.
    if "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    else:
        texto = texto.replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", texto)
    return float(match.group()) if match else 0.0


def limpiar_longitud_m(val: Any) -> float:
    """Convierte una longitud con unidad explícita o de plano a metros."""
    numero = limpiar_numero(val)
    if numero <= 0:
        return 0.0
    texto = str(val).lower().replace(" ", "")
    if "mm" in texto or ("m" not in texto and numero >= 100):
        return numero / 1000
    if "cm" in texto or ("m" not in texto and 10 <= numero < 100):
        return numero / 100
    return numero


def sanitizar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y tipa la tabla antes de cualquier cálculo o llamada a OR-Tools."""
    datos = df.copy()
    for columna in COLUMNAS:
        if columna not in datos:
            datos[columna] = "" if columna == "Posición" else 0
    datos["Posición"] = (datos["Posición"].fillna("").astype(str)
                         .str.replace(r"\s+", " ", regex=True).str.strip())
    datos["Cantidad"] = datos["Cantidad"].map(limpiar_numero)
    datos["Diámetro (mm)"] = datos["Diámetro (mm)"].map(limpiar_numero)
    datos["Longitud (m)"] = datos["Longitud (m)"].map(limpiar_longitud_m)

    # Tipos explícitos: no se permite texto o float en los rangos del solver.
    datos["Cantidad"] = pd.to_numeric(datos["Cantidad"], errors="coerce").fillna(0).astype(int)
    datos["Diámetro (mm)"] = pd.to_numeric(datos["Diámetro (mm)"], errors="coerce").fillna(0).astype(int)
    datos["Longitud (m)"] = pd.to_numeric(datos["Longitud (m)"], errors="coerce").fillna(0.0).astype(float)
    datos = datos[(datos["Cantidad"] > 0) & (datos["Longitud (m)"] > 0) & (datos["Diámetro (mm)"] > 0)]
    datos["Longitud (m)"] = datos["Longitud (m)"].round(4)
    faltantes = datos["Posición"] == ""
    datos.loc[faltantes, "Posición"] = [f"Sin posición {i + 1}" for i in range(int(faltantes.sum()))]
    return datos[COLUMNAS].reset_index(drop=True)


def peso_nominal(diametro: int) -> float:
    if int(diametro) not in PESOS_NOMINALES:
        admitidos = ", ".join(str(x) for x in PESOS_NOMINALES)
        raise ValueError(f"Ø {diametro} mm no está en la tabla de pesos nominales ({admitidos} mm).")
    return PESOS_NOMINALES[int(diametro)]


def _indice(celdas: list[Any], claves: tuple[str, ...]) -> int | None:
    for indice, celda in enumerate(celdas):
        if any(clave in str(celda or "").lower() for clave in claves):
            return indice
    return None


def _encabezado(celdas: list[Any]) -> tuple[int, int, int, int] | None:
    pos = _indice(celdas, ("pos", "marca", "item"))
    diam = _indice(celdas, ("diam", "diá", "ø", "∅", "φ"))
    cant = _indice(celdas, ("cant", "unid", "qty", "quant"))
    largo = _indice(celdas, ("long", "largo", "length"))
    return (pos, diam, cant, largo) if None not in (pos, diam, cant, largo) else None


def _registros_tabla(filas: Iterable[list[Any]]) -> list[dict[str, Any]]:
    registros: list[dict[str, Any]] = []
    cabecera: tuple[int, int, int, int] | None = None
    for fila in filas:
        fila = list(fila)
        nueva_cabecera = _encabezado(fila)
        if nueva_cabecera:
            cabecera = nueva_cabecera
            continue
        if cabecera is None:
            continue
        try:
            pos, diam, cant, largo = (fila[i] for i in cabecera)
        except IndexError:
            continue
        if str(pos or "").strip():
            registros.append({"Posición": str(pos).strip(), "Diámetro (mm)": diam,
                              "Cantidad": cant, "Longitud (m)": largo})
    return registros


def _registros_texto(texto: str) -> list[dict[str, Any]]:
    """Respaldo para PDFs/DXFs que no preservan su cuadrícula de tabla."""
    registros: list[dict[str, Any]] = []
    for linea in texto.splitlines():
        partes = [x for x in re.split(r"\s+", linea.strip()) if x]
        if len(partes) < 4:
            continue
        indice_diam = next((i for i, x in enumerate(partes) if re.search(r"[ø∅φ]", x.lower())), 1)
        if indice_diam == 0 or indice_diam + 2 >= len(partes):
            continue
        registros.append({"Posición": partes[0], "Diámetro (mm)": partes[indice_diam],
                          "Cantidad": partes[indice_diam + 1], "Longitud (m)": partes[indice_diam + 2]})
    return registros


def _normalizar_registros(registros: list[dict[str, Any]]) -> pd.DataFrame:
    if not registros:
        return tabla_vacia()
    datos = sanitizar_datos(pd.DataFrame(registros, columns=COLUMNAS))
    return datos if not datos.empty else tabla_vacia()


def extraer_pdf(contenido: bytes) -> pd.DataFrame:
    registros: list[dict[str, Any]] = []
    textos: list[str] = []
    with pdfplumber.open(BytesIO(contenido)) as pdf:
        for pagina in pdf.pages:
            filas: list[list[Any]] = []
            for tabla in pagina.extract_tables():
                filas.extend(fila for fila in tabla if fila)
            registros.extend(_registros_tabla(filas))
            textos.append(pagina.extract_text() or "")
    if not registros:
        registros = _registros_texto("\n".join(textos))
    return _normalizar_registros(registros)


def extraer_dxf(contenido: bytes) -> pd.DataFrame:
    """Lee textos CAD y los recompone en filas ordenadas por coordenada Y/X."""
    try:
        documento = ezdxf.read(StringIO(contenido.decode("utf-8", errors="ignore")))
    except Exception as exc:
        raise ValueError(f"No se pudo leer el DXF: {exc}") from exc
    textos: list[tuple[float, float, str]] = []
    for entidad in documento.modelspace():
        if entidad.dxftype() == "TEXT":
            valor = entidad.dxf.text
        elif entidad.dxftype() == "MTEXT":
            valor = entidad.plain_text()
        else:
            continue
        if str(valor).strip():
            punto = entidad.dxf.insert
            textos.append((float(punto.y), float(punto.x), str(valor).replace("\\P", " ").strip()))
    textos.sort(key=lambda x: (-x[0], x[1]))
    grupos: list[list[tuple[float, float, str]]] = []
    for texto in textos:
        if not grupos or abs(grupos[-1][0][0] - texto[0]) > 2.0:
            grupos.append([texto])
        else:
            grupos[-1].append(texto)
    filas = [[valor for _, _, valor in sorted(grupo, key=lambda x: x[1])] for grupo in grupos]
    registros = _registros_tabla(filas)
    if not registros:
        registros = _registros_texto("\n".join(" ".join(fila) for fila in filas))
    return _normalizar_registros(registros)


def extraer_archivo(archivo: Any) -> pd.DataFrame:
    contenido, nombre = archivo.getvalue(), archivo.name.lower()
    if nombre.endswith(".pdf"):
        return extraer_pdf(contenido)
    if nombre.endswith(".dxf"):
        return extraer_dxf(contenido)
    raise ValueError("Formato no admitido. Carga un PDF o DXF.")


def generar_patrones(longitudes_mm: list[int], demandas: list[int], barra_mm: int,
                     kerf_mm: int, limite: int = 25000) -> list[tuple[int, ...]]:
    """Genera combinaciones factibles enteras; cada patrón usa hasta +10 mm."""
    largos = [int(x) for x in longitudes_mm]
    requeridos = [int(x) for x in demandas]
    barra_mm, kerf_mm = int(barra_mm), int(kerf_mm)
    consumos = [int(largo) + kerf_mm for largo in largos]
    patrones: set[tuple[int, ...]] = set()
    cantidad_tipos = int(len(largos))

    for i, consumo in enumerate(consumos):
        if int(consumo) <= barra_mm + TOLERANCIA_MM:
            unitario = [0] * cantidad_tipos
            unitario[i] = 1
            patrones.add(tuple(unitario))

    def explorar(indice: int, restante: int, actual: list[int]) -> None:
        if len(patrones) >= int(limite):
            return
        if int(indice) == cantidad_tipos:
            if any(actual):
                patrones.add(tuple(int(x) for x in actual))
            return
        maximo = min(int(requeridos[indice]), int((int(restante) + TOLERANCIA_MM) // int(consumos[indice])))
        for unidades in range(int(maximo), -1, -1):
            actual.append(int(unidades))
            explorar(int(indice) + 1, int(restante) - int(unidades) * int(consumos[indice]), actual)
            actual.pop()
            if len(patrones) >= int(limite):
                return

    explorar(0, int(barra_mm), [])
    return sorted(patrones, key=lambda patron: (sum(patron), patron), reverse=True)


def resolver_diametro(datos: pd.DataFrame, barra_mm: int, kerf_mm: int,
                      minimo_reutilizable_mm: int) -> dict[str, Any]:
    """Resuelve un diámetro y consolida barras con el mismo patrón de corte."""
    datos = sanitizar_datos(datos)
    barra_mm, kerf_mm = int(barra_mm), int(kerf_mm)
    minimo_reutilizable_mm = int(minimo_reutilizable_mm)
    datos["Longitud mm"] = (datos["Longitud (m)"] * 1000).round().astype(int)
    demanda = datos.groupby("Longitud mm")["Cantidad"].sum().sort_index(ascending=False)
    largos_mm = [int(x) for x in demanda.index]
    cantidades = [int(x) for x in demanda.values]
    imposibles = [x for x in largos_mm if int(x) + kerf_mm > barra_mm + TOLERANCIA_MM]
    if imposibles:
        raise ValueError(f"Piezas mayores que la barra: {[x / 1000 for x in imposibles]}")
    patrones = generar_patrones(largos_mm, cantidades, barra_mm, kerf_mm)
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError("OR-Tools no pudo iniciar SCIP ni CBC.")
    solver.SetTimeLimit(30_000)
    variables = [solver.IntVar(0, solver.infinity(), f"patron_{i}") for i in range(int(len(patrones)))]
    for tipo, demanda_tipo in enumerate(cantidades):
        solver.Add(sum(int(patron[tipo]) * variables[i] for i, patron in enumerate(patrones)) == int(demanda_tipo))
    objetivo = solver.Objective()
    for variable in variables:
        objetivo.SetCoefficient(variable, 1)
    objetivo.SetMinimization()
    if solver.Solve() not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("No se encontró una solución factible.")

    etiquetas: dict[int, deque[str]] = defaultdict(deque)
    for _, fila in datos.sort_values(["Longitud mm", "Posición"], ascending=[False, True]).iterrows():
        etiquetas[int(fila["Longitud mm"])].extend([str(fila["Posición"])] * int(fila["Cantidad"]))
    cortes_individuales: list[dict[str, Any]] = []
    retazos: list[float] = []
    for indice, patron in enumerate(patrones):
        for _ in range(int(round(variables[indice].solution_value()))):
            piezas: list[tuple[str, int]] = []
            total_piezas_mm = 0
            for tipo, unidades in enumerate(patron):
                for _ in range(int(unidades)):
                    largo_mm = int(largos_mm[tipo])
                    piezas.append((etiquetas[largo_mm].popleft(), largo_mm))
                    total_piezas_mm += largo_mm
            sobrante_mm = max(0, int(barra_mm - total_piezas_mm - kerf_mm * int(len(piezas))))
            clasificacion = ("Sobrante de Stock" if sobrante_mm >= minimo_reutilizable_mm
                             else "Desperdicio / Chatarra")
            conteo: dict[tuple[str, int], int] = {}
            for posicion, largo_mm in piezas:
                clave = (posicion, largo_mm)
                conteo[clave] = conteo.get(clave, 0) + 1
            descripcion = " + ".join(f"{int(cantidad)}x {posicion} ({largo / 1000:.2f} m)"
                                     for (posicion, largo), cantidad in conteo.items())
            cortes_individuales.append({"Patrón de Corte": descripcion,
                                        "Sobrante por Barra (m)": sobrante_mm / 1000,
                                        "Clasificación": clasificacion})
            if clasificacion == "Sobrante de Stock":
                retazos.append(sobrante_mm / 1000)

    agrupados: dict[tuple[str, float, str], int] = {}
    for corte in cortes_individuales:
        clave = (corte["Patrón de Corte"], round(corte["Sobrante por Barra (m)"], 3), corte["Clasificación"])
        agrupados[clave] = agrupados.get(clave, 0) + 1
    plan = [{"Cantidad de Barras": int(cantidad), "Patrón de Corte": patron,
             "Sobrante por Barra (m)": sobrante, "Clasificación": clasificacion}
            for (patron, sobrante, clasificacion), cantidad in agrupados.items()]
    chatarra_m = sum(c["Sobrante por Barra (m)"] for c in cortes_individuales
                     if c["Clasificación"] == "Desperdicio / Chatarra")
    metros_piezas = sum(int(largo) * int(cantidad) for largo, cantidad in zip(largos_mm, cantidades)) / 1000
    return {"barras": int(len(cortes_individuales)), "plan": plan, "retazos": retazos,
            "metros_piezas": metros_piezas, "metros_chatarra": chatarra_m}


def optimizar(datos: pd.DataFrame, longitud_barra_m: float, kerf_cm: float,
              minimo_reutilizable_m: float) -> dict[str, Any]:
    """Calcula cada diámetro por separado y consolida resultados físicos."""
    datos = sanitizar_datos(datos)
    if datos.empty:
        raise ValueError("No hay filas válidas para optimizar.")
    for diametro in datos["Diámetro (mm)"].unique():
        peso_nominal(int(diametro))
    barra_mm = int(float(longitud_barra_m) * 1000)
    kerf_mm = int(float(kerf_cm) * 10)
    minimo_mm = int(float(minimo_reutilizable_m) * 1000)
    if barra_mm <= 0 or kerf_mm < 0 or minimo_mm < 0:
        raise ValueError("Los parámetros de corte no son válidos.")

    resumen, plan_total, retazos_total = [], [], []
    for diametro, grupo in datos.groupby("Diámetro (mm)", sort=True):
        diametro = int(diametro)
        solucion = resolver_diametro(grupo, barra_mm, kerf_mm, minimo_mm)
        peso = peso_nominal(diametro)
        for fila in solucion["plan"]:
            fila["Diámetro (mm)"] = diametro
            plan_total.append(fila)
        for largo in solucion["retazos"]:
            retazos_total.append({"Diámetro (mm)": diametro, "Longitud (m)": largo})
        resumen.append({"Diámetro (mm)": diametro, "Barras a comprar": solucion["barras"],
                        "Metros de piezas": solucion["metros_piezas"],
                        "Kg de piezas": solucion["metros_piezas"] * peso,
                        "Kg de chatarra": solucion["metros_chatarra"] * peso})
    df_resumen = pd.DataFrame(resumen)
    df_plan = pd.DataFrame(plan_total)
    if not df_plan.empty:
        df_plan = df_plan[["Cantidad de Barras", "Diámetro (mm)", "Patrón de Corte",
                           "Sobrante por Barra (m)", "Clasificación"]]
    df_retazos = pd.DataFrame(retazos_total)
    if df_retazos.empty:
        df_retazos = pd.DataFrame(columns=["Diámetro (mm)", "Longitud (m)", "Cantidad"])
    else:
        df_retazos["Longitud (m)"] = df_retazos["Longitud (m)"].round(2)
        df_retazos = (df_retazos.groupby(["Diámetro (mm)", "Longitud (m)"], as_index=False)
                      .size().rename(columns={"size": "Cantidad"})
                      .sort_values(["Diámetro (mm)", "Longitud (m)"], ascending=[True, False]))
    total_barras = int(df_resumen["Barras a comprar"].sum())
    metros_piezas = float(df_resumen["Metros de piezas"].sum())
    return {"resumen": df_resumen, "plan": df_plan, "retazos": df_retazos,
            "kg_total": float(df_resumen["Kg de piezas"].sum()),
            "kg_chatarra": float(df_resumen["Kg de chatarra"].sum()),
            "aprovechamiento": 100 * metros_piezas / (total_barras * float(longitud_barra_m))}


def mostrar_dos_decimales(df: pd.DataFrame) -> pd.DataFrame:
    copia = df.copy()
    for columna in copia.select_dtypes(include="number").columns:
        if columna not in {"Cantidad", "Cantidad de Barras", "Barras a comprar"}:
            copia[columna] = copia[columna].round(2)
    return copia


def crear_excel(resultado: dict[str, Any], longitud_barra_m: float, kerf_cm: float,
                minimo_reutilizable_m: float) -> bytes:
    salida = BytesIO()
    with pd.ExcelWriter(salida, engine="xlsxwriter") as escritor:
        libro = escritor.book
        titulo = libro.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF"})
        decimal = libro.add_format({"num_format": "0.00"})
        porcentaje = libro.add_format({"num_format": "0.00%"})
        resumen = resultado["resumen"].copy()
        resumen.loc[len(resumen)] = ["TOTAL", int(resumen["Barras a comprar"].sum()),
                                     resumen["Metros de piezas"].sum(), resultado["kg_total"], resultado["kg_chatarra"]]
        resumen.to_excel(escritor, sheet_name="Resumen General", index=False, startrow=5)
        hoja = escritor.sheets["Resumen General"]
        hoja.write("A1", "Configuración", titulo)
        hoja.write_row("A2", ["Longitud barra (m)", longitud_barra_m, "Kerf (cm)", kerf_cm,
                                "Longitud mínima reutilizable (m)", minimo_reutilizable_m])
        hoja.write("A3", "Aprovechamiento global", titulo)
        hoja.write_number("B3", resultado["aprovechamiento"] / 100, porcentaje)
        for indice, nombre in enumerate(resumen.columns): hoja.write(5, indice, nombre, titulo)
        hoja.set_column("A:A", 20); hoja.set_column("B:E", 20, decimal)

        resultado["plan"].to_excel(escritor, sheet_name="Plan de Corte Agrupado", index=False)
        hoja = escritor.sheets["Plan de Corte Agrupado"]
        for indice, nombre in enumerate(resultado["plan"].columns): hoja.write(0, indice, nombre, titulo)
        hoja.set_column("A:B", 18); hoja.set_column("C:C", 70); hoja.set_column("D:D", 25, decimal); hoja.set_column("E:E", 25)

        resultado["retazos"].to_excel(escritor, sheet_name="Inventario Sobrantes Stock", index=False)
        hoja = escritor.sheets["Inventario Sobrantes Stock"]
        for indice, nombre in enumerate(resultado["retazos"].columns): hoja.write(0, indice, nombre, titulo)
        hoja.set_column("A:C", 24, decimal)
    return salida.getvalue()


def cargar_archivo_si_corresponde(archivo: Any) -> None:
    if archivo is None:
        return
    identificador = (archivo.name, archivo.size)
    if st.session_state.get("archivo_cargado") == identificador:
        return
    try:
        st.session_state["tabla_hierros"] = extraer_archivo(archivo)
        st.session_state["archivo_cargado"] = identificador
        st.session_state.pop("editor_hierros", None)  # evita conservar edición de otro archivo
        if st.session_state["tabla_hierros"].empty:
            st.session_state["tabla_hierros"] = tabla_vacia()
            st.warning("No se detectaron filas. Completa la tabla manualmente.")
        else:
            st.success(f"Se detectaron {len(st.session_state['tabla_hierros'])} filas. Revísalas antes de optimizar.")
    except Exception as exc:
        st.session_state["tabla_hierros"] = tabla_vacia()
        st.session_state.pop("editor_hierros", None)
        st.error(f"No se pudo procesar el archivo: {exc}")


st.set_page_config(page_title="Optimizador de corte de acero", layout="wide")
st.title("Optimizador de corte de varillas de acero")
st.caption("Carga un plano PDF/DXF o ingresa manualmente la lista de hierros.")
with st.sidebar:
    st.header("Configuración")
    longitud_barra = st.number_input("Longitud de barra comercial (m)", min_value=0.10, value=12.00, step=0.10, format="%.2f")
    kerf_cm = st.number_input("Desgaste del disco / kerf (cm)", min_value=0.00, value=0.00, step=0.10, format="%.2f")
    minimo_reutilizable = st.number_input("Longitud mínima reutilizable (m)", min_value=0.00, value=0.50, step=0.05, format="%.2f")
    archivo = st.file_uploader("Plano o planilla", type=["pdf", "dxf"])
    st.caption("Tolerancia de corte: ±0.01 m.")

if "tabla_hierros" not in st.session_state:
    st.session_state["tabla_hierros"] = tabla_vacia()
cargar_archivo_si_corresponde(archivo)

st.subheader("Lista de hierros")
tabla_editada = st.data_editor(
    st.session_state["tabla_hierros"], key="editor_hierros", num_rows="dynamic", use_container_width=True,
    column_config={
        "Posición": st.column_config.TextColumn("Posición", required=False),
        "Diámetro (mm)": st.column_config.NumberColumn("Diámetro Ø (mm)", min_value=0, step=1, format="%d"),
        "Cantidad": st.column_config.NumberColumn("Cantidad", min_value=0, step=1, format="%d"),
        "Longitud (m)": st.column_config.NumberColumn("Longitud (m)", min_value=0.0, step=0.01, format="%.2f"),
    },
)
# La asignación posterior al widget conserva la edición confirmada con Enter en
# el siguiente rerun, sin reinyectar el dataframe original en el editor.
st.session_state["tabla_hierros"] = tabla_editada

if st.button("Optimizar plan de corte", type="primary", use_container_width=True):
    datos = sanitizar_datos(st.session_state["tabla_hierros"])
    if datos.empty:
        st.error("Ingresa al menos una posición con diámetro, cantidad y longitud válidos.")
    else:
        try:
            with st.spinner("Calculando patrones óptimos en milímetros..."):
                st.session_state["resultado"] = optimizar(datos, longitud_barra, kerf_cm, minimo_reutilizable)
                st.session_state["config_resultado"] = (longitud_barra, kerf_cm, minimo_reutilizable)
        except Exception as exc:
            st.error(f"No fue posible optimizar: {exc}")

if "resultado" in st.session_state:
    resultado = st.session_state["resultado"]
    st.subheader("Resumen ejecutivo")
    total_barras = int(resultado["resumen"]["Barras a comprar"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Barras comerciales", total_barras)
    c2.metric("Acero requerido", f"{resultado['kg_total']:.2f} kg")
    c3.metric("Aprovechamiento global", f"{resultado['aprovechamiento']:.2f}%")
    c4.metric("Chatarra generada", f"{resultado['kg_chatarra']:.2f} kg")
    st.subheader("Resumen por diámetro")
    st.dataframe(mostrar_dos_decimales(resultado["resumen"]), use_container_width=True, hide_index=True)
    st.subheader("Plan de corte agrupado")
    st.dataframe(mostrar_dos_decimales(resultado["plan"]), use_container_width=True, hide_index=True)
    st.subheader("Inventario de sobrantes de stock")
    st.dataframe(mostrar_dos_decimales(resultado["retazos"]), use_container_width=True, hide_index=True)
    st.download_button("Descargar reporte Excel", data=crear_excel(resultado, *st.session_state["config_resultado"]),
                       file_name="plan_corte_acero.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
