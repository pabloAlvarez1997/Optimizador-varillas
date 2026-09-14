"""Optimizador de corte de acero: PDF/DXF, patrones MIP y reporte Excel."""
from __future__ import annotations

from collections import defaultdict, deque
from io import BytesIO, StringIO
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
COL_POS = "Posición"
COL_DIAM = "Diámetro Ø (mm)"
COL_CANT = "Cantidad"
COL_LARGO = "Longitud (m)"
COLUMNAS = [COL_POS, COL_DIAM, COL_CANT, COL_LARGO]
TOLERANCIA_MM = 10
DIAMETROS_RE = "6|8|10|12|16|20|25|32"


def tabla_inicial() -> pd.DataFrame:
    """Cinco filas editables, incluso si aún no se cargó un plano."""
    return pd.DataFrame({
        COL_POS: [f"P{i}" for i in range(1, 6)],
        COL_DIAM: [12, 10, 12, 16, 20],
        COL_CANT: [0, 0, 0, 0, 0],
        COL_LARGO: [0.0, 0.0, 0.0, 0.0, 0.0],
    })


def limpiar_numero(val: Any) -> float:
    """Extrae un número de una celda con símbolos, texto u OCR imperfecto."""
    if val is None or pd.isna(val):
        return 0.0
    texto = str(val).lower().strip().replace(" ", "")
    if "," in texto and "." in texto:
        texto = (texto.replace(".", "").replace(",", ".")
                 if texto.rfind(",") > texto.rfind(".") else texto.replace(",", ""))
    else:
        texto = texto.replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", texto)
    return float(match.group()) if match else 0.0


def limpiar_longitud_m(val: Any) -> float:
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
    """Limpia texto y fuerza tipos numéricos antes de llegar al solver."""
    datos = df.copy()
    for columna in COLUMNAS:
        if columna not in datos:
            datos[columna] = "" if columna == COL_POS else 0
    datos[COL_POS] = (datos[COL_POS].fillna("").astype(str)
                       .str.replace(r"\s+", " ", regex=True).str.strip())
    datos[COL_CANT] = datos[COL_CANT].map(limpiar_numero)
    datos[COL_DIAM] = datos[COL_DIAM].map(limpiar_numero)
    datos[COL_LARGO] = datos[COL_LARGO].map(limpiar_longitud_m)
    datos[COL_CANT] = pd.to_numeric(datos[COL_CANT], errors="coerce").fillna(0).astype(int)
    datos[COL_DIAM] = pd.to_numeric(datos[COL_DIAM], errors="coerce").fillna(0).astype(int)
    datos[COL_LARGO] = pd.to_numeric(datos[COL_LARGO], errors="coerce").fillna(0.0).astype(float)
    datos = datos[(datos[COL_CANT] > 0) & (datos[COL_LARGO] > 0) & (datos[COL_DIAM] > 0)].copy()
    datos[COL_LARGO] = datos[COL_LARGO].round(4)
    vacias = datos[COL_POS] == ""
    datos.loc[vacias, COL_POS] = [f"Sin posición {i + 1}" for i in range(int(vacias.sum()))]
    return datos[COLUMNAS].reset_index(drop=True)


def peso_nominal(diametro: int) -> float:
    if int(diametro) not in PESOS_NOMINALES:
        disponibles = ", ".join(map(str, PESOS_NOMINALES))
        raise ValueError(f"Ø {diametro} mm no tiene peso nominal. Admitidos: {disponibles} mm.")
    return PESOS_NOMINALES[int(diametro)]


def _encabezado_indices(fila: list[Any]) -> tuple[int, int, int, int] | None:
    def buscar(terminos: tuple[str, ...]) -> int | None:
        for i, celda in enumerate(fila):
            if any(t in str(celda or "").lower() for t in terminos):
                return i
        return None
    indices = (buscar(("pos", "marca", "item")), buscar(("diam", "diá", "ø", "∅", "φ")),
               buscar(("cant", "unid", "qty")), buscar(("long", "largo", "length")))
    return indices if None not in indices else None


def _extraer_filas_tabla(filas: Iterable[list[Any]]) -> list[dict[str, Any]]:
    cabecera: tuple[int, int, int, int] | None = None
    longitud_en_cm = False
    registros: list[dict[str, Any]] = []
    for fila in filas:
        fila = list(fila)
        nueva = _encabezado_indices(fila)
        if nueva:
            cabecera = nueva
            longitud_en_cm = "cm" in str(fila[cabecera[3]] or "").lower()
            continue
        if cabecera is None:
            continue
        try:
            pos, diam, cant, largo = (fila[i] for i in cabecera)
        except IndexError:
            continue
        if str(pos or "").strip():
            if longitud_en_cm:
                largo = f"{largo} cm"
            registros.append({COL_POS: pos, COL_DIAM: diam, COL_CANT: cant, COL_LARGO: largo})
    return registros


def extraer_lista_hierros_texto(texto: str) -> list[dict[str, Any]]:
    """Parser de planos: POS, Ø, CANT, LONG UNIT (cm), LONG TOTAL (m)."""
    patron = re.compile(
        rf"^\s*(?P<pos>\d{{1,4}})\s+(?:[Øø∅φ]\s*)?(?P<diam>{DIAMETROS_RE})\s+"
        r"(?P<cant>\d+)\s+(?P<long>[\d.,]+)(?:\s+[\d.,]+)?\s*$",
        re.IGNORECASE,
    )
    lineas = texto.splitlines()
    hay_bloque = any("lista de hierros" in linea.lower() for linea in lineas)
    en_bloque = not hay_bloque
    registros: list[dict[str, Any]] = []
    for linea in lineas:
        if "lista de hierros" in linea.lower():
            en_bloque = True
            continue
        if not en_bloque:
            continue
        coincidencia = patron.match(linea.strip())
        if not coincidencia:
            continue
        largo_crudo = limpiar_numero(coincidencia.group("long"))
        # LONG UNIT del plano se expresa normalmente en cm. Valores >= 100
        # se convierten directamente a metros como solicita el formato estructural.
        largo_m = largo_crudo / 100 if largo_crudo >= 100 else largo_crudo
        registros.append({COL_POS: f"P{coincidencia.group('pos')}",
                          COL_DIAM: coincidencia.group("diam"),
                          COL_CANT: coincidencia.group("cant"), COL_LARGO: largo_m})
    return registros


def extraer_pdf(contenido: bytes) -> pd.DataFrame:
    textos, registros_tabla = [], []
    with pdfplumber.open(BytesIO(contenido)) as documento:
        for pagina in documento.pages:
            texto = pagina.extract_text() or ""
            textos.append(texto)
            for tabla in pagina.extract_tables():
                registros_tabla.extend(_extraer_filas_tabla([fila for fila in tabla if fila]))
    registros = extraer_lista_hierros_texto("\n".join(textos))
    if not registros:
        registros = registros_tabla
    datos = sanitizar_datos(pd.DataFrame(registros, columns=COLUMNAS)) if registros else pd.DataFrame()
    return datos if not datos.empty else tabla_inicial()


def extraer_dxf(contenido: bytes) -> pd.DataFrame:
    try:
        documento = ezdxf.read(StringIO(contenido.decode("utf-8", errors="ignore")))
    except Exception as exc:
        raise ValueError(f"DXF no legible: {exc}") from exc
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
    textos.sort(key=lambda item: (-item[0], item[1]))
    grupos: list[list[tuple[float, float, str]]] = []
    for texto in textos:
        if not grupos or abs(grupos[-1][0][0] - texto[0]) > 2:
            grupos.append([texto])
        else:
            grupos[-1].append(texto)
    filas = [[valor for _, _, valor in sorted(grupo, key=lambda item: item[1])] for grupo in grupos]
    plano = "\n".join(" ".join(fila) for fila in filas)
    registros = extraer_lista_hierros_texto(plano) or _extraer_filas_tabla(filas)
    datos = sanitizar_datos(pd.DataFrame(registros, columns=COLUMNAS)) if registros else pd.DataFrame()
    return datos if not datos.empty else tabla_inicial()


def extraer_archivo(archivo: Any) -> pd.DataFrame:
    if archivo.name.lower().endswith(".pdf"):
        return extraer_pdf(archivo.getvalue())
    if archivo.name.lower().endswith(".dxf"):
        return extraer_dxf(archivo.getvalue())
    raise ValueError("Solo se admiten archivos PDF y DXF.")


def generar_patrones(largos_mm: list[int], demandas: list[int], barra_mm: int,
                     kerf_mm: int, limite: int = 25000) -> list[tuple[int, ...]]:
    """Enumera patrones con enteros en mm y tolerancia global de 10 mm."""
    largos = [int(x) for x in largos_mm]
    requeridos = [int(x) for x in demandas]
    consumos = [int(largo) + int(kerf_mm) for largo in largos]
    patrones: set[tuple[int, ...]] = set()
    n = int(len(largos))
    for i, consumo in enumerate(consumos):
        if int(consumo) <= int(barra_mm) + TOLERANCIA_MM:
            p = [0] * n
            p[i] = 1
            patrones.add(tuple(p))

    def explorar(i: int, restante: int, actual: list[int]) -> None:
        if len(patrones) >= int(limite):
            return
        if int(i) == n:
            if any(actual):
                patrones.add(tuple(int(x) for x in actual))
            return
        maximo = min(int(requeridos[i]), int((int(restante) + TOLERANCIA_MM) // int(consumos[i])))
        for unidades in range(int(maximo), -1, -1):
            actual.append(int(unidades))
            explorar(int(i) + 1, int(restante) - int(unidades) * int(consumos[i]), actual)
            actual.pop()
            if len(patrones) >= int(limite):
                return
    explorar(0, int(barra_mm), [])
    return sorted(patrones, key=lambda p: (sum(p), p), reverse=True)


def resolver_diametro(datos: pd.DataFrame, barra_mm: int, kerf_mm: int,
                      minimo_mm: int) -> dict[str, Any]:
    datos = sanitizar_datos(datos)
    datos["Largo mm"] = (datos[COL_LARGO] * 1000).round().astype(int)
    demanda = datos.groupby("Largo mm")[COL_CANT].sum().sort_index(ascending=False)
    largos, cantidades = [int(x) for x in demanda.index], [int(x) for x in demanda.values]
    if any(int(x) + int(kerf_mm) > int(barra_mm) + TOLERANCIA_MM for x in largos):
        raise ValueError("Una o más piezas no caben dentro de la longitud comercial.")
    patrones = generar_patrones(largos, cantidades, int(barra_mm), int(kerf_mm))
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError("No se pudo iniciar el solver SCIP/CBC.")
    solver.SetTimeLimit(30_000)
    variables = [solver.IntVar(0, solver.infinity(), f"p_{i}") for i in range(int(len(patrones)))]
    for tipo, demanda_tipo in enumerate(cantidades):
        solver.Add(sum(int(patron[tipo]) * variables[i] for i, patron in enumerate(patrones)) == int(demanda_tipo))
    objetivo = solver.Objective()
    for variable in variables: objetivo.SetCoefficient(variable, 1)
    objetivo.SetMinimization()
    if solver.Solve() not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("No se encontró un plan de corte factible.")

    etiquetas: dict[int, deque[str]] = defaultdict(deque)
    for _, fila in datos.sort_values(["Largo mm", COL_POS], ascending=[False, True]).iterrows():
        etiquetas[int(fila["Largo mm"])].extend([str(fila[COL_POS])] * int(fila[COL_CANT]))
    individuales, retazos = [], []
    for i, patron in enumerate(patrones):
        for _ in range(int(round(variables[i].solution_value()))):
            piezas, usados_mm = [], 0
            for tipo, unidades in enumerate(patron):
                for _ in range(int(unidades)):
                    largo = int(largos[tipo])
                    piezas.append((etiquetas[largo].popleft(), largo))
                    usados_mm += largo
            sobrante_mm = max(0, int(barra_mm) - usados_mm - int(kerf_mm) * int(len(piezas)))
            clasificacion = "Sobrante de Stock" if sobrante_mm >= int(minimo_mm) else "Desperdicio / Chatarra"
            conteo: dict[tuple[str, int], int] = {}
            for posicion, largo in piezas:
                conteo[(posicion, largo)] = conteo.get((posicion, largo), 0) + 1
            patron_texto = " + ".join(f"{int(cantidad)}x {posicion} ({largo / 1000:.2f} m)"
                                       for (posicion, largo), cantidad in conteo.items())
            individuales.append({"Patrón de Corte": patron_texto, "Sobrante por Barra (m)": sobrante_mm / 1000,
                                "Clasificación": clasificacion})
            if clasificacion == "Sobrante de Stock": retazos.append(sobrante_mm / 1000)
    grupos: dict[tuple[str, float, str], int] = {}
    for fila in individuales:
        clave = (fila["Patrón de Corte"], round(fila["Sobrante por Barra (m)"], 3), fila["Clasificación"])
        grupos[clave] = grupos.get(clave, 0) + 1
    plan = [{"Cantidad de Barras": int(cantidad), "Patrón de Corte": patron,
             "Sobrante por Barra (m)": sobrante, "Clasificación": clasificacion}
            for (patron, sobrante, clasificacion), cantidad in grupos.items()]
    return {"barras": int(len(individuales)), "plan": plan, "retazos": retazos,
            "metros_piezas": sum(int(largo) * int(cantidad) for largo, cantidad in zip(largos, cantidades)) / 1000,
            "metros_chatarra": sum(f["Sobrante por Barra (m)"] for f in individuales
                                    if f["Clasificación"] == "Desperdicio / Chatarra")}


def optimizar(datos: pd.DataFrame, barra_m: float, kerf_cm: float, minimo_m: float) -> dict[str, Any]:
    datos = sanitizar_datos(datos)
    if datos.empty: raise ValueError("No hay posiciones válidas para optimizar.")
    for diametro in datos[COL_DIAM].unique(): peso_nominal(int(diametro))
    barra_mm, kerf_mm, minimo_mm = int(float(barra_m) * 1000), int(float(kerf_cm) * 10), int(float(minimo_m) * 1000)
    if barra_mm <= 0 or kerf_mm < 0 or minimo_mm < 0: raise ValueError("Configuración de corte inválida.")
    resumen, planes, retazos = [], [], []
    for diametro, grupo in datos.groupby(COL_DIAM, sort=True):
        diametro, peso = int(diametro), peso_nominal(int(diametro))
        solucion = resolver_diametro(grupo, barra_mm, kerf_mm, minimo_mm)
        for fila in solucion["plan"]:
            fila[COL_DIAM] = diametro
            planes.append(fila)
        retazos.extend({COL_DIAM: diametro, COL_LARGO: largo} for largo in solucion["retazos"])
        resumen.append({COL_DIAM: diametro, "Barras a comprar": solucion["barras"],
                        "Metros de piezas": solucion["metros_piezas"], "Kg de piezas": solucion["metros_piezas"] * peso,
                        "Kg de chatarra": solucion["metros_chatarra"] * peso})
    df_resumen = pd.DataFrame(resumen)
    df_plan = pd.DataFrame(planes)
    if not df_plan.empty: df_plan = df_plan[["Cantidad de Barras", COL_DIAM, "Patrón de Corte", "Sobrante por Barra (m)", "Clasificación"]]
    df_retazos = pd.DataFrame(retazos)
    if df_retazos.empty:
        df_retazos = pd.DataFrame(columns=[COL_DIAM, COL_LARGO, "Cantidad"])
    else:
        df_retazos[COL_LARGO] = df_retazos[COL_LARGO].round(2)
        df_retazos = (df_retazos.groupby([COL_DIAM, COL_LARGO], as_index=False).size()
                      .rename(columns={"size": "Cantidad"}).sort_values([COL_DIAM, COL_LARGO], ascending=[True, False]))
    barras = int(df_resumen["Barras a comprar"].sum())
    metros = float(df_resumen["Metros de piezas"].sum())
    return {"resumen": df_resumen, "plan": df_plan, "retazos": df_retazos,
            "kg_total": float(df_resumen["Kg de piezas"].sum()), "kg_chatarra": float(df_resumen["Kg de chatarra"].sum()),
            "aprovechamiento": 100 * metros / (barras * float(barra_m))}


def formatear(df: pd.DataFrame) -> pd.DataFrame:
    copia = df.copy()
    for columna in copia.select_dtypes(include="number"):
        if columna not in {COL_CANT, "Cantidad", "Cantidad de Barras", "Barras a comprar"}: copia[columna] = copia[columna].round(2)
    return copia


def crear_excel(resultado: dict[str, Any], barra_m: float, kerf_cm: float, minimo_m: float) -> bytes:
    salida = BytesIO()
    with pd.ExcelWriter(salida, engine="xlsxwriter") as escritor:
        libro = escritor.book
        cabecera, decimal, porcentaje = (libro.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF"}),
                                         libro.add_format({"num_format": "0.00"}), libro.add_format({"num_format": "0.00%"}))
        resumen = resultado["resumen"].copy()
        resumen.loc[len(resumen)] = ["TOTAL", int(resumen["Barras a comprar"].sum()), resumen["Metros de piezas"].sum(), resultado["kg_total"], resultado["kg_chatarra"]]
        resumen.to_excel(escritor, sheet_name="Resumen General", index=False, startrow=5)
        ws = escritor.sheets["Resumen General"]
        ws.write("A1", "Configuración", cabecera); ws.write_row("A2", ["Longitud barra (m)", barra_m, "Kerf (cm)", kerf_cm, "Longitud mínima reutilizable (m)", minimo_m])
        ws.write("A3", "Aprovechamiento global", cabecera); ws.write_number("B3", resultado["aprovechamiento"] / 100, porcentaje)
        for i, nombre in enumerate(resumen.columns): ws.write(5, i, nombre, cabecera)
        ws.set_column("A:A", 20); ws.set_column("B:E", 20, decimal)
        for nombre_hoja, datos, anchos in [("Plan de Corte Agrupado", resultado["plan"], (18, 70)), ("Inventario Sobrantes Stock", resultado["retazos"], (24, 24))]:
            datos.to_excel(escritor, sheet_name=nombre_hoja, index=False)
            ws = escritor.sheets[nombre_hoja]
            for i, nombre in enumerate(datos.columns): ws.write(0, i, nombre, cabecera)
            ws.set_column("A:B", anchos[0]); ws.set_column("C:C", anchos[1]); ws.set_column("D:D", 24, decimal)
    return salida.getvalue()


def actualizar_tabla() -> None:
    """Persiste la edición confirmada por Enter antes del siguiente rerun."""
    cambios = st.session_state.get("editor_activo", {})
    if not isinstance(cambios, dict): return
    base = st.session_state["datos_tabla"].copy()
    for indice, valores in cambios.get("edited_rows", {}).items():
        indice = int(indice)
        if indice < len(base):
            for columna, valor in valores.items(): base.at[indice, columna] = valor
    st.session_state["datos_tabla"] = base


def cargar_plano(archivo: Any) -> None:
    if archivo is None: return
    identidad = (archivo.name, archivo.size)
    if st.session_state.get("archivo_actual") == identidad: return
    try:
        st.session_state["datos_tabla"] = extraer_archivo(archivo)
        st.session_state["archivo_actual"] = identidad
        st.session_state.pop("editor_activo", None)
        st.rerun()
    except Exception as exc:
        st.session_state["datos_tabla"] = tabla_inicial()
        st.session_state.pop("editor_activo", None)
        st.error(f"No se pudo leer el archivo: {exc}")


st.set_page_config(page_title="Optimizador de corte de acero", layout="wide")
st.title("Optimizador de corte de varillas de acero")
with st.sidebar:
    st.header("Configuración")
    barra_m = st.number_input("Longitud de barra comercial (m)", 0.10, value=12.00, step=0.10, format="%.2f")
    kerf_cm = st.number_input("Desgaste del disco / kerf (cm)", 0.00, value=0.00, step=0.10, format="%.2f")
    minimo_m = st.number_input("Longitud mínima reutilizable (m)", 0.00, value=0.50, step=0.05, format="%.2f")
    archivo = st.file_uploader("Plano PDF o DXF", type=["pdf", "dxf"])
    st.caption("Tolerancia aplicada en el corte: ±0.01 m.")

if "datos_tabla" not in st.session_state: st.session_state["datos_tabla"] = tabla_inicial()
cargar_plano(archivo)
st.subheader("Lista de hierros")
df_editado = st.data_editor(st.session_state["datos_tabla"], key="editor_activo", num_rows="dynamic", on_change=actualizar_tabla, use_container_width=True,
    column_config={COL_POS: st.column_config.TextColumn(COL_POS), COL_DIAM: st.column_config.NumberColumn(COL_DIAM, min_value=0, step=1, format="%d"),
                   COL_CANT: st.column_config.NumberColumn(COL_CANT, min_value=0, step=1, format="%d"), COL_LARGO: st.column_config.NumberColumn(COL_LARGO, min_value=0.0, step=0.01, format="%.2f")})
st.session_state["datos_tabla"] = df_editado

if st.button("Optimizar plan de corte", type="primary", use_container_width=True):
    try:
        with st.spinner("Optimizando patrones por diámetro..."):
            st.session_state["resultado"] = optimizar(st.session_state["datos_tabla"], barra_m, kerf_cm, minimo_m)
            st.session_state["config_resultado"] = (barra_m, kerf_cm, minimo_m)
    except Exception as exc:
        st.error(f"No fue posible optimizar: {exc}")

if "resultado" in st.session_state:
    resultado = st.session_state["resultado"]
    st.subheader("Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Barras comerciales", int(resultado["resumen"]["Barras a comprar"].sum()))
    c2.metric("Acero requerido", f"{resultado['kg_total']:.2f} kg")
    c3.metric("Aprovechamiento global", f"{resultado['aprovechamiento']:.2f}%")
    c4.metric("Chatarra generada", f"{resultado['kg_chatarra']:.2f} kg")
    st.subheader("Resumen por diámetro"); st.dataframe(formatear(resultado["resumen"]), use_container_width=True, hide_index=True)
    st.subheader("Plan de corte agrupado"); st.dataframe(formatear(resultado["plan"]), use_container_width=True, hide_index=True)
    st.subheader("Inventario de sobrantes de stock"); st.dataframe(formatear(resultado["retazos"]), use_container_width=True, hide_index=True)
    st.download_button("Descargar reporte Excel", crear_excel(resultado, *st.session_state["config_resultado"]), "plan_corte_acero.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
