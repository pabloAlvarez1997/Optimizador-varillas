"""Aplicación Streamlit para optimizar el corte de barras de acero por diámetro."""
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


# Pesos lineales teóricos de acero, en kg/m. Esta es la única fuente de peso.
PESOS_NOMINALES = {
    6: 0.222, 8: 0.395, 10: 0.617, 12: 0.888,
    16: 1.578, 20: 2.466, 25: 3.853, 32: 6.313,
}
COLUMNAS = ["Posición", "Diámetro (mm)", "Cantidad", "Longitud (m)"]
TOLERANCIA_CORTE_M = 0.01
NUMERO_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def convertir_numero(valor: Any) -> float | None:
    """Convierte formatos de planos, incluidos 1.200,50 y 1,20."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return None
    coincidencia = NUMERO_RE.search(str(valor).replace(" ", ""))
    if not coincidencia:
        return None
    texto = coincidencia.group(0)
    if "," in texto and "." in texto:
        texto = (texto.replace(".", "").replace(",", ".")
                 if texto.rfind(",") > texto.rfind(".") else texto.replace(",", ""))
    else:
        texto = texto.replace(",", ".")
    try:
        return float(texto)
    except ValueError:
        return None


def convertir_longitud_m(valor: Any) -> float | None:
    """Normaliza longitudes de PDF/DXF en m, cm o mm a metros."""
    numero = convertir_numero(valor)
    if numero is None or numero <= 0:
        return None
    texto = str(valor).lower().replace(" ", "")
    if "mm" in texto or ("m" not in texto and numero >= 100):
        return numero / 1000
    if "cm" in texto or ("m" not in texto and 10 <= numero < 100):
        return numero / 100
    return numero


def peso_nominal(diametro: float) -> float:
    """Devuelve solo el peso de tabla para un diámetro admitido."""
    for diametro_tabla, peso in PESOS_NOMINALES.items():
        if abs(float(diametro) - diametro_tabla) < 1e-8:
            return peso
    admitidos = ", ".join(str(x) for x in PESOS_NOMINALES)
    raise ValueError(f"Ø {diametro:g} mm no tiene peso nominal. Admitidos: {admitidos} mm.")


def indice_columna(celdas: list[Any], terminos: tuple[str, ...]) -> int | None:
    for indice, celda in enumerate(celdas):
        texto = str(celda or "").lower()
        if any(termino in texto for termino in terminos):
            return indice
    return None


def es_encabezado(celdas: list[Any]) -> tuple[int, int, int, int] | None:
    posicion = indice_columna(celdas, ("pos", "marca", "item"))
    diametro = indice_columna(celdas, ("diá", "diam", "ø", "φ", "∅"))
    cantidad = indice_columna(celdas, ("cant", "unid", "qty", "quant"))
    longitud = indice_columna(celdas, ("long", "largo", "length"))
    if None in (posicion, diametro, cantidad, longitud):
        return None
    return posicion, diametro, cantidad, longitud


def registros_desde_filas(filas: Iterable[list[Any]]) -> list[dict[str, Any]]:
    registros: list[dict[str, Any]] = []
    encabezado: tuple[int, int, int, int] | None = None
    for fila in filas:
        fila = list(fila)
        encontrado = es_encabezado(fila)
        if encontrado is not None:
            encabezado = encontrado
            continue
        if encabezado is None:
            continue
        try:
            pos, diam, cant, largo = (fila[i] for i in encabezado)
        except IndexError:
            continue
        d, c, l = convertir_numero(diam), convertir_numero(cant), convertir_longitud_m(largo)
        if pos not in (None, "") and d and c and l:
            registros.append({"Posición": str(pos).strip(), "Diámetro (mm)": d,
                              "Cantidad": int(round(c)), "Longitud (m)": l})
    return registros


def registros_desde_texto(texto: str) -> list[dict[str, Any]]:
    """Respaldo para planillas que el origen expone como texto plano."""
    registros: list[dict[str, Any]] = []
    for linea in texto.splitlines():
        partes = [p for p in re.split(r"\s+", linea.strip()) if p]
        if len(partes) < 4:
            continue
        i_diametro = next((i for i, p in enumerate(partes) if re.search(r"[Øø∅φΦ]", p)), 1)
        if i_diametro == 0 or i_diametro + 2 >= len(partes):
            continue
        d = convertir_numero(partes[i_diametro])
        c = convertir_numero(partes[i_diametro + 1])
        l = convertir_longitud_m(partes[i_diametro + 2])
        if d and c and l and c.is_integer() and c > 0:
            registros.append({"Posición": partes[0], "Diámetro (mm)": d,
                              "Cantidad": int(c), "Longitud (m)": l})
    return registros


def normalizar_registros(registros: list[dict[str, Any]]) -> pd.DataFrame:
    if not registros:
        return pd.DataFrame(columns=COLUMNAS)
    resultado = pd.DataFrame(registros, columns=COLUMNAS).drop_duplicates()
    return resultado.sort_values(["Diámetro (mm)", "Posición"]).reset_index(drop=True)


def extraer_lista_hierros_pdf(contenido: bytes) -> pd.DataFrame:
    registros: list[dict[str, Any]] = []
    texto_completo: list[str] = []
    with pdfplumber.open(BytesIO(contenido)) as documento:
        for pagina in documento.pages:
            filas: list[list[Any]] = []
            for tabla in pagina.extract_tables():
                filas.extend(fila for fila in tabla if fila)
            registros.extend(registros_desde_filas(filas))
            texto_completo.append(pagina.extract_text() or "")
    if not registros:
        registros = registros_desde_texto("\n".join(texto_completo))
    return normalizar_registros(registros)


def _filas_dxf_por_coordenada(contenido: bytes) -> tuple[list[list[str]], str]:
    """Construye filas visuales a partir de entidades TEXT y MTEXT de un DXF."""
    try:
        documento = ezdxf.read(StringIO(contenido.decode("utf-8", errors="ignore")))
    except Exception as exc:
        raise ValueError(f"El archivo DXF no es legible: {exc}") from exc
    elementos: list[tuple[float, float, str]] = []
    for entidad in documento.modelspace():
        if entidad.dxftype() == "TEXT":
            texto = entidad.dxf.text
        elif entidad.dxftype() == "MTEXT":
            texto = entidad.plain_text()
        else:
            continue
        texto = str(texto).replace("\\P", " ").strip()
        if texto:
            punto = entidad.dxf.insert
            elementos.append((float(punto.y), float(punto.x), texto))
    if not elementos:
        return [], ""
    elementos.sort(key=lambda e: (-e[0], e[1]))
    grupos: list[list[tuple[float, float, str]]] = []
    for elemento in elementos:
        if not grupos or abs(grupos[-1][0][0] - elemento[0]) > 2.0:
            grupos.append([elemento])
        else:
            grupos[-1].append(elemento)
    filas = [[texto for _, _, texto in sorted(grupo, key=lambda e: e[1])] for grupo in grupos]
    return filas, "\n".join(" ".join(fila) for fila in filas)


def extraer_lista_hierros_dxf(contenido: bytes) -> pd.DataFrame:
    filas, texto = _filas_dxf_por_coordenada(contenido)
    registros = registros_desde_filas(filas)
    if not registros:
        registros = registros_desde_texto(texto)
    return normalizar_registros(registros)


def extraer_lista_hierros(archivo: Any) -> pd.DataFrame:
    """Extrae la lista de hierros desde un UploadedFile PDF o DXF."""
    nombre, contenido = archivo.name.lower(), archivo.getvalue()
    if nombre.endswith(".pdf"):
        return extraer_lista_hierros_pdf(contenido)
    if nombre.endswith(".dxf"):
        return extraer_lista_hierros_dxf(contenido)
    raise ValueError("Formato no admitido. Carga un archivo PDF o DXF.")


def generar_patrones(longitudes: list[float], demandas: list[int], longitud_barra: float,
                     kerf_m: float, limite: int = 25000) -> list[tuple[int, ...]]:
    """Genera patrones que permiten una diferencia acumulada máxima de 1 cm."""
    consumos = [longitud + kerf_m for longitud in longitudes]
    patrones: set[tuple[int, ...]] = set()
    n = len(longitudes)

    def agregar(patron: list[int]) -> None:
        if any(patron):
            patrones.add(tuple(patron))

    # Patrones unitarios: factibilidad garantizada ante una enumeración limitada.
    for i, consumo in enumerate(consumos):
        if consumo <= longitud_barra + TOLERANCIA_CORTE_M:
            patron = [0] * n
            patron[i] = 1
            agregar(patron)

    def recorrer(indice: int, restante: float, actual: list[int]) -> None:
        if len(patrones) >= limite:
            return
        if indice == n:
            agregar(actual)
            return
        maximo = min(demandas[indice], int((restante + TOLERANCIA_CORTE_M) // consumos[indice]))
        for cantidad in range(maximo, -1, -1):
            actual.append(cantidad)
            recorrer(indice + 1, restante - cantidad * consumos[indice], actual)
            actual.pop()
            if len(patrones) >= limite:
                return

    recorrer(0, longitud_barra, [])
    # Añade patrones densos cuando se corta la enumeración por el límite.
    for inicio in range(n):
        restante, patron = longitud_barra, [0] * n
        for i in [inicio] + [x for x in range(n) if x != inicio]:
            cantidad = min(demandas[i], int((restante + TOLERANCIA_CORTE_M) // consumos[i]))
            if cantidad:
                patron[i] = cantidad
                restante -= cantidad * consumos[i]
        agregar(patron)
    return sorted(patrones, key=lambda p: (sum(p), p), reverse=True)


def resolver_diametro(datos: pd.DataFrame, longitud_barra: float, kerf_m: float,
                      minimo_reutilizable: float) -> dict[str, Any]:
    demanda = datos.groupby("Longitud (m)", sort=False)["Cantidad"].sum().sort_index(ascending=False)
    longitudes, demandas = [float(x) for x in demanda.index], [int(x) for x in demanda.values]
    no_caben = [x for x in longitudes if x + kerf_m > longitud_barra + TOLERANCIA_CORTE_M]
    if no_caben:
        raise ValueError(f"Piezas que no caben en una barra incluso con tolerancia: {no_caben}")
    patrones = generar_patrones(longitudes, demandas, longitud_barra, kerf_m)
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError("No fue posible iniciar el solver SCIP/CBC de OR-Tools.")
    solver.SetTimeLimit(30_000)
    variables = [solver.IntVar(0, solver.infinity(), f"patron_{i}") for i in range(len(patrones))]
    for tipo, cantidad in enumerate(demandas):
        solver.Add(sum(patron[tipo] * variables[i] for i, patron in enumerate(patrones)) == cantidad)
    objetivo = solver.Objective()
    for variable in variables:
        objetivo.SetCoefficient(variable, 1)
    objetivo.SetMinimization()
    estado = solver.Solve()
    if estado not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("El solver no encontró un plan de corte factible.")

    etiquetas: dict[float, deque[str]] = defaultdict(deque)
    for _, fila in datos.sort_values(["Longitud (m)", "Posición"], ascending=[False, True]).iterrows():
        etiquetas[float(fila["Longitud (m)"])].extend([str(fila["Posición"])] * int(fila["Cantidad"]))
    cortes: list[dict[str, Any]] = []
    retazos: list[dict[str, Any]] = []
    numero_barra = 1
    for i, patron in enumerate(patrones):
        for _ in range(int(round(variables[i].solution_value()))):
            piezas, metros_piezas = [], 0.0
            for tipo, cantidad in enumerate(patron):
                for _ in range(cantidad):
                    piezas.append(f"{etiquetas[longitudes[tipo]].popleft()}: {longitudes[tipo]:.2f} m")
                    metros_piezas += longitudes[tipo]
            sobrante_fisico = longitud_barra - metros_piezas - kerf_m * len(piezas)
            # Un déficit máximo de 1 cm es tolerancia admisible, no sobrante negativo.
            sobrante = max(0.0, sobrante_fisico)
            clasificacion = ("Sobrante de Stock" if sobrante + 1e-9 >= minimo_reutilizable
                             else "Desperdicio / Chatarra")
            cortes.append({"Barra N°": numero_barra, "Piezas a cortar": " | ".join(piezas),
                           "Metros de piezas": metros_piezas, "Kerf total (m)": kerf_m * len(piezas),
                           "Sobrante (m)": sobrante, "Clasificación": clasificacion})
            if clasificacion == "Sobrante de Stock":
                retazos.append({"Barra origen": numero_barra, "Longitud (m)": sobrante})
            numero_barra += 1
    return {"barras": numero_barra - 1, "cortes": cortes, "retazos": retazos,
            "metros_piezas": sum(largo * cantidad for largo, cantidad in zip(longitudes, demandas)),
            "metros_chatarra": sum(c["Sobrante (m)"] for c in cortes
                                    if c["Clasificación"] == "Desperdicio / Chatarra")}


def optimizar(datos: pd.DataFrame, longitud_barra: float, kerf_cm: float,
              minimo_reutilizable: float) -> dict[str, Any]:
    kerf_m = kerf_cm / 100
    resumen, cortes, retazos = [], [], []
    for diametro, grupo in datos.groupby("Diámetro (mm)", sort=True):
        peso = peso_nominal(float(diametro))
        solucion = resolver_diametro(grupo, longitud_barra, kerf_m, minimo_reutilizable)
        for corte in solucion["cortes"]:
            corte["Diámetro (mm)"] = float(diametro)
            cortes.append(corte)
        for retazo in solucion["retazos"]:
            retazo["Diámetro (mm)"] = float(diametro)
            retazos.append(retazo)
        resumen.append({"Diámetro (mm)": float(diametro), "Barras a comprar": solucion["barras"],
                        "Metros de piezas": solucion["metros_piezas"],
                        "Kg de piezas": solucion["metros_piezas"] * peso,
                        "Kg de chatarra": solucion["metros_chatarra"] * peso})
    df_resumen = pd.DataFrame(resumen)
    total_barras = int(df_resumen["Barras a comprar"].sum())
    metros_piezas = float(df_resumen["Metros de piezas"].sum())
    return {"resumen": df_resumen, "cortes": pd.DataFrame(cortes), "retazos": pd.DataFrame(retazos),
            "aprovechamiento": 100 * metros_piezas / (total_barras * longitud_barra) if total_barras else 0.0,
            "kg_total": float(df_resumen["Kg de piezas"].sum()),
            "kg_chatarra": float(df_resumen["Kg de chatarra"].sum())}


def redondear_para_mostrar(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    enteros = {"Barra N°", "Barras a comprar", "Barra origen"}
    for columna in resultado.select_dtypes(include="number").columns:
        if columna not in enteros:
            resultado[columna] = resultado[columna].round(2)
    return resultado


def crear_excel(resultado: dict[str, Any], longitud_barra: float, kerf_cm: float,
                minimo_reutilizable: float) -> bytes:
    """Genera las tres hojas solicitadas, con formato numérico de dos decimales."""
    salida = BytesIO()
    with pd.ExcelWriter(salida, engine="xlsxwriter") as escritor:
        libro = escritor.book
        encabezado = libro.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF"})
        dos_decimales = libro.add_format({"num_format": "0.00"})
        porcentaje = libro.add_format({"num_format": "0.00%"})
        resumen = resultado["resumen"].copy()
        total = {"Diámetro (mm)": "TOTAL", "Barras a comprar": int(resumen["Barras a comprar"].sum()),
                 "Metros de piezas": resumen["Metros de piezas"].sum(), "Kg de piezas": resultado["kg_total"],
                 "Kg de chatarra": resultado["kg_chatarra"]}
        resumen = pd.concat([resumen, pd.DataFrame([total])], ignore_index=True)
        resumen.to_excel(escritor, sheet_name="Resumen General", index=False, startrow=5)
        hoja = escritor.sheets["Resumen General"]
        hoja.write("A1", "Configuración", encabezado)
        hoja.write_row("A2", ["Longitud barra (m)", longitud_barra, "Kerf (cm)", kerf_cm,
                                "Longitud mínima reutilizable (m)", minimo_reutilizable])
        hoja.write("A3", "Aprovechamiento global", encabezado)
        hoja.write_number("B3", resultado["aprovechamiento"] / 100, porcentaje)
        hoja.write("D3", "Kg totales", encabezado); hoja.write_number("E3", resultado["kg_total"], dos_decimales)
        hoja.write("F3", "Kg chatarra", encabezado); hoja.write_number("G3", resultado["kg_chatarra"], dos_decimales)
        for i, nombre in enumerate(resumen.columns): hoja.write(5, i, nombre, encabezado)
        hoja.set_column("A:A", 18); hoja.set_column("B:E", 20, dos_decimales)

        detalle = resultado["cortes"].copy()
        detalle.to_excel(escritor, sheet_name="Plan de Corte Detallado", index=False)
        hoja = escritor.sheets["Plan de Corte Detallado"]
        for i, nombre in enumerate(detalle.columns): hoja.write(0, i, nombre, encabezado)
        hoja.set_column("A:B", 16); hoja.set_column("C:C", 65); hoja.set_column("D:G", 22, dos_decimales)

        inventario = resultado["retazos"].copy()
        if inventario.empty:
            inventario = pd.DataFrame(columns=["Diámetro (mm)", "Barra origen", "Longitud (m)"])
        inventario.to_excel(escritor, sheet_name="Inventario Sobrantes Stock", index=False)
        hoja = escritor.sheets["Inventario Sobrantes Stock"]
        for i, nombre in enumerate(inventario.columns): hoja.write(0, i, nombre, encabezado)
        hoja.set_column("A:C", 24, dos_decimales)
    return salida.getvalue()


def validar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errores: list[str] = []
    datos = df.copy()
    for columna in ("Diámetro (mm)", "Cantidad", "Longitud (m)"):
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
    datos["Posición"] = datos["Posición"].fillna("").astype(str).str.strip()
    datos = datos.dropna(subset=["Diámetro (mm)", "Cantidad", "Longitud (m)"])
    datos = datos[(datos["Diámetro (mm)"] > 0) & (datos["Cantidad"] > 0) & (datos["Longitud (m)"] > 0)]
    if datos.empty:
        return pd.DataFrame(columns=COLUMNAS), ["Ingresa al menos una fila válida antes de optimizar."]
    if not (datos["Cantidad"] % 1 == 0).all():
        errores.append("La cantidad de cada posición debe ser un número entero.")
    datos["Cantidad"] = datos["Cantidad"].round().astype(int)
    datos["Diámetro (mm)"] = datos["Diámetro (mm)"].round(8)
    datos["Longitud (m)"] = datos["Longitud (m)"].round(4)
    for diametro in sorted(datos["Diámetro (mm)"].unique()):
        try: peso_nominal(float(diametro))
        except ValueError as exc: errores.append(str(exc))
    vacias = datos["Posición"] == ""
    datos.loc[vacias, "Posición"] = [f"Sin posición {i + 1}" for i in range(int(vacias.sum()))]
    return datos[COLUMNAS], errores


st.set_page_config(page_title="Optimizador de corte de acero", layout="wide")
st.title("Optimizador de corte de varillas de acero")
st.caption("Carga una lista de hierros PDF o DXF, revísala y obtén el plan de corte separado por diámetro.")
with st.sidebar:
    st.header("Configuración")
    longitud_barra = st.number_input("Longitud de barra (m)", min_value=0.10, value=12.00, step=0.10, format="%.2f")
    kerf_cm = st.number_input("Desgaste del disco / kerf (cm)", min_value=0.00, value=0.00, step=0.10, format="%.2f")
    minimo_reutilizable = st.number_input("Longitud mínima reutilizable (m)", min_value=0.00, value=0.50, step=0.05, format="%.2f")
    archivo = st.file_uploader("Plano o planilla", type=["pdf", "dxf"])
    st.caption("Tolerancia de corte aplicada: ±0.01 m.")

if "datos_hierros" not in st.session_state:
    st.session_state.datos_hierros = pd.DataFrame(columns=COLUMNAS)
if archivo is not None:
    identificador = (archivo.name, archivo.size)
    if st.session_state.get("archivo_actual") != identificador:
        try:
            st.session_state.datos_hierros = extraer_lista_hierros(archivo)
            st.session_state.archivo_actual = identificador
            if st.session_state.datos_hierros.empty: st.warning("No se detectaron filas. Completa la tabla manualmente.")
            else: st.success(f"Se extrajeron {len(st.session_state.datos_hierros)} filas. Revísalas antes de optimizar.")
        except Exception as exc:
            st.session_state.datos_hierros = pd.DataFrame(columns=COLUMNAS)
            st.error(f"No se pudo extraer la lista: {exc}")

st.subheader("Lista de hierros")
editado = st.data_editor(
    st.session_state.datos_hierros, num_rows="dynamic", use_container_width=True, key="editor_lista_hierros",
    column_config={"Posición": st.column_config.TextColumn("Posición", required=True),
                   "Diámetro (mm)": st.column_config.NumberColumn("Diámetro Ø (mm)", min_value=0.01, format="%.2f"),
                   "Cantidad": st.column_config.NumberColumn("Cantidad", min_value=1, step=1, format="%d"),
                   "Longitud (m)": st.column_config.NumberColumn("Longitud (m)", min_value=0.01, format="%.2f")},
)
st.session_state.datos_hierros = editado
if st.button("Optimizar plan de corte", type="primary", use_container_width=True):
    datos, errores = validar_datos(editado)
    if longitud_barra <= 0: errores.append("La longitud de barra debe ser mayor que cero.")
    if errores:
        for error in errores: st.error(error)
    else:
        try:
            with st.spinner("Calculando patrones óptimos por diámetro..."):
                st.session_state.resultado = optimizar(datos, longitud_barra, kerf_cm, minimo_reutilizable)
                st.session_state.config_resultado = (longitud_barra, kerf_cm, minimo_reutilizable)
        except Exception as exc:
            st.error(f"No fue posible optimizar: {exc}")

if "resultado" in st.session_state:
    resultado = st.session_state.resultado
    st.subheader("Resultado")
    col1, col2, col3 = st.columns(3)
    col1.metric("Barras a comprar", int(resultado["resumen"]["Barras a comprar"].sum()))
    col2.metric("Aprovechamiento global", f"{resultado['aprovechamiento']:.2f}%")
    col3.metric("Chatarra estimada", f"{resultado['kg_chatarra']:.2f} kg")
    st.dataframe(redondear_para_mostrar(resultado["resumen"]), use_container_width=True, hide_index=True)
    st.subheader("Plan de corte detallado")
    st.dataframe(redondear_para_mostrar(resultado["cortes"]), use_container_width=True, hide_index=True)
    st.subheader("Inventario de sobrantes de stock")
    st.dataframe(redondear_para_mostrar(resultado["retazos"]), use_container_width=True, hide_index=True)
    st.download_button("Descargar reporte Excel", data=crear_excel(resultado, *st.session_state.config_resultado),
                       file_name="plan_corte_acero.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

