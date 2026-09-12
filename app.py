"""Optimizador de corte de barras de acero a partir de una lista de hierros."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from io import BytesIO
import math
import re
from typing import Any

import pandas as pd
import pdfplumber
import streamlit as st
from ortools.linear_solver import pywraplp


COLUMNAS = ["Posición", "Diámetro (mm)", "Cantidad", "Longitud (m)"]
PATRON_NUMERO = re.compile(r"-?\d+(?:[.,]\d+)?")


def a_numero(valor: Any) -> float | None:
    """Convierte números de planos, incluidos formatos 1.200,50 y 1,20."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return None
    texto = str(valor).strip().replace(" ", "")
    if not texto:
        return None
    texto = texto.replace("Ø", "").replace("ø", "").replace("∅", "")
    encontrado = PATRON_NUMERO.search(texto)
    if not encontrado:
        return None
    numero = encontrado.group(0)
    if "," in numero and "." in numero:
        # El último separador suele ser el decimal.
        if numero.rfind(",") > numero.rfind("."):
            numero = numero.replace(".", "").replace(",", ".")
        else:
            numero = numero.replace(",", "")
    else:
        numero = numero.replace(",", ".")
    try:
        return float(numero)
    except ValueError:
        return None


def normalizar_longitud(valor: Any) -> float | None:
    n = a_numero(valor)
    if n is None or n <= 0:
        return None
    # En planos de armaduras la longitud suele figurar en cm o mm. Se conserva
    # explícitamente la unidad cuando está indicada; sin unidad, 100+ es mm y
    # 10--100 se interpreta como cm, que es el formato más habitual.
    texto = str(valor).lower().replace(" ", "")
    if "mm" in texto or ("m" not in texto and n >= 100):
        return n / 1000
    if "cm" in texto or ("m" not in texto and 10 <= n < 100):
        return n / 100
    return n


def _encabezado_indice(celdas: list[Any], terminos: tuple[str, ...]) -> int | None:
    for i, celda in enumerate(celdas):
        texto = str(celda or "").lower()
        if any(t in texto for t in terminos):
            return i
    return None


def _filas_de_tablas(pagina: Any) -> list[list[Any]]:
    filas: list[list[Any]] = []
    for tabla in pagina.extract_tables():
        filas.extend(fila for fila in tabla if fila)
    return filas


def extraer_lista_hierros(pdf: Any) -> pd.DataFrame:
    """Extrae filas de tablas con posición, diámetro, cantidad y longitud.

    Acepta bytes o un objeto UploadedFile. El resultado siempre se puede editar;
    por ello se privilegia recuperar datos plausibles antes que descartar una fila
    por una cabecera imperfecta del PDF.
    """
    contenido = pdf if isinstance(pdf, bytes) else pdf.getvalue()
    registros: list[dict[str, Any]] = []
    with pdfplumber.open(BytesIO(contenido)) as documento:
        for pagina in documento.pages:
            filas = _filas_de_tablas(pagina)
            encabezados: list[Any] | None = None
            indices: tuple[int, int, int, int] | None = None
            for fila in filas:
                pos = _encabezado_indice(fila, ("pos", "marca", "nº", "no."))
                diam = _encabezado_indice(fila, ("diá", "diam", "ø", "φ", "∅"))
                cant = _encabezado_indice(fila, ("cant", "unid", "n°"))
                largo = _encabezado_indice(fila, ("long", "largo", "length"))
                if None not in (pos, diam, cant, largo):
                    encabezados, indices = fila, (pos, diam, cant, largo)
                    continue
                if indices is None:
                    continue
                try:
                    p, d, c, l = (fila[i] for i in indices)
                except IndexError:
                    continue
                diametro, cantidad, longitud = a_numero(d), a_numero(c), normalizar_longitud(l)
                if p and diametro and cantidad and longitud:
                    registros.append({
                        "Posición": str(p).strip(), "Diámetro (mm)": diametro,
                        "Cantidad": int(round(cantidad)), "Longitud (m)": longitud,
                    })

            # Respaldo para PDFs en los que la tabla se extrajo como texto plano.
            if not registros and not filas:
                for linea in (pagina.extract_text() or "").splitlines():
                    partes = re.split(r"\s+", linea.strip())
                    if len(partes) < 4:
                        continue
                    diam_i = next((i for i, x in enumerate(partes) if re.search(r"[Øø∅]", x)), None)
                    if diam_i is None or diam_i == 0 or diam_i + 2 >= len(partes):
                        continue
                    d, c, l = a_numero(partes[diam_i]), a_numero(partes[diam_i + 1]), normalizar_longitud(partes[diam_i + 2])
                    if d and c and l:
                        registros.append({"Posición": partes[0], "Diámetro (mm)": d,
                                          "Cantidad": int(round(c)), "Longitud (m)": l})
    if not registros:
        return pd.DataFrame(columns=COLUMNAS)
    resultado = pd.DataFrame(registros, columns=COLUMNAS).drop_duplicates()
    return resultado.sort_values(["Diámetro (mm)", "Posición"]).reset_index(drop=True)


def generar_patrones(longitudes: list[float], demandas: list[int], barra: float, kerf: float,
                     limite: int = 25000) -> list[tuple[int, ...]]:
    """Enumera patrones viables. Cada pieza consume su longitud y un kerf."""
    usados = [x + kerf for x in longitudes]
    patrones: set[tuple[int, ...]] = set()
    n = len(longitudes)

    def agregar(vector: list[int]) -> None:
        if any(vector):
            patrones.add(tuple(vector))

    # Los patrones unitarios garantizan factibilidad aun si se alcanza el límite.
    for i, uso in enumerate(usados):
        if uso <= barra + 1e-9:
            v = [0] * n
            v[i] = 1
            agregar(v)

    def explorar(indice: int, restante: float, actual: list[int]) -> None:
        if len(patrones) >= limite:
            return
        if indice == n:
            agregar(actual)
            return
        maximo = min(demandas[indice], int((restante + 1e-9) // usados[indice]))
        for cantidad in range(maximo, -1, -1):
            actual.append(cantidad)
            explorar(indice + 1, restante - cantidad * usados[indice], actual)
            actual.pop()
            if len(patrones) >= limite:
                return

    explorar(0, barra, [])
    # Agrega patrones de llenado greedy desde cada tipo, útiles con gran variedad.
    for inicio in range(n):
        restante, v = barra, [0] * n
        for i in [inicio] + [j for j in range(n) if j != inicio]:
            cantidad = min(demandas[i], int((restante + 1e-9) // usados[i]))
            if cantidad:
                v[i] = cantidad
                restante -= cantidad * usados[i]
        agregar(v)
    return sorted(patrones, key=lambda p: (sum(p), p), reverse=True)


def resolver_diametro(datos: pd.DataFrame, barra: float, kerf: float, umbral: float) -> dict[str, Any]:
    agrupado = datos.groupby("Longitud (m)", sort=False)["Cantidad"].sum().sort_index(ascending=False)
    longitudes = [float(x) for x in agrupado.index]
    demandas = [int(x) for x in agrupado.values]
    if any(l + kerf > barra + 1e-9 for l in longitudes):
        invalidas = [l for l in longitudes if l + kerf > barra + 1e-9]
        raise ValueError(f"Hay piezas que no caben en una barra: {invalidas}")
    patrones = generar_patrones(longitudes, demandas, barra, kerf)
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise RuntimeError("OR-Tools no pudo inicializar SCIP ni CBC.")
    solver.SetTimeLimit(30000)
    variables = [solver.IntVar(0, solver.infinity(), f"patron_{i}") for i in range(len(patrones))]
    for i, demanda in enumerate(demandas):
        solver.Add(sum(p[i] * variables[j] for j, p in enumerate(patrones)) == demanda)
    objetivo = solver.Objective()
    for variable in variables:
        objetivo.SetCoefficient(variable, 1)
    objetivo.SetMinimization()
    estado = solver.Solve()
    if estado not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("No se encontró una solución factible para este diámetro.")

    # Etiquetas de posiciones para que el taller pueda identificar cada pieza.
    etiquetas: dict[float, deque[str]] = defaultdict(deque)
    for _, fila in datos.sort_values(["Longitud (m)", "Posición"], ascending=[False, True]).iterrows():
        etiquetas[float(fila["Longitud (m)"])].extend([str(fila["Posición"])] * int(fila["Cantidad"]))
    cortes, retazos = [], []
    numero = 1
    for j, patron in enumerate(patrones):
        for _ in range(int(round(variables[j].solution_value()))):
            piezas: list[str] = []
            longitud_cortada = 0.0
            for i, cantidad in enumerate(patron):
                for _ in range(cantidad):
                    etiqueta = etiquetas[longitudes[i]].popleft()
                    piezas.append(f"{etiqueta}: {longitudes[i]:.3f} m")
                    longitud_cortada += longitudes[i]
            sobrante = max(0.0, barra - longitud_cortada - kerf * len(piezas))
            es_util = sobrante + 1e-9 >= umbral
            cortes.append({"Barra N°": numero, "Piezas a cortar": " | ".join(piezas),
                           "Longitud piezas (m)": longitud_cortada, "Kerf total (m)": kerf * len(piezas),
                           "Sobrante residual (m)": sobrante,
                           "Clasificación": "Retazo útil" if es_util else "Chatarra"})
            if es_util:
                retazos.append({"Barra origen": numero, "Longitud (m)": sobrante})
            numero += 1
    return {"cortes": cortes, "retazos": retazos, "barras": numero - 1,
            "metros_piezas": sum(l * q for l, q in zip(longitudes, demandas)),
            "metros_chatarra": sum(x["Sobrante residual (m)"] for x in cortes if x["Clasificación"] == "Chatarra")}


def optimizar(datos: pd.DataFrame, barra: float, kerf_cm: float, umbral: float) -> dict[str, Any]:
    kerf = kerf_cm / 100
    cortes, retazos, resumen = [], [], []
    for diametro, grupo in datos.groupby("Diámetro (mm)", sort=True):
        solucion = resolver_diametro(grupo, barra, kerf, umbral)
        for fila in solucion["cortes"]:
            fila["Diámetro (mm)"] = float(diametro)
            cortes.append(fila)
        for fila in solucion["retazos"]:
            fila["Diámetro (mm)"] = float(diametro)
            retazos.append(fila)
        kg_piezas = solucion["metros_piezas"] * float(diametro) ** 2 / 162
        kg_chatarra = solucion["metros_chatarra"] * float(diametro) ** 2 / 162
        resumen.append({"Diámetro (mm)": float(diametro), "Barras a comprar": solucion["barras"],
                        "Metros de piezas": solucion["metros_piezas"], "Kg de piezas": kg_piezas,
                        "Kg de chatarra": kg_chatarra})
    df_resumen = pd.DataFrame(resumen)
    total_barras = int(df_resumen["Barras a comprar"].sum())
    total_piezas = float(df_resumen["Metros de piezas"].sum())
    return {"resumen": df_resumen, "cortes": pd.DataFrame(cortes), "retazos": pd.DataFrame(retazos),
            "aprovechamiento": 100 * total_piezas / (total_barras * barra) if total_barras else 0,
            "kg_total": float(df_resumen["Kg de piezas"].sum()),
            "kg_chatarra": float(df_resumen["Kg de chatarra"].sum())}


def crear_excel(resultado: dict[str, Any], barra: float, kerf_cm: float, umbral: float) -> bytes:
    salida = BytesIO()
    with pd.ExcelWriter(salida, engine="xlsxwriter") as escritor:
        libro = escritor.book
        titulo = libro.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF"})
        numero = libro.add_format({"num_format": "0.000"})
        general = resultado["resumen"].copy()
        general.loc[len(general)] = {"Diámetro (mm)": "TOTAL", "Barras a comprar": int(general["Barras a comprar"].sum()),
                                     "Metros de piezas": general["Metros de piezas"].sum(), "Kg de piezas": resultado["kg_total"],
                                     "Kg de chatarra": resultado["kg_chatarra"]}
        general.to_excel(escritor, sheet_name="Resumen General", index=False, startrow=5)
        hoja = escritor.sheets["Resumen General"]
        hoja.write("A1", "Configuración", titulo)
        hoja.write_row("A2", ["Longitud barra (m)", barra, "Kerf (cm)", kerf_cm, "Umbral retazo (m)", umbral])
        hoja.write_row("A3", ["Aprovechamiento global", resultado["aprovechamiento"] / 100, "Kg totales", resultado["kg_total"], "Kg chatarra", resultado["kg_chatarra"]])
        hoja.set_column("A:A", 18); hoja.set_column("B:E", 18, numero); hoja.set_column("F:F", 18)
        for col, nombre in enumerate(general.columns): hoja.write(5, col, nombre, titulo)
        resultado["cortes"].to_excel(escritor, sheet_name="Plan de Corte Detallado", index=False)
        detalle = escritor.sheets["Plan de Corte Detallado"]
        detalle.set_column("A:A", 13); detalle.set_column("B:B", 16); detalle.set_column("C:C", 65); detalle.set_column("D:F", 22, numero)
        for col, nombre in enumerate(resultado["cortes"].columns): detalle.write(0, col, nombre, titulo)
        inventario = resultado["retazos"]
        if inventario.empty:
            inventario = pd.DataFrame(columns=["Diámetro (mm)", "Barra origen", "Longitud (m)"])
        inventario.to_excel(escritor, sheet_name="Inventario Retazos Útiles", index=False)
        inv = escritor.sheets["Inventario Retazos Útiles"]; inv.set_column("A:C", 22, numero)
        for col, nombre in enumerate(inventario.columns): inv.write(0, col, nombre, titulo)
    return salida.getvalue()


def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errores: list[str] = []
    datos = df.copy()
    for col in ("Diámetro (mm)", "Cantidad", "Longitud (m)"):
        datos[col] = pd.to_numeric(datos[col], errors="coerce")
    datos["Posición"] = datos["Posición"].fillna("").astype(str).str.strip()
    datos = datos.dropna(subset=["Diámetro (mm)", "Cantidad", "Longitud (m)"])
    datos = datos[(datos["Diámetro (mm)"] > 0) & (datos["Cantidad"] > 0) & (datos["Longitud (m)"] > 0)]
    if datos.empty:
        errores.append("Agrega al menos una fila válida con posición, diámetro, cantidad y longitud.")
    if not datos.empty:
        datos["Cantidad"] = datos["Cantidad"].round().astype(int)
        datos["Diámetro (mm)"] = datos["Diámetro (mm)"].round(3)
        datos["Longitud (m)"] = datos["Longitud (m)"].round(4)
        datos.loc[datos["Posición"] == "", "Posición"] = [f"Sin posición {i + 1}" for i in range((datos["Posición"] == "").sum())]
    return datos[COLUMNAS], errores


st.set_page_config(page_title="Optimizador de corte de acero", layout="wide")
st.title("Optimizador de corte de varillas de acero")
st.caption("Carga una lista de hierros desde un plano PDF, corrígela si hace falta y genera el plan de corte por diámetro.")

with st.sidebar:
    st.header("Configuración")
    longitud_barra = st.number_input("Longitud de barra (m)", min_value=0.1, value=12.0, step=0.1)
    kerf = st.number_input("Desgaste del disco / kerf (cm)", min_value=0.0, value=0.0, step=0.1)
    umbral = st.number_input("¿Desde qué longitud un sobrante es Retazo Útil? (m)", min_value=0.0, value=0.50, step=0.05)
    archivo = st.file_uploader("Plano en PDF", type=["pdf"])

if "datos" not in st.session_state:
    st.session_state.datos = pd.DataFrame(columns=COLUMNAS)
if archivo is not None:
    identificador = (archivo.name, archivo.size)
    if st.session_state.get("pdf_actual") != identificador:
        try:
            st.session_state.datos = extraer_lista_hierros(archivo)
            st.session_state.pdf_actual = identificador
            if st.session_state.datos.empty:
                st.warning("No se detectaron filas automáticamente. Completa la tabla manualmente.")
            else:
                st.success(f"Se extrajeron {len(st.session_state.datos)} filas. Verifica los valores antes de optimizar.")
        except Exception as exc:
            st.session_state.datos = pd.DataFrame(columns=COLUMNAS)
            st.error(f"No se pudo leer el PDF: {exc}. Puedes cargar la lista manualmente.")

st.subheader("Lista de hierros")
editado = st.data_editor(st.session_state.datos, num_rows="dynamic", use_container_width=True,
                         column_config={"Posición": st.column_config.TextColumn(required=True),
                                        "Diámetro (mm)": st.column_config.NumberColumn(min_value=0.1, format="%.3f"),
                                        "Cantidad": st.column_config.NumberColumn(min_value=1, step=1),
                                        "Longitud (m)": st.column_config.NumberColumn(min_value=0.001, format="%.4f")},
                         key="editor_hierros")
st.session_state.datos = editado

if st.button("Optimizar plan de corte", type="primary", use_container_width=True):
    datos_validos, errores = limpiar_datos(editado)
    if longitud_barra <= 0 or umbral < 0:
        errores.append("La longitud de barra y el umbral deben ser válidos.")
    if errores:
        for error in errores: st.error(error)
    else:
        try:
            with st.spinner("Calculando patrones óptimos por diámetro..."):
                st.session_state.resultado = optimizar(datos_validos, longitud_barra, kerf, umbral)
                st.session_state.config_resultado = (longitud_barra, kerf, umbral)
        except Exception as exc:
            st.error(f"No fue posible optimizar: {exc}")

if "resultado" in st.session_state:
    resultado = st.session_state.resultado
    st.subheader("Resultado")
    a, b, c = st.columns(3)
    a.metric("Barras a comprar", int(resultado["resumen"]["Barras a comprar"].sum()))
    b.metric("Aprovechamiento global", f"{resultado['aprovechamiento']:.2f}%")
    c.metric("Chatarra estimada", f"{resultado['kg_chatarra']:.2f} kg")
    st.dataframe(resultado["resumen"], use_container_width=True, hide_index=True)
    st.subheader("Plan de corte detallado")
    st.dataframe(resultado["cortes"], use_container_width=True, hide_index=True)
    st.subheader("Inventario de retazos útiles")
    st.dataframe(resultado["retazos"], use_container_width=True, hide_index=True)
    cfg = st.session_state.config_resultado
    st.download_button("Descargar reporte Excel", data=crear_excel(resultado, *cfg),
                       file_name="plan_corte_acero.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
