"""Crea los notebooks de notebooks/ (luego se ejecutan con nbconvert).

    python scripts/crear_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
INICIO = """import sys, os
from pathlib import Path
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(RAIZ)
sys.path.insert(0, str(RAIZ / "src"))
import pandas as pd
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)"""

NOTEBOOKS = {
    "01_datos_y_capa_de_kpis.ipynb": [
        ("md", "# 1 · Datos del Banco Mundial y capa de KPIs determinista\n\n"
               "La capa de KPIs es la **fuente de verdad**: todo número que aparezca en un informe tiene que "
               "existir aquí. Datos: World Development Indicators (API v2, sin clave), licencia CC BY 4.0. "
               "La instantánea `data/instantanea_wdi.csv` permite reproducir todo sin red."),
        ("code", INICIO + "\nfrom kpis import capa_kpis, config\nimport json\n"
                 "print(json.loads(config.METADATOS.read_text(encoding='utf-8'))['descargado'])\n"
                 "df = capa_kpis.cargar(); df.shape"),
        ("md", "## Cobertura: valores por indicador y año (últimos años)"),
        ("code", "df[df.anio >= 2019].pivot_table(index='indicador', columns='anio', values='valor', aggfunc='count')"),
        ("md", "Pobreza, Gini y deuda externa son escasos (no todos los países publican cada año, y la deuda "
               "externa de WDI no cubre a los países de ingreso alto). La capa usa el último dato de hasta "
               "3 años antes y lo dice en la tabla."),
        ("md", "## Tabla larga: variación interanual, mediana regional, brecha, ranking y anomalía"),
        ("code", "t = capa_kpis.tabla_larga()\n"
                 "t[(t.iso3 == 'ECU') & (t.anio == 2024)][['indicador', 'valor_previo', 'valor', 'variacion', "
                 "'unidad_variacion', 'mediana_region', 'brecha_region', 'ranking', 'n_paises', 'z_anomalia']].round(2)"),
        ("md", "Las variaciones de indicadores en % se expresan en **puntos porcentuales**; las de niveles "
               "(PIB en USD, población) en **% relativo**. El ranking es 1 = valor más alto."),
        ("md", "## Ranking regional de crecimiento 2024"),
        ("code", "capa_kpis.ranking_regional('crecimiento', 2024).round(2)"),
        ("md", "## Ecuador frente a la región"),
        ("code", "c = capa_kpis.comparacion_con_region('ECU', 2024)\n"
                 "c.assign(indicador=c.indicador.map(lambda k: config.IND_POR_CLAVE[k].nombre)).round(2)"),
        ("md", "![Ecuador vs región](../reports/figures/01_ecuador_vs_region.png)"),
        ("md", "## Anomalías: cambios fuera de lo normal\n\n"
               "z robusto = (variación − mediana histórica) / (1,4826 · MAD), usando **solo años anteriores** "
               "(sin mirar el futuro) y un piso de escala (0,5 pp o 2 %) para que series muy suaves, como la "
               "población, no marquen décimas como anomalías."),
        ("code", "a = capa_kpis.anomalias(desde=2022)\na.round(2)"),
        ("code", "a.groupby('anio').size()"),
        ("md", "![Anomalías](../reports/figures/02_mapa_anomalias.png)\n\n"
               "Los picos de 2009 (crisis financiera) y 2020–2021 (pandemia y rebote) aparecen solos: es una "
               "comprobación de que el detector marca lo que un analista esperaría."),
        ("md", "## Lo que recibe el LLM\n\nLa misma tabla, con las cifras ya formateadas en español."),
        ("code", "print(capa_kpis.tabla_para_prompt('ECU', 2024))"),
    ],
    "02_extractor_y_verificador.ipynb": [
        ("md", "# 2 · Extractor de cifras y verificador\n\n"
               "El verificador extrae cada número del texto, deduce de qué indicador, país y año habla (o usa la "
               "referencia estructurada), y lo compara con la capa de KPIs con tolerancia de redondeo y "
               "compatibilidad de unidades."),
        ("code", INICIO + "\nfrom kpis.numeros import extraer\nfrom kpis.verificador import verificar_texto"),
        ("md", "## Casos difíciles de formato"),
        ("code", "casos = ['bajó −1,9 %', 'una contracción del 3.5%', 'subió 2,6 puntos porcentuales',\n"
                 "         '6.827 USD', '6,718.14 dólares', '123,8 mil millones de USD', '121.000 millones de dólares',\n"
                 "         '18,1 millones de habitantes', 'US$ 1,2 billones', 'el tercer lugar en remesas', 'puesto 18 de 18',\n"
                 "         'En 2024 no hubo cambios']\n"
                 "pd.DataFrame([{'texto': s, 'cifras': [(c.texto, c.candidatos, c.unidad, c.decimales) for c in extraer(s)]}\n"
                 "              for s in casos])"),
        ("md", "Cuando el texto es ambiguo (`6,718`, `billones`), el extractor devuelve **varios candidatos** y el "
               "verificador acepta el que coincida. Los años sueltos no son cifras."),
        ("md", "## Verificación en contexto"),
        ("code", "frases = [\n"
                 "  'En 2024 la economía de Ecuador se contrajo 1,9 %, frente a una mediana regional de 3,4 %.',\n"
                 "  'La inflación fue de 2,2 % en 2024.',\n"
                 "  'La inflación bajó a 1,5 %, 0,7 puntos porcentuales menos que en 2023.',\n"
                 "  'Las remesas crecieron 0,8 % en 2024.',\n"
                 "  'La IED fue de 0,4 % del PIB, el tercer puesto más bajo.',\n"
                 "  'Se espera un repunte de 4,2 % el próximo año.',\n"
                 "  'El PIB creció 2,5 % en 2024.']\n"
                 "filas = []\n"
                 "for f in frases:\n"
                 "    for c in verificar_texto(f, 'ECU', 2024):\n"
                 "        filas.append({'frase': f[:60], 'cifra': c.texto, 'estado': c.estado, 'real': c.real_formateado,\n"
                 "                      'hecho': c.hecho_id, 'marcas': c.marcas})\n"
                 "pd.DataFrame(filas)"),
        ("md", "Dos casos para leer con calma:\n\n"
               "- «Las remesas crecieron 0,8 %» es **correcta pero mal expresada**: el cambio real es +0,8 pp. "
               "Se acepta y se marca `pp_escrito_como_%`.\n"
               "- «El PIB creció 2,5 % en 2024» se acepta porque el **PIB en dólares corrientes** subió 2,5 %, "
               "pero el PIB real cayó 1,9 %. El número existe; la frase engaña. Esto es una limitación de "
               "cualquier verificador de cifras: comprueba números, no la lectura que hace un gerente."),
    ],
    "03_evaluacion_de_enfoques.ipynb": [
        ("md", "# 3 · Evaluación de los cuatro enfoques\n\n"
               "(a) LLM libre · (b) LLM estructurado con referencias · (c) (b) + verificación y reintento · "
               "(d) plantilla determinista. Validación (5 informes, 2023) para ajustar el prompt; prueba "
               "(12 informes, 2024, otros países) corrida una sola vez con la versión elegida. Todo sale de las "
               "respuestas grabadas: este notebook no necesita Ollama."),
        ("code", INICIO + "\nimport json\nres = json.loads(open('reports/resultados.json', encoding='utf-8').read())\n"
                 "print('Versión de prompt elegida en validación:', res['version_prompt_elegida'])\n"
                 "pd.read_csv('reports/resumen_enfoques.csv')[['conjunto', 'enfoque', 'version', 'informes', 'cifras', "
                 "'tasa_alucinacion', 'tasa_alucinacion_ic', 'cobertura', 'cobertura_ic', 'informes_sin_error', "
                 "'segundos_por_informe']].round(3)"),
        ("md", "![Alucinación y cobertura](../reports/figures/03_alucinacion_y_cobertura.png)"),
        ("md", "## Diferencias pareadas en prueba (mismos 12 informes, bootstrap por informe)"),
        ("code", "pd.read_csv('reports/comparaciones_pareadas.csv').round(3)"),
        ("md", "## Qué tipo de errores comete cada enfoque"),
        ("code", "cif = pd.read_csv('reports/detalle_cifras.csv')\n"
                 "p = cif[cif.conjunto == 'prueba']\n"
                 "p.groupby(['enfoque', 'estado']).size().unstack(fill_value=0)"),
        ("code", "p[p.estado != 'correcta'][['enfoque', 'pais', 'seccion', 'cifra', 'estado', 'valor_real', "
                 "'oracion']].head(25)"),
        ("md", "![Tipos de error](../reports/figures/04_tipos_de_error.png)"),
        ("md", "## Reintentos de (c)"),
        ("code", "pd.read_csv('reports/reintentos.csv').query(\"conjunto == 'prueba'\")"),
        ("md", "![Reintentos](../reports/figures/05_reintentos.png)"),
        ("md", "## Ajuste del prompt (solo validación)"),
        ("md", "![Prompt](../reports/figures/06_mejora_prompt_validacion.png)"),
        ("md", "## Auditoría manual del verificador y legibilidad"),
        ("code", "aud = pd.read_csv('data/auditoria_verificador.csv')\n"
                 "pd.crosstab(aud.estado_verificador, aud.juicio_humano, margins=True)"),
        ("code", "pd.read_csv('data/rubrica_legibilidad.csv').merge(\n"
                 "    pd.DataFrame(json.loads(open('data/rubrica_mapa.json', encoding='utf-8').read())).T.reset_index()\n"
                 "    .rename(columns={'index': 'id'}), on='id').groupby('enfoque')[['claridad', 'concision', 'utilidad', 'tono']].mean().round(2)"),
        ("md", "![Legibilidad](../reports/figures/07_legibilidad_rubrica.png)\n\n"
               "![Tiempo](../reports/figures/08_tiempo_por_informe.png)"),
    ],
}

for nombre, celdas in NOTEBOOKS.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if t == "md" else nbf.v4.new_code_cell(c) for t, c in celdas]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    (RAIZ / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, RAIZ / "notebooks" / nombre)
    print(nombre)
