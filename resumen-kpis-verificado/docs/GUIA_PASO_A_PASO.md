# Guía paso a paso: resúmenes ejecutivos de KPIs con verificación de cada cifra

Esta guía explica cómo está construido el proyecto, en el orden en que se ejecuta, y termina con preguntas
de entrevista con su respuesta. Las cifras de resultados vienen de `reports/resultados.json`.

## 1. El problema

Gerencia quiere un párrafo, no una tabla: «¿qué pasó con la economía de este país el último año?». Un LLM
redacta bien ese párrafo, pero puede **inventar o equivocar cifras** y nadie en gerencia va a revisar 25
números contra la fuente. La idea del proyecto es separar dos trabajos:

- **Calcular** (capa de KPIs determinista, en `src/kpis/capa_kpis.py`): todas las cifras citables.
- **Redactar** (LLM local `qwen2.5:7b` en Ollama) y **comprobar** (verificador en `src/kpis/verificador.py`):
  cada número del texto se busca en la capa de KPIs.

## 2. Datos

- API v2 del Banco Mundial (World Development Indicators), sin clave: `python -m kpis.descargar`.
- 14 indicadores × 18 países de América Latina × 2000–2025. Cada respuesta cruda se guarda en `data/raw/`
  (caché, no se versiona) y el resultado limpio en `data/instantanea_wdi.csv` (sí se versiona, con la fecha de
  descarga en `data/metadatos_wdi.json`).
- Venezuela está en los datos (cuenta para la mediana y los rankings cuando tiene dato), pero no se evalúan
  informes de Venezuela: le faltan inflación, pobreza y Gini recientes.
- Licencia de los datos: CC BY 4.0 (citar «Banco Mundial, World Development Indicators»).

## 3. Capa de KPIs (la fuente de verdad)

Para cada indicador, país y año:

| Métrica | Cómo se calcula | Unidad |
|---|---|---|
| `valor` | dato de WDI | %, USD, personas o índice |
| `variacion` | año contra año anterior (solo si son consecutivos) | **pp** para tasas, **%** para niveles |
| `mediana_region` | mediana de los países con dato (mínimo 10) | la del indicador |
| `brecha_region` | valor − mediana (pp) o valor / mediana − 1 (%) | pp o % |
| `ranking` | 1 = valor más alto, «de N» países con dato | puesto |
| `anomalia` | \|z robusto\| ≥ 3 de la variación frente a los años **anteriores** del mismo país | sí/no |

Decisiones que importan:
- **pp frente a %.** Si la inflación pasa de 2,2 % a 1,5 %, el cambio es −0,7 pp, no −32 %. Mezclar las dos es
  el error de unidades más común en informes y el verificador lo detecta.
- **Anomalía sin mirar el futuro.** El z de 2024 usa solo la historia hasta 2023, con mediana y MAD (robustos a
  2009 y 2020) y un piso de escala (0,5 pp o 2 %) para que la población, que cambia en décimas, no salga
  «anómala» todos los años.
- **Datos escasos.** Pobreza, Gini y deuda externa usan el último dato de hasta 3 años antes, con su año real.

## 4. Extractor de cifras (`src/kpis/numeros.py`)

Convierte texto en cifras con valor, unidad, escala y decimales escritos. Casos que resuelve:
coma y punto decimal, separador de miles, menos tipográfico, «mil millones» / «millones» / «billones»,
%, pp, USD / US$ / dólares, habitantes, «puesto 3 de 18», ordinales («tercer lugar»). Ignora años sueltos,
marcadores de lista, el total del ranking («de 18») y las cifras que son parte del nombre del indicador
(«USD 3,00 al día», «PPA 2021», «0–100»). Si el texto es ambiguo (`6,718`; `billones`) devuelve varios
candidatos.

## 5. Verificador (`src/kpis/verificador.py`)

Para cada cifra:
1. Si la afirmación trae **referencias estructuradas** (indicador, país, año, métrica, valor), compara primero
   con esas cifras de la capa.
2. Si no, deduce el contexto de la oración: el indicador más cercano (por palabras clave), el país (el del
   informe, otro país nombrado o la región) y los años (los nombrados más el del informe).
3. **Correcta** si coincide con una cifra compatible en unidad dentro de la tolerancia de redondeo (media unidad
   del último dígito escrito: «3,5 %» acepta 3,45–3,55). **Incorrecta** si el indicador tiene dato y el número
   no coincide (se guarda el valor real). **Sin fuente** si no se puede atribuir a nada.
4. Marcas: `pp_escrito_como_%` (correcta en valor, mal en unidad), `atribucion_debil` (coincide sin indicador
   explícito en la oración).

Las referencias del JSON también se verifican aparte (`referencias_ok`), porque un modelo puede escribir bien la
cifra en el texto y mal en la referencia.

## 5 bis. Chequeo de dirección (`src/kpis/comparaciones.py`)

Una cifra correcta puede ir en una frase falsa. Este chequeo revisa, por cláusula, tres patrones: «por encima /
por debajo de la mediana» contra el signo de la brecha; «aumentó / disminuyó» contra el signo de la variación
(no se aplica al crecimiento del PIB, donde «creció» describe el nivel); y la variación del PIB en dólares
corrientes presentada como crecimiento. Si no identifica un único indicador, no opina. Se reporta como métrica
aparte y no participa en los reintentos.

## 6. Los cuatro enfoques (`src/kpis/generadores.py`)

| Enfoque | Qué recibe el LLM | Qué devuelve |
|---|---|---|
| (a) libre | tabla de KPIs + pedido de informe | Markdown |
| (b) estructurado | lo mismo + esquema pydantic en `format` de Ollama | JSON con referencia por cifra |
| (c) verificado | (b) y, si el verificador encuentra errores, un mensaje con cada error y su valor real; hasta 2 reintentos | JSON |
| (d) plantilla | — (sin LLM) | texto armado desde la capa |

La primera respuesta de (c) es exactamente la de (b) (mismo prompt, temperatura 0, semilla fija), así que la
diferencia entre (b) y (c) mide solo el efecto de verificar y reintentar.

## 7. Evaluación

- **Validación** (5 informes, 2023: Argentina, Chile, Perú, Guatemala, Paraguay): se compara el prompt `v1`
  con `v2` (reglas explícitas de cifras y formato de las referencias). Regla de elección: menor tasa de cifras
  alucinadas de (b); si empatan, mayor cobertura. Como `v2` dejó un informe sin cifras (0 % de alucinación
  trivial), se completó en validación: la versión debe promediar al menos 4 cifras correctas por informe.
- **Prueba** (12 informes, 2024, otros 12 países, Ecuador incluido): se corre una sola vez con la versión
  elegida.
- **Tasa de cifras alucinadas** = (incorrectas + sin fuente) / cifras extraídas.
- **Cobertura**: `data/hallazgos_referencia.json` tiene 4 hallazgos imprescindibles por informe, armados a mano
  antes de ver las salidas de prueba. Cuenta como cubierto si el titular, los hallazgos o los riesgos nombran
  su indicador.
- **IC 95 %**: bootstrap por informe (2.000 remuestreos). Se remuestrean informes completos porque las cifras de
  un mismo informe no son independientes.
- **Legibilidad**: rúbrica de 1 a 5 en cuatro criterios, puntuada a mano y a ciegas (ids barajados, sin saber
  el enfoque) en 6 informes de prueba × 4 enfoques:
  - *Claridad*: se entiende a la primera, sin jerga técnica ni claves internas.
  - *Concisión*: dice lo importante sin relleno ni repeticiones.
  - *Utilidad para decidir*: prioriza lo relevante y los riesgos o recomendaciones son accionables.
  - *Tono ejecutivo*: suena a informe para gerencia, no a volcado de tabla.
- **Auditoría del verificador**: todas las cifras de prueba que el verificador marcó como malas, más 40
  marcadas como correctas, revisadas a mano (`data/auditoria_verificador.csv`).

## 8. Cómo reproducir

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -e .
.venv/Scripts/python -m pytest -q -p no:cacheprovider
.venv/Scripts/python -m kpis.ejecutar_evaluacion
```

Todo lo anterior funciona sin red y sin Ollama. Para volver a generar las respuestas del LLM:
`ollama serve`, luego `KPIS_PROVEEDOR=ollama python -m kpis.correr_llm validacion`, `validacion_c` y `prueba`
(son reanudables: lo ya grabado no se repite).

## 9. Resultados en una tabla

| Enfoque | Cifras (por informe) | Alucinadas (IC 95 %) | Incorrectas | Sin fuente | Informes sin ningún error | Cobertura de hallazgos (IC 95 %) | Lecturas engañosas detectadas | Llamadas al LLM | Tiempo por informe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (a) LLM libre | 101 (8,4) | 4,0 % (1,1–6,8 %) | 4 | 0 | 66,7 % | 66,7 % (56,2–77,1 %) | 12 de 46 revisadas | 1,00 | 56 s |
| (b) LLM estructurado | 99 (8,2) | 6,1 % (0,0–14,1 %) | 6 | 0 | 83,3 % | 81,2 % (68,8–93,8 %) | 15 de 44 revisadas | 1,00 | 144 s |
| **(c) (b) + verificación y reintento** | 97 (8,1) | 3,1 % (0,0–7,9 %) | 3 | 0 | 91,7 % | 81,2 % (68,8–93,8 %) | 13 de 42 revisadas | 1,50 | 230 s |
| (d) Plantilla determinista | 288 (24,0) | 0,0 % (0,0–0,0 %) | 0 | 0 | 100,0 % | 64,6 % (54,2–72,9 %) | 0 de 24 revisadas | 0,00 | < 1 s |

Prompt elegido en validación: `v1`. Detalle e interpretación en el README.

## Preguntas de entrevista

### P1. ¿Por qué no basta con pedirle al LLM que «use solo las cifras de la tabla»?
Porque no hay forma de saber si lo hizo sin comprobarlo. En prueba, aun con la tabla en el prompt, el texto libre tuvo 4,0 % de cifras alucinadas y la versión estructurada 6,1 %. Es poco, pero en un informe para gerencia una sola cifra mala destruye la confianza, y nadie va a revisar 25 números a mano. El verificador convierte «confío en el modelo» en «sé cuáles cifras están bien».

### P2. ¿Cómo decide el verificador a qué indicador se refiere un número?
Primero con la referencia estructurada que trae la afirmación (indicador, país, año, métrica). Si no la hay o no coincide, mira la oración: la palabra clave de indicador más cercana (prefiere las que van antes del número), los países nombrados o «región», y los años. Compara solo con cifras de unidad compatible (% con %, pp con pp, USD con USD, puesto con ranking). Si la oración no nombra ningún indicador, busca en toda la tabla, pero solo si la cifra tiene decimales o escala, para no aceptar un «3» por casualidad.

### P3. ¿Qué tolerancia usas y por qué no un 5 %?
Media unidad del último dígito escrito: «3,5 %» acepta 3,45–3,55 y «123,8 mil millones» acepta ±0,05 mil millones. Es lo que un redondeo correcto permite. Un margen fijo del 5 % dejaría pasar 3,3 % cuando el dato es 3,47 %, o 6.500 USD cuando es 6.827.

### P4. La tasa de alucinación es baja en todos los enfoques. ¿Entonces el problema está resuelto?
No. El hallazgo más importante del proyecto es que el riesgo se movió de las cifras a las conclusiones: entre el 26,1 % y el 34,1 % de las lecturas cualitativas revisables estaban al revés o presentaban la variación del PIB en dólares como crecimiento («el PIB de Ecuador creció 2,5 %» con el PIB real cayendo 1,9 %). Cada número de esas frases existe en la tabla, así que un verificador de cifras las aprueba. Por eso agregué el chequeo de dirección.

### P5. ¿Por qué la salida estructurada no redujo la alucinación?
Porque la mayoría de errores no eran valores de indicador, sino puestos de ranking inventados y cálculos propios, que el modelo comete igual con o sin JSON. La estructura sí ayudó en cobertura (81,2 % frente a 66,7 %) y permite verificar cada referencia, pero tuvo costos: más tokens y tiempo, referencias mal escritas (solo 77,9 % correctas) y tuplas técnicas pegadas en el texto que bajan la legibilidad.

### P6. ¿Qué aprendiste del reintento con corrección?
Que funciona a medias. Subió los informes sin error de 83,3 % a 91,7 %, pero en Honduras el modelo «corrigió» el desempleo poniendo el dato del año anterior, que el verificador acepta. Un LLM que se corrige a sí mismo puede introducir errores nuevos; el reintento necesita un verificador más estricto con los años o una salida acotada (por ejemplo, elegir el id del hecho en vez de escribir el número).

### P7. ¿Cómo elegiste el prompt sin sobreajustar a la prueba?
Con un conjunto de validación separado (5 países en 2023) y una regla de elección fijada antes: menor tasa de alucinación de (b). Al ver que `v2` dejaba un informe sin ninguna cifra (0 % de alucinación trivial), completé la regla en validación: la versión debe promediar al menos 4 cifras correctas por informe. Con o sin ese cambio ganó `v1`. La prueba (12 países en 2024) se corrió una sola vez con `v1`. La única corrección que vio datos de prueba fue un fallo del extractor con la coma inglesa de miles, y lo declaro en el README.

### P8. ¿Por qué los intervalos de confianza son bootstrap por informe?
Porque las cifras de un informe no son independientes: si el modelo se confunde con Honduras, se equivoca en varias cifras a la vez. Remuestrear cifras sueltas daría IC demasiado estrechos. Remuestreando informes completos, los IC de la tasa de alucinación de los enfoques de LLM se solapan, y lo digo así: con 12 informes no puedo ordenar esos enfoques entre sí.

### P9. ¿Cómo sabes que el verificador funciona?
Con pruebas de casos difíciles (formatos, unidades, ordinales, ambigüedades) y con una auditoría manual en prueba: de 13 cifras marcadas como malas, 10 lo eran; de 40 marcadas como correctas, 38 lo eran. Además, la plantilla determinista sirve de control: solo copia cifras de la capa y el verificador le da 0 errores en 288 cifras.

### P10. Si tuvieras que llevar esto a producción en un banco, ¿qué harías?
Usaría la plantilla como base del informe (fiable y gratis) y el LLM solo para el párrafo de contexto, con el chequeo de cifras y el de dirección dentro del reintento. Haría que el modelo cite el id del hecho en vez de escribir el número, para que el texto final se arme con el valor exacto. Añadiría un umbral: si después de los reintentos queda una cifra roja, el informe no se envía sin revisión humana. Y mediría con más informes y más de un evaluador de legibilidad.

### P11. ¿Por qué la variación del PIB en dólares engaña tanto?
Porque mezcla crecimiento real, inflación y tipo de cambio. Colombia en 2024: el PIB en dólares corrientes subió 14,6 % (por la apreciación del peso), mientras que el PIB real creció 1,5 %. Para un gerente «el PIB creció 14,6 %» suena a boom. La tabla trae los dos indicadores; el modelo elige el número más llamativo para el titular.
