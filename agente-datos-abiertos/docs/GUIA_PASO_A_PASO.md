# Guía paso a paso (Windows) · Agente de Datos Abiertos (`agente-datos-abiertos`)

El repositorio incluye la **instantánea** de las APIs (`data/instantanea_indicadores.json` del Banco Mundial y `data/instantanea_clima.json` de Open-Meteo), los **dos conjuntos de evaluación** (`data/conjunto_validacion.json`, 44 preguntas; `data/conjunto_prueba.json`, 60 preguntas nuevas), las **respuestas grabadas de `qwen2.5:7b`** (`data/grabacion_llm.jsonl`) y todos los reportes. Por eso las pruebas, la evaluación, la API y la app corren **sin Ollama y sin internet** para cualquier pregunta de los conjuntos. Una pregunta nueva necesita Ollama; el modo `vivo` necesita internet.

---

## 1 · Crear el entorno e instalar dependencias

Abre PowerShell en la carpeta del proyecto (la que contiene este `README.md`):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 2 · Pruebas automáticas (sin Ollama ni red)

```powershell
pytest -q
```

Resultado esperado: **`20 passed`**. Entre otras cosas verifican que:
- las cifras clave del README coinciden con `reports/resultados.json`, y que esta guía no contiene cifras que no estén en la instantánea;
- el cliente HTTP reintenta ante `429`, respeta `Retry-After` y usa la caché (con `httpx.MockTransport`, sin red);
- `motor_agente.py` no contiene palabras de las preguntas trampa (`bitcoin`, `nasdaq`, `marte`…);
- las 104 preguntas × 3 enfoques con LLM están grabadas (si faltara una, la prueba falla en vez de llamar a Ollama);
- no hay rutas personales en docs, notebooks ni salidas.

## 3 · Reproducir la evaluación

```powershell
python -m agente.ejecutar_evaluacion          # ~1 min: usa las respuestas grabadas
```

Para volver a consultar `qwen2.5:7b` (necesita `ollama serve` y `ollama pull qwen2.5:7b`). En CPU tarda **horas**: el agente tarda entre 30 y 50 s por pregunta. Es reanudable: lo ya grabado no se repite.

```powershell
cd src
python -m agente.correr_llm validacion       # agente v1 y v2 + líneas base en las 44 de validación
python -m agente.correr_llm prueba_agente    # agente con la versión de prompt elegida en validación
python -m agente.correr_llm prueba_base      # líneas base en las 60 de prueba
cd ..
```

O todo junto: `powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1` (añade `-ConLLM` para completar grabaciones).

## 4 · Los dos modos de datos

| Modo | Variable | De dónde lee | Cobertura | Para qué |
|---|---|---|---|---|
| `instantanea` (por defecto) | `AGENTE_MODO_DATOS=instantanea` | `data/instantanea_*.json` | Banco Mundial 1990–2024 · clima 2020–2024 | evaluación reproducible, pruebas, demo sin red |
| `vivo` | `AGENTE_MODO_DATOS=vivo` | `api.worldbank.org/v2` y `archive-api.open-meteo.com` | Banco Mundial 1960–2024 · clima desde 1940 hasta hoy − 7 días | datos actuales y años fuera de la instantánea |

La ventana 2020–2024 del clima es una **decisión de diseño** de la instantánea (para evaluar sin red siempre igual), no un límite de Open-Meteo. El cliente HTTP (`src/agente/clientes_api.py`) guarda cada respuesta en `data/cache_api/` (clave SHA-1 de URL + parámetros), deja al menos 1 s entre peticiones y reintenta hasta 4 veces con espera exponencial ante `429`/`5xx`.

```powershell
$env:AGENTE_MODO_DATOS = "vivo"
python -m agente.verificacion_vivo verdad    # compara la verdad terreno con las APIs en vivo
python -m agente.verificacion_vivo agente    # corre el agente en ambos modos sobre 5 preguntas
```

Aviso práctico: el día de la verificación la API del Banco Mundial tardó unos 30 s por petición; la caché en disco hace que la segunda vez sea instantánea.

## 5 · API REST

```powershell
uvicorn api.main:app --reload       # documentación en http://localhost:8000/docs
```

```powershell
$q = @{
  pregunta = "Cruza datos económicos y climáticos de Ecuador (ECU) en 2023: ¿cuál fue su crecimiento del PIB real y cuál fue la precipitación acumulada en Quito entre 2023-01-01 y 2023-12-31?"
  enfoque = "tool_calling_nativo"
  modo_datos = "instantanea"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/consultar -Body $q -ContentType "application/json; charset=utf-8"
```

La respuesta incluye `rastro_pasos_llm` (cada turno del LLM, los argumentos que propuso, los que quedaron tras validar y los rechazos con su motivo), `texto_llm` (lo que redactó el modelo) y `respuesta_final` (plantilla con las cifras de las herramientas).

Para auditar una traza externa con `/evaluar_traza`, con las cifras reales de la instantánea:

```powershell
$traza = @{
  pregunta = "¿Cuál fue el PIB per cápita en USD de Ecuador (ECU) entre 2020 y 2023 según el Banco Mundial?"
  llamadas_herramientas = @(@{
    herramienta = "consultar_indicador_banco_mundial"
    argumentos = @{ codigo_pais = "ECU"; indicador = "pib_per_capita_usd"; anio_inicio = 2020; anio_fin = 2023 }
  })
  respuesta_final = "Según el Banco Mundial, el PIB per cápita de Ecuador fue 5463.65 USD en 2020 y 6718.14 USD en 2023."
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://localhost:8000/evaluar_traza -Body $traza -ContentType "application/json; charset=utf-8"
```

Devuelve `cumple_umbral_1pct = true`. Si cambias las cifras por otras que no salen de la herramienta, devuelve `false`.

## 6 · App Streamlit con el rastro de herramientas

```powershell
streamlit run app\app_streamlit.py
```

Elige una pregunta grabada (funciona sin Ollama) o escribe una nueva (necesita Ollama). La app muestra la respuesta final, el texto del LLM, cada turno con las herramientas pedidas (✅ ejecutada / ⛔ rechazada y por qué) y las observaciones JSON que devolvió cada herramienta.

## 7 · Notebooks

```powershell
cd notebooks
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 01_ingesta_apis_open_meteo_y_banco_mundial.ipynb 02_arquitectura_agente_tool_calling_y_guardrails.ipynb 03_evaluacion_llamadas_fidelidad_y_api.ipynb
```

---

## Cómo funciona por dentro

1. **Datos (`datos.py`, `clientes_api.py`).** 12 países × 8 indicadores WDI y 12 ciudades × 6 variables diarias de Open-Meteo. Las herramientas piden datos a `datos.serie_indicador`, `datos.clima_mensual` y `datos.clima_anual`, que leen la instantánea o llaman a la API según el modo.
2. **Herramientas (`herramientas.py`).** Cinco funciones deterministas con un modelo Pydantic cada una: normaliza alias (`"Perú"→"PER"`, `"CDMX"→"Ciudad de México"`, `"2022"→"2022-01-01"`) y rechaza rangos fuera de la cobertura del modo activo. La herramienta de datos devuelve datos; las variaciones las calcula solo `calcular_correlacion_o_cambio`.
3. **Agente (`motor_agente.ejecutar_agente_herramientas`).** Bucle de hasta 3 turnos con la API `tools` de Ollama. **El LLM decide** qué herramientas llamar, con qué argumentos y cuándo abstenerse. **El código** valida con Pydantic, verifica el anclaje y devuelve al LLM la observación o el error para que corrija. La respuesta final se arma con plantilla a partir de las observaciones.
4. **Verificación de anclaje.** Regla general basada en las capacidades de las herramientas, sin listas de temas prohibidos: el país o la ciudad deben aparecer en la pregunta; los años deben estar dentro del periodo que menciona; el indicador o la variable deben corresponder a algún término de su descripción (`config.TERMINOS_*`, congelado antes de escribir el conjunto de prueba); las cifras que se pasan a la calculadora deben salir de observaciones anteriores o de la pregunta.
5. **Evaluación (`evaluacion.py`, `ejecutar_evaluacion.py`).** Validación (44) para ajustar y elegir el prompt; prueba (60) corrida una sola vez. Las métricas `llm_*` puntúan lo que propuso el LLM antes de que el código toque nada; las métricas de sistema puntúan lo que se ejecutó y respondió.

---

## Preguntas de entrevista (con respuesta)

### 1. ¿Por qué un LLM de 7B necesita herramientas para cifras del Banco Mundial?
Porque los pesos no guardan tablas exactas. En la validación, `qwen2.5:7b` sin herramientas respondió que el PIB per cápita de Ecuador fue «aproximadamente $4,100 en 2020 … $4,300», cuando la serie `NY.GDP.PCAP.CD` de la instantánea da **5.463,65 USD (2020)** y **6.718,14 USD (2023)**. Para la inflación de Argentina en 2023 dijo 9,8 %; el dato es **133,49 %**.

### 2. ¿Por qué una calculadora como herramienta aparte?
Para que ninguna cifra derivada dependa de aritmética del modelo. Con la serie real de Ecuador, de **5.976,16 USD (2015)** a **6.718,14 USD (2023)**, la variación es (6.718,14 − 5.976,16) / 5.976,16 × 100 = **12,42 %**; esa cifra la calcula `calcular_correlacion_o_cambio`, no el LLM. Además, la herramienta de datos ya no devuelve la variación precalculada: si el LLM no llama a la calculadora, la respuesta no la trae.

### 3. ¿Qué hace el LLM y qué hace el código en tu agente?
El LLM elige herramienta, argumentos y si abstenerse, y decide si necesita otro turno (por ejemplo, para la calculadora). El código valida tipos y rangos, verifica el anclaje, ejecuta y redacta la respuesta final con una plantilla. Por eso reporto por separado la exactitud de selección y de argumentos **del LLM** (antes de cualquier corrección) y el resultado **del sistema**. La fidelidad numérica de la plantilla es 100 % por construcción y no la presento como mérito del modelo; la del texto que redacta el LLM sí se mide (ver README).

### 4. ¿Qué errores de diseño evitaste al medir el agente?
Tres. (1) Un «guardrail» de dominio hecho con una lista de regex de palabras de las preguntas trampa (`bitcoin`, `nasdaq`, `marte`…) sobreajusta al benchmark: aquí no hay listas, las reglas se derivan de las capacidades de las herramientas. (2) Si el código sobrescribe los argumentos del LLM con los de una regex y añade por su cuenta la segunda herramienta y la calculadora, se mide el código y no el modelo; la ablación lo muestra (añadir Pydantic no mueve nada y todo el salto viene del encadenamiento fijo). Aquí se ejecuta lo que propone el LLM y los errores vuelven a él. (3) Evaluar solo en las preguntas con las que se ajustó todo: por eso hay validación (44) y una prueba nueva (60) corrida una vez, con la selección de herramientas y argumentos medida sobre lo que propone el LLM (95,5 % en prueba) y 93,2 % en validación frente a 95,0 % en prueba del sistema.

### 5. ¿Cómo detectas preguntas fuera de dominio sin listas de palabras?
En dos capas. Primero el propio LLM puede no llamar herramientas: en la prueba lo hizo en 9 de las 16 preguntas fuera de dominio (por ejemplo, población de Ecuador o elecciones). Segundo, si llama, cada argumento pasa por Pydantic (catálogo y cobertura) y por la verificación de anclaje: las otras 7 llamadas (Venezuela, Toronto, Cuenca, humedad, deuda pública, pobreza, clima de 2010) las rechazó Pydantic por estar fuera del catálogo o de la cobertura, el LLM leyó el error y se abstuvo. Resultado: 16/16 sin ejecutar ninguna herramienta, sin una sola lista de palabras prohibidas.

### 6. ¿Qué es la verificación de anclaje y qué riesgo tiene?
Exigir que cada argumento tenga respaldo en la pregunta o en observaciones previas. Su riesgo son las **falsas abstenciones**: «¿Qué porcentaje de los mexicanos usaba internet en 2020?» no nombra «México», y el anclaje rechaza `MEX`. Por eso reporto la tasa de falsa abstención en dominio (6,8 % en prueba, 3 de 44) junto a la abstención correcta. A cambio, en validación frenó tres llamadas a la calculadora con cifras que el LLM inventó antes de ver los datos.

### 7. ¿Cómo mediste la mejora de prompt sin contaminar la prueba?
Comparé dos versiones solo en validación: v1 (descripciones originales) y v2 (cuándo usar cada herramienta, formato de argumentos y un ejemplo inventado por herramienta, más 4 reglas de proceso en el prompt del sistema). La regla de elección (más respuestas correctas; si empatan, mejor selección del LLM) se fijó antes y la versión ganadora es la única que se corrió en prueba. v2 subió la respuesta correcta de 84,1 % a 93,2 %, pero el LLM pasó a abstenerse solo menos veces fuera de dominio (75,0 % → 37,5 %); esa diferencia la cubre la validación del código.

### 8. ¿Por qué IC de Wilson en algunos casos?
Con 44 o 60 preguntas, si un enfoque acierta todas, el bootstrap percentil da «1,00–1,00», que no es un intervalo honesto. Para métricas binarias con todos los valores iguales uso Wilson (por ejemplo, 16/16 da 0,81–1,00); en el resto, bootstrap con 2.000 remuestreos.

### 9. ¿Cuánto cuesta en latencia?
En CPU, el agente tardó 31,6 s de media por pregunta en prueba (1,95 turnos del LLM, cada uno con unos 1.500 tokens de prompt) y 49,2 s en validación, que se corrió mientras otro proyecto usaba el mismo Ollama. El LLM sin herramientas tarda 8,0 s y el enrutador por reglas menos de 1 ms. Para tráfico real haría falta GPU, o reservar el agente para las preguntas compuestas.

### 10. ¿Qué cambia en el modo vivo?
Las herramientas consultan las APIs por HTTP con caché en disco, pausa de 1 s y reintentos exponenciales; la cobertura pasa a ser la de las fuentes (clima desde 1940). «¿Cuál fue la precipitación acumulada en Quito entre 1995-01-01 y 1995-12-31?» se abstiene en modo instantánea (Pydantic rechaza la fecha) y en modo vivo responde 2.860,4 mm; el resultado está en `reports/prueba_modo_vivo.json`. Además, las 98 llamadas de datos de la verdad terreno se re-ejecutaron contra las APIs y las 98 coinciden con la instantánea. Ojo con las condiciones de uso: los datos de Open-Meteo son CC BY 4.0, pero su API gratuita es **solo para uso no comercial**; para un uso comercial hace falta su plan de pago o un servidor propio.
