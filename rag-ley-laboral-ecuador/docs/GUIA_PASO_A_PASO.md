# Guía paso a paso — Asistente RAG de la ley laboral del Ecuador

## 1. Qué necesitas

- Windows con Python 3.10+ (probado con 3.14), ~4 GB libres (el re-ranker pesa 2,3 GB).
- [Ollama](https://ollama.com) con los modelos `qwen2.5:7b` (respuestas) y `bge-m3` (embeddings):
  `ollama pull qwen2.5:7b` y `ollama pull bge-m3`. Sin GPU funciona, solo es lento.
- **Para las pruebas, la demo y los notebooks no hace falta Ollama**: todo lo que el modelo respondió en la
  evaluación está grabado.

## 2. Correrlo

Todo de una vez (reanudable; si la PC se apaga, vuelve a correrlo y sigue donde quedó):

```powershell
powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1
```

Paso a paso:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
$env:PYTHONUTF8 = "1"

python descargar_datos.py                       # 1. PDF oficiales → data/raw/ (SERCOP es lento: ~10 min)
python -m rag_laboral.extraer                   # 2. un fragmento por artículo → data/articulos.jsonl
$env:RAG_GRABAR = "1"                           #    (graba embeddings, puntajes y respuestas del LLM)
python -m rag_laboral.indexar                   # 3. embeddings bge-m3 → data/indice/emb_articulos.npy
python -m rag_laboral.evaluar_recuperacion      # 4. Recall@k y MRR de los 4 métodos × 3 variantes de sinónimos
python -m rag_laboral.evaluar_generacion --correr   # 5. qwen2.5:7b responde las 145 preguntas (~1 min c/u)
python -m rag_laboral.graficos                  # 6. figuras de reports/figures/
python -m pytest -q                             # pruebas (sin Ollama)
```

Usarlo:

```powershell
streamlit run app/streamlit_app.py                       # chat con citas clicables
uvicorn api.main:app --port 8000                         # API: POST /preguntar, GET /articulo/CT-69
$env:RAG_PROVEEDOR = "grabado"; streamlit run app/streamlit_app.py   # demo sin Ollama (preguntas de la evaluación)
```

## 3. Cómo está hecho (en orden)

1. **Extracción** (`extraer.py`): pypdf lee cada página; se quitan «Página N de M» y el pie de Lexis; un renglón
   `Art. 69` abre un artículo; `TÍTULO`, `Capítulo` y `Parágrafo` (con el nombre en el renglón siguiente) quedan como
   metadatos. Las notas editoriales («Artículo sustituido por…»), CONCORDANCIAS y JURISPRUDENCIA se guardan aparte:
   no son ley y no se pueden citar. Se excluyen 61 artículos derogados.
2. **Índices** (`recuperar.py`): BM25 (`rank-bm25`) sobre tokens sin tildes ni palabras vacías y con plurales
   recortados; embeddings `bge-m3` en una matriz numpy normalizada (coseno = producto punto); fusión **Reciprocal Rank
   Fusion** (k = 60).
3. **Re-ranking**: los 20 mejores del híbrido pasan por `BAAI/bge-reranker-v2-m3` (cross-encoder: lee pregunta y
   artículo juntos) y se quedan 5.
4. **Generación** (`generar.py`): qwen2.5:7b recibe solo esos 5 artículos con su identificador (`[CT-69]`) y responde
   con un JSON validado por pydantic: `articulos_citados → cubierto_por_la_ley → respuesta → confianza`.
   En código: las citas que no estaban en el contexto se eliminan y se cuentan; si no queda ninguna cita válida, la
   respuesta es «la ley cargada no cubre esa pregunta».
5. **Rechazo por umbral** (opcional): si el mejor puntaje del re-ranker es menor que τ, se rechaza sin llamar al LLM.
   τ se eligió en el conjunto principal y se reporta también en el de preguntas nuevas.

## 4. Cómo se armó el conjunto de evaluación

Ver [`data/evaluacion/preguntas.csv`](../data/evaluacion/preguntas.csv) y el detalle en el README. Resumen:

- Leí el texto extraído de ~110 artículos de los temas que más pregunta la gente (vacaciones, jornada, horas extra,
  décimos, utilidades, contratos, maternidad y paternidad, despido, desahucio, visto bueno, jubilación patronal,
  sindicatos, prescripción) y escribí una pregunta como la haría un trabajador, a veces coloquial
  («me botaron», «décimo tercero»), con su artículo correcto, una respuesta de referencia y una **frase de evidencia
  copiada literal del artículo**. Una prueba verifica que cada artículo exista en el índice y que la evidencia esté en
  su texto.
- Si dos artículos responden (p. ej. art. 4 del Código y art. 326 de la Constitución), ambos cuentan como correctos.
- **Trampas**: preguntas que la ley cargada no responde. *Cercanas* (parecen laborales pero son de la Ley de Seguridad
  Social, la LOSEP o el Código de la Niñez: aportes al IESS, vacaciones de servidores públicos, pensión alimenticia) y
  *lejanas* (IVA, RUC, multas de tránsito).
- **Preguntas nuevas** (20 + 5 trampas): escritas antes de mirar cualquier resultado y usadas solo al final para medir
  si los sinónimos v2 y el umbral se sobreajustaron al conjunto principal.

## 5. Preguntas de entrevista (con respuesta)

**1. ¿Por qué trocear por artículo y no cada 500 caracteres?**
Porque la unidad de cita de una ley es el artículo: «art. 69 del Código del Trabajo» es verificable; un trozo de 500
caracteres puede mezclar el final de un artículo con el inicio de otro y el modelo terminaría citando lo que no dice.
El costo son artículos muy largos (el 42 tiene 35 numerales): para esos, el re-ranker y el LLM leen una *ventana* con
los numerales más parecidos a la pregunta, pero la cita sigue siendo el artículo completo.

**2. ¿Qué aporta la búsqueda híbrida sobre solo embeddings?**
BM25 acierta con términos exactos y números («art. 188», «décimo cuarto», «visto bueno»); los embeddings, con
paráfrasis («¿me pueden echar si estoy enferma?» → despido durante enfermedad no profesional). RRF combina posiciones,
no puntajes, así que no hay que calibrar escalas distintas. Las cifras por método están en el README.

**3. ¿Qué es un re-ranker y por qué no usarlo sobre todo el corpus?**
Un cross-encoder lee pregunta y documento *juntos* y da un puntaje de relevancia mucho más fino que comparar dos
vectores calculados por separado, pero es caro: un pase por par. Por eso solo re-ordena los 20 candidatos del híbrido
(20 pares por pregunta en vez de 627).

**4. ¿Cómo evitas que el modelo invente artículos?**
Tres capas: (1) el modelo solo ve 5 artículos con identificador y el prompt le prohíbe citar otros; (2) en código,
cualquier cita que no esté en esos 5 se elimina y se cuenta (métrica «citas inventadas»); (3) si no queda ninguna cita
válida, no se muestra la respuesta del modelo sino el mensaje de «no cubierto». Además, cada artículo correcto del
conjunto de evaluación se verifica contra el índice en las pruebas.

**5. ¿Cómo mides que el sistema dice «no sé» cuando debe?**
Con preguntas trampa sin respuesta en la ley: rechazo correcto = porcentaje en que responde que la ley cargada no lo
cubre. Y su contraparte: abstención indebida en las respondibles. Subir una baja la otra; por eso se reportan juntas y
el umbral se elige con un criterio explícito en un conjunto y se reporta en otro.

**6. ¿Por qué el orden de los campos del JSON importa?**
La salida estructurada de Ollama genera los campos en el orden del esquema. Si `respuesta` va antes que
`articulos_citados`, el modelo redacta primero y luego busca qué citar (racionaliza). Con la evidencia primero, elige
los artículos y después escribe a partir de ellos. Es la misma lección del RAG de normativa aduanera (proyecto IA): un bug de 40 puntos.

**7. ¿Cómo sabes que los sinónimos no son sobreajuste?**
La v1 se escribió antes de ver resultados; la v2 corrigió los fallos del conjunto principal (por ejemplo «renunciar a
mis derechos» activaba el sinónimo de renuncia al trabajo). Para no engañarme medí ambas en 20 preguntas nuevas que no
usé para ajustar: si la v2 solo mejora el conjunto principal, es sobreajuste.

**8. ¿Qué es MRR y por qué además de Recall@5?**
Recall@5 dice si el artículo correcto está entre los 5 que lee el LLM (lo que importa para responder). MRR premia que
esté *primero*: 1 si está en la posición 1, 0,5 en la 2, etc. Un re-ranker suele mejorar más el MRR que el Recall@20,
porque reordena candidatos que ya estaban.

**9. ¿En qué se diferencia de tu RAG de normativa aduanera (proyecto IA)?**
Ese ya tenía BM25 + embeddings + RRF sobre normativa aduanera con 45 preguntas. Aquí se agrega: re-ranking con
cross-encoder, sinónimos coloquiales medidos (v1 vs v2), rechazo por umbral calibrado además del rechazo del LLM,
eliminación de citas fuera del contexto, 145 preguntas (120 respondibles) con evidencia literal verificada, trampas cercanas y lejanas,
y evaluación por pieza de las 4 variantes de recuperación.

**10. ¿Qué harías para llevarlo a producción?**
Monitorear la fecha de última reforma (re-descargar y re-indexar ante cada reforma, con pruebas de regresión sobre el
conjunto de evaluación), GPU o un modelo más pequeño para bajar la latencia, registrar preguntas sin respuesta para
ampliar el corpus (Ley de Seguridad Social, LOSEP), y revisión de un abogado laboralista sobre una muestra de
respuestas. Nunca presentarlo como asesoría legal.

## 6. Problemas comunes

| Síntoma | Solución |
|---|---|
| `No hay conexión con Ollama` | `ollama serve` en otra terminal; `ollama list` debe mostrar `qwen2.5:7b` y `bge-m3`. |
| `Modo grabado: esta consulta no tiene embedding guardado` | Estás en `RAG_PROVEEDOR=grabado` con una pregunta nueva: abre Ollama y quita esa variable. |
| La descarga del Código se queda en 260 KB | El portal del SERCOP corta: vuelve a correr `descargar_datos.py`, continúa desde donde quedó. |
| Caracteres raros (`Ã­`) en PowerShell | `$env:PYTHONUTF8 = "1"` y usa Windows Terminal. |
