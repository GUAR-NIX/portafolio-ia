# Guía paso a paso (Windows)

## 0 · Requisitos (una sola vez)

1. **Python 3.10+** (probado con 3.14).
2. **Ollama** → [ollama.com/download](https://ollama.com/download) y los modelos (solo si quieres volver a correr el
   LLM o los embeddings; las pruebas, la evaluación y la app funcionan con lo grabado):

```powershell
ollama pull qwen2.5:7b      # 4,7 GB; corre en CPU, sin GPU
ollama pull bge-m3          # 1,2 GB; embeddings multilingües
```

3. ~10 GB de ancho de banda: la descarga **lee** 13 GB de Hugging Face en streaming pero solo **guarda** ~7 MB.

## 1 · Todo en un comando

```powershell
cd resenas-sentimiento-aspectos        # la carpeta del proyecto, donde lo hayas clonado
powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1            # con lo grabado (~20 min)
powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1 -ConLLM    # vuelve a llamar a Ollama (~3 h en CPU)
```

## 2 · Paso a paso (lo mismo, a mano)

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
$env:PYTHONIOENCODING = "utf-8"

python descargar_datos.py                 # [1/3] metadatos 4 GB → productos con ≥300 calificaciones
                                          # [2/3] 40 productos más calificados de 8 tipos
                                          # [3/3] reseñas 9,3 GB → reservorio de 200 por producto (~10 min, reanudable)
python -m resenas.muestra                 # reconstruye el texto de las 700 reseñas etiquetadas desde la descarga
                                          # (solo se versionan ids, huellas y etiquetas: licencia de investigación)
pytest -q                                 # pruebas, sin Ollama

# Opcional, con Ollama abierto (todo queda grabado y se puede cortar y reanudar):
python -m resenas.enfoque_embeddings      # embeddings bge-m3 de las reseñas etiquetadas (~25 min)
$env:RESENAS_GRABAR = "1"; python -m resenas.correr_llm    # qwen2.5:7b: 40 dev + 320 test + 80 español (~2,5 h)

python -m resenas.evaluar                 # métricas de los 3 enfoques → reports\evaluacion.json + figuras 01-03
python -m resenas.aplicar                 # enfoque clásico sobre las 55 082 reseñas (~2 min)
python -m resenas.hallazgos               # tablas de negocio → reports\hallazgos\*.csv + figuras 04-08
```

## 3 · La app

```powershell
streamlit run app\streamlit_app.py                                   # http://localhost:8501
$env:RESENAS_PROVEEDOR = "grabado"; streamlit run app\streamlit_app.py   # sin Ollama
```

- **Analizar una reseña:** elige una del test y compara la etiqueta hecha a mano con los tres enfoques, o escribe
  una nueva (el clásico responde al instante; el LLM tarda ~20 s en CPU).
- **Evaluación:** tabla de métricas, acuerdo entre las dos pasadas de etiquetado, figuras.
- **Hallazgos:** aspectos con más quejas, mapa tipo de producto × aspecto, productos y marcas, evolución, estrellas.

## 4 · Cómo etiqueté (y cómo agregar etiquetas)

1. `docs/GUIA_ANOTACION.md` define los 12 aspectos con ejemplos y 12 reglas para casos dudosos.
2. Las reseñas se muestran **sin las estrellas** (para no copiar la calificación en el sentimiento). El texto se
   reconstruye con `python -m resenas.muestra` (no se versiona).
3. Formato compacto, una reseña por línea, en `data/etiquetas/*.txt`:

   ```
   R015003 | 0 | carga+: Fast charge ; calidad-: Weak wires
   ```

4. `python -m resenas.etiquetas data\etiquetas\dev.txt data\etiquetas\dev.jsonl` valida el formato con pydantic y
   avisa si una cita no aparece literal en la reseña.
5. Segunda pasada: 160 reseñas del test en otro orden (semilla 7), sin mirar la primera → `test_pasada2.txt`.

## 5 · Estructura del código

| Módulo | Qué hace |
|---|---|
| `descarga.py` | streaming HTTP con reintentos y `Range`, muestreo de reservorio por producto, punto de control |
| `preparar.py` | une reseñas y productos, sortea dev/test (estratificado por estrellas) y el conjunto en español |
| `texto.py` | título + cuerpo, cláusulas (oraciones y "but/however/pero"), citas literales, detector de español |
| `esquema.py` | modelo pydantic de la anotación = salida del LLM = verdad; JSON Schema para Ollama |
| `enfoque_clasico.py` | diccionario de patrones por aspecto + TF-IDF (palabras y caracteres) + regresión logística |
| `enfoque_embeddings.py` | caché de bge-m3, cláusula ↔ cita, 12 regresiones logísticas + polaridad + global |
| `enfoque_llm.py` / `llm.py` | prompt con el catálogo y las reglas, salida estructurada, modo grabado |
| `metricas.py` | P/R/F1 de aspectos, F1 aspecto+polaridad, exactitud de polaridad, bootstrap, kappa |
| `evaluar.py` | ajuste en dev, test, español, costos, figuras |
| `aplicar.py` / `hallazgos.py` | 55 mil reseñas → tablas de negocio, corrección de prevalencia, regresión con estrellas |

## 6 · Preguntas de entrevista (con respuesta)

**¿Qué es el análisis de sentimiento por aspecto y por qué no basta el sentimiento global?**
El global dice "la reseña es negativa"; el ABSA dice *de qué* se queja ("el cable dejó de cargar") y de qué está
contento ("pero el precio es bueno"). Para negocio lo accionable es el aspecto: una marca de fundas con 4,4★ de
media puede tener un problema concreto de amarilleo que el promedio esconde.

**¿Por qué etiquetaste a mano en vez de usar las estrellas como verdad?**
Las estrellas no dicen qué aspecto se menciona, y su polaridad no coincide con la del aspecto (una reseña de 5★
se queja del envío). Sin verdad independiente no hay forma de comparar enfoques. Etiqueté sin ver las estrellas para
no contaminar la verdad.

**¿Cómo mediste la calidad de tus propias etiquetas?**
Con una segunda pasada sobre 160 reseñas del test, en otro orden y sin mirar la primera: F1 entre pasadas 0,96 y
kappa de polaridad 0,96. Es acuerdo **intra-anotador**: mide consistencia, no si mi criterio coincide con el de otra
persona; lo esperable con dos anotadores distintos es más bajo (ABSA suele estar en 0,7–0,8).

**¿Por qué separaste dev y test?**
Las reglas del diccionario, los umbrales y los hiperparámetros se ajustaron mirando solo dev. El test (320) se usó
una vez al final. Si hubiera ajustado mirando el test, el clásico parecería mejor de lo que es: en dev su F1 pasó de
0,65 a 0,76 al afinar las reglas viendo los errores.

**¿Por qué TF-IDF con caracteres además de palabras?**
Las reseñas tienen faltas ("quit working" por "quite", "chager") y mezcla de idiomas; los n-gramas de caracteres
2–5 las absorben sin diccionario.

**¿De dónde salen las etiquetas para entrenar el enfoque de embeddings?**
De las citas del conjunto dev: cada cláusula que contiene la cita anotada es un ejemplo positivo de ese aspecto
(1 203 cláusulas, 557 con aspecto). Para la polaridad y el global se suman 3 000 reseñas no etiquetadas con su etiqueta de
estrellas (supervisión débil, con menos peso).

**¿Por qué salida estructurada con pydantic en el LLM?**
Ollama restringe la generación al JSON Schema: el modelo solo puede devolver aspectos del catálogo y polaridades
válidas, y nunca rompe el JSON. pydantic valida y descarta aspectos repetidos. Además se pide la **cita textual**
para poder auditar cada etiqueta y detectar alucinaciones (citas que no están en la reseña).

**¿Cómo elegiste el enfoque para las 55 mil reseñas?**
Por costo-efectividad: el LLM tardaría ~12 días de CPU, los embeddings ~15 horas y el clásico 2 minutos. Las cifras de
calidad están en el README; el clásico queda cerca de los otros en detección de aspectos y sus errores medidos
permiten corregir las prevalencias (precisión/exhaustividad por aspecto).

**¿Qué es la corrección de prevalencia?**
Si un aspecto se detecta con precisión P y exhaustividad R, el número real de menciones ≈ detectadas × P / R. Con
el test etiquetado se estima por aspecto (solo donde hay ≥ 10 menciones de soporte).

**La relación queja → estrellas, ¿no es circular?**
En parte sí: el clasificador de polaridad del clásico aprendió de las estrellas. Por eso repetí la regresión con las
620 reseñas etiquetadas a mano (sin ver estrellas), ponderando para deshacer el muestreo estratificado. El orden de
los efectos se mantiene y las cifras con etiquetas a mano son incluso mayores: el ruido del modelo los atenúa.

**¿Cómo evitas que el cambio de productos en el tiempo parezca una tendencia?**
Restando a cada reseña la tasa media de su producto antes de promediar por año (efecto fijo de producto). Si la
tendencia sobrevive, no es solo que en 2022 se vendían otros productos.

**¿Por qué muestreo de reservorio y no las primeras 200 reseñas de cada producto?**
El archivo está ordenado por usuario, no por fecha; tomar las primeras sesgaría hacia ciertos usuarios. El reservorio
da una muestra uniforme de todas las reseñas del producto en una sola pasada y con memoria fija.

**¿Qué harías con más tiempo o presupuesto?**
(1) Un segundo anotador para medir acuerdo inter-anotador; (2) iterar el prompt del LLM en dev (tiende a decir
"neutro" en reseñas positivas cortas); (3) usar el LLM como etiquetador para generar miles de ejemplos y entrenar el
clasificador de embeddings (destilación); (4) GPU: con ~50 tokens/s el LLM cubriría las 55 mil reseñas en un día.
