# Guía paso a paso (Windows)

<!-- Archivo generado por `python -m quejas.informe` desde scripts/plantillas/GUIA_PASO_A_PASO.md. No editar a mano. -->

El repositorio **ya incluye lo que tarda horas en CPU**: los vectores `bge-m3` (`data/embeddings_bge_m3.npz`), las
respuestas reales de `qwen2.5:7b` en zero-shot y few-shot con sus logprobs (`data/grabacion_llm.jsonl`), los 6 ejemplos
del prompt few-shot (`data/ejemplos_few_shot.json`) y las 60 quejas traducidas al español
(`data/quejas_es_traducidas.jsonl`). Con eso, la evaluación completa, las pruebas, la API y la app corren **sin Ollama**.

## 1 · Crear el entorno e instalar dependencias

Abre PowerShell en la carpeta del proyecto (la que contiene este repositorio) y ejecuta:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 2 · Descargar los datos del CFPB (una vez, ~42 MB)

```powershell
python descargar_datos.py
```

Construye `data/raw/quejas_cfpb.parquet` con 50.568 quejas con narrativa (de 2015-03-19 a 2018-04-25),
la etiqueta de urgencia y las particiones temporales.

## 3 · Reproducir la evaluación con lo grabado (~2 min, sin Ollama)

```powershell
$env:QUEJAS_PROVEEDOR = "grabado"
python -m quejas.entrenar      # modelos, métricas, figuras y reports/resultados.json
python -m quejas.informe       # vuelve a escribir README, MODEL_CARD y esta guía con las cifras nuevas
pytest -q
```

O todo junto: `powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1`

## 4 · (Opcional) Volver a consultar Ollama

Necesitas Ollama con `qwen2.5:7b` y `bge-m3`. Las corridas son **reanudables**: si se cortan, al relanzarlas
continúan donde quedaron y nunca repiten lo ya grabado.

```powershell
ollama serve                                  # si no está corriendo
python -m quejas.grabar embeddings            # bge-m3 (~0,7 s por texto en CPU)
python -m quejas.grabar llm --hasta 300       # zero-shot y few-shot, en bloques alternos de 120 quejas
python -m quejas.grabar llm_es                # las 60 quejas en español
```

La evaluación usa el mayor prefijo del conjunto común que tenga respuesta en **ambos** modos (el orden está
intercalado por mesa, así que cualquier prefijo múltiplo de 6 queda balanceado).

## 5 · API y app

```powershell
uvicorn api.main:app --reload                 # http://localhost:8000/docs
streamlit run app/app_streamlit.py            # pega una queja y mira la ruta y la decisión
```

```powershell
$q = @{ texto_queja = "The bank started foreclosure on my home while my loan modification is under review." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/enrutar -Body $q -ContentType "application/json"
```

## 6 · Notebooks

```powershell
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_datos_particion_temporal_y_deriva.ipynb 02_comparacion_llm_cascada_y_confianza.ipynb 03_urgencia_espanol_y_api.ipynb
```

---

## Preguntas de entrevista (con respuesta)

**1. ¿El LLM clasifica mejor que un TF-IDF?**
Sin ejemplos, no: sobre las mismas 300 quejas recientes, el LLM zero-shot logra 0,695 de Macro-F1
(IC 95 % 0,643–0,742) y TF-IDF + LR (entrenado con 25.722 quejas) 0,825 (0,779–0,865); la
diferencia pareada es +0,131 (IC 0,072 a 0,190). Con 6 ejemplos en el prompt el LLM sube a
0,799 (0,754–0,842) y la diferencia con TF-IDF (+0,027, IC -0,022 a 0,075) ya es
no distinguible. Por eso no afirmo que uno «supere» al otro: digo que empatan y que TF-IDF es miles de veces más
barato. Uso bootstrap **pareado** (las mismas quejas en cada remuestreo) porque los dos modelos se equivocan en
quejas parecidas y comparar dos IC sueltos sería demasiado conservador.

**2. ¿Por qué no usaste la confianza que el LLM escribe en su respuesta?**
Porque no informa nada: en el experimento con ese campo, 53 de 54 respuestas decían 0,99. La confianza
sale entonces de los **logprobs** de la primera ficha de la categoría y la comparé con otras señales por AUROC
(acierto vs. error). Para la cascada uso la probabilidad **calibrada** de TF-IDF, que se puede ajustar en validación
sin gastar llamadas al LLM.

**3. ¿Cómo mediste la deriva?**
Entrené con quejas recibidas hasta el 31-03-2017 y comparé una reserva aleatoria de la misma época
(Macro-F1 0,846) con las quejas desde octubre de 2017 (0,774). En abril de 2017 el CFPB
cambió su lista de productos y motivos, así que la partición aleatoria habría sobreestimado el desempeño.

**4. ¿Cómo elegiste los umbrales?**
Todos en validación (abril–septiembre de 2017), nunca en prueba: la temperatura de calibración, el umbral de
abstención a triaje humano (mínimo costo con 14 USD por mal enrutamiento y 3,50 USD por triaje), el umbral de
urgencia (máximo F1) y el de la cascada (el 25 % de menor confianza va al LLM).

**5. ¿De dónde sale la etiqueta de urgencia?**
Del motivo (*Issue/Sub-issue*) que el propio consumidor eligió en el formulario del CFPB: fraude o robo de identidad,
riesgo de ejecución hipotecaria, reposesión del vehículo y amenazas de arresto, embargo o demandas sin aviso. No sale
del texto, así que sirve como referencia independiente para evaluar detectores basados en el texto; a cambio, es
ruidosa (por ejemplo, «Loan modification, collection, foreclosure» mezcla modificaciones rutinarias con ejecuciones).

**6. ¿Las quejas en español son reales?**
Son traducciones de 60 quejas **reales** del conjunto de prueba, con su etiqueta del CFPB. Las traduje con un modelo
distinto del evaluado (Claude), para no darle ventaja a qwen2.5:7b clasificando sus propias traducciones. Con 60
quejas, cada una vale 1,7 puntos de exactitud: los intervalos son anchos.

**7. ¿Cómo decide la cascada cuándo llamar al LLM, y para qué sirve si TF-IDF ya empata?**
TF-IDF da una probabilidad calibrada por queja. Si es ≥ τ (0,704, fijado en validación para que el
25 % de menor confianza vaya al LLM) decide TF-IDF; si no, decide el LLM few-shot. En prueba llegó al
LLM el 23,3 % de las quejas. La mejora de Macro-F1 sobre TF-IDF solo no es distinguible con 300
quejas, así que no la vendo como ganancia demostrada: lo que sí se ve es que la cascada se defiende donde TF-IDF
falla (en español casi todas las quejas tienen baja confianza y pasan al LLM) a cambio de más latencia en esas quejas.

**8. ¿Qué cuidarías para llevarlo a producción?**
(1) Monitorear la deriva: el Macro-F1 por trimestre ya cae desde 0,846 (misma época) a 0,774
(quejas recientes), así que habría que reentrenar con una ventana reciente y vigilar la distribución de mesas. (2) Dejar
que la abstención mande a revisión humana las quejas de baja confianza calibrada. (3) Tratar la urgencia como alerta,
no como decisión: su etiqueta es un proxy ruidoso (el motivo que eligió el consumidor) y ningún detector es bueno todavía.
(4) Texto personal: el CFPB ya enmascara datos con `XXXX`, pero una entidad real tendría que anonimizar antes de
guardar o enviar nada a un modelo.

**9. ¿Qué limitaciones tiene la evaluación?**
El conjunto común son 300 quejas balanceadas por mesa, no la distribución real: sirve para comparar
enfoques, no para estimar la exactitud en producción. Las mesas son una agrupación propia de los productos del CFPB y
la etiqueta sigue al producto que eligió el consumidor, no a lo que dice el texto (de ahí muchos errores de «reporte de
crédito»). Las quejas en español son 60 traducciones hechas con un modelo, sin traductor profesional, y las latencias
se midieron en una CPU compartida. Por eso reporto intervalos de confianza y marco como «no distinguible» lo que el IC
no separa.
