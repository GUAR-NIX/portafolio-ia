# Ficha del modelo · Enrutador de quejas financieras (CFPB) v2.0.0

<!-- Archivo generado por `python -m quejas.informe` desde scripts/plantillas/MODEL_CARD.md. No editar a mano. -->

## Qué hace

A partir del texto libre de una queja financiera **en inglés**:
1. Propone una de **6 mesas de atención** (agrupación propia de los productos del CFPB; ver la tabla de mapeo en el README).
2. Decide la **ruta**: si la confianza calibrada de TF-IDF + LR es ≥ 0,704, enruta directo; si no, consulta al
   LLM (`qwen2.5:7b` few-shot); si el LLM no está disponible, envía la queja a triaje humano.
3. Marca la **urgencia** (fraude o robo de identidad, riesgo de ejecución hipotecaria, reposesión, amenazas de arresto,
   embargo o demanda sin aviso) con TF-IDF + LR binario (umbral 0,65, elegido en validación).

## Datos

- **Fuente:** CFPB Consumer Complaint Database (dominio público), espejo Parquet `milesbutler/consumer_complaints`
  (69.416 filas). Se usan 50.568 narrativas de 15 a 260 palabras, sin duplicados, recibidas entre
  2015-03-19 y 2018-04-25. Los textos están en inglés.
- **Partición temporal:** entrenamiento con 25.722 quejas recibidas hasta el 31-03-2017; validación
  11.076 (abril–septiembre de 2017); prueba 10.770 (desde octubre de 2017). Una reserva aleatoria de
  3.000 quejas antiguas sirve de referencia sin deriva.
- **Conjunto común:** 300 quejas recientes balanceadas por mesa, evaluadas con todos los enfoques.
- **Español:** 60 quejas reales del conjunto común traducidas con Claude (no revisadas por un traductor profesional).

## Métricas (conjunto común, n = 300)

| Enfoque | Macro-F1 (IC 95 %) | Exactitud (IC 95 %) | Top-2 | Brier | Latencia CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + LR (25.722 ej.) | **0,825** (0,779–0,865) | 82,7 % (78,3 %–86,7 %) | 93,3 % | 0,255 | 0,10 ms |
| TF-IDF + LR (1.800 ej.) | **0,820** (0,773–0,860) | 82,0 % (77,3 %–86,0 %) | 93,7 % | 0,284 | 0,09 ms |
| Zero-Shot prototipos bge-m3 | **0,690** (0,639–0,737) | 69,3 % (64,3 %–74,3 %) | 83,7 % | 0,532 | 789 ms |
| bge-m3 + LR (1.800 ej.) | **0,798** (0,752–0,842) | 80,0 % (75,7 %–84,7 %) | 90,0 % | 0,298 | 789 ms |
| LLM zero-shot (qwen2.5:7b) | **0,695** (0,643–0,742) | 69,0 % (63,7 %–74,0 %) | 84,3 % | 0,599 | 13,9 s |
| LLM few-shot (qwen2.5:7b, 6 ej.) | **0,799** (0,754–0,842) | 80,0 % (75,7 %–84,3 %) | 89,3 % | 0,399 | 12,3 s |
| Cascada TF-IDF → LLM zero-shot | **0,830** (0,787–0,867) | 83,0 % (78,7 %–87,0 %) | 90,7 % | 0,315 | 3,3 s |
| Cascada TF-IDF → LLM few-shot | **0,842** (0,799–0,880) | 84,3 % (80,3 %–88,3 %) | 91,7 % | 0,284 | 2,9 s |

## Limitaciones

1. **Deriva:** el desempeño baja con el tiempo (Macro-F1 0,846 en la misma época vs. 0,774
   en quejas recientes). Hay que reentrenar periódicamente y vigilar la distribución de mesas.
2. **Etiquetas del CFPB:** la mesa sale del producto que eligió el consumidor; muchas quejas tocan dos productos
   (por ejemplo, cobranza que termina en el reporte de crédito). Parte del «error» es ambigüedad de la etiqueta.
3. **Urgencia:** la etiqueta es un proxy (motivo elegido en el formulario) y es ruidosa.
4. **Español:** el clasificador léxico se entrenó solo en inglés y en español acierta 29 de 60
   (frente a 52 en inglés); en la cascada la mayoría de esas quejas va al LLM. Ver el README.
5. **Latencias** medidas en una CPU sin GPU con otros procesos compitiendo por Ollama: son órdenes de magnitud, no
   cifras de producción.
