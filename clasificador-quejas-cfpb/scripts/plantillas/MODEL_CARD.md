# Ficha del modelo · Enrutador de quejas financieras (CFPB) v2.0.0

<!-- Archivo generado por `python -m quejas.informe` desde scripts/plantillas/MODEL_CARD.md. No editar a mano. -->

## Qué hace

A partir del texto libre de una queja financiera **en inglés**:
1. Propone una de **6 mesas de atención** (agrupación propia de los productos del CFPB; ver la tabla de mapeo en el README).
2. Decide la **ruta**: si la confianza calibrada de TF-IDF + LR es ≥ {{tau_cascada}}, enruta directo; si no, consulta al
   LLM (`qwen2.5:7b` few-shot); si el LLM no está disponible, envía la queja a triaje humano.
3. Marca la **urgencia** (fraude o robo de identidad, riesgo de ejecución hipotecaria, reposesión, amenazas de arresto,
   embargo o demanda sin aviso) con TF-IDF + LR binario (umbral {{umbral_urgencia}}, elegido en validación).

## Datos

- **Fuente:** CFPB Consumer Complaint Database (dominio público), espejo Parquet `milesbutler/consumer_complaints`
  ({{n_parquet}} filas). Se usan {{n_corpus_total}} narrativas de 15 a 260 palabras, sin duplicados, recibidas entre
  {{fecha_min}} y {{fecha_max}}. Los textos están en inglés.
- **Partición temporal:** entrenamiento con {{n_train_tfidf}} quejas recibidas hasta el 31-03-2017; validación
  {{n_val}} (abril–septiembre de 2017); prueba {{n_test_reciente}} (desde octubre de 2017). Una reserva aleatoria de
  {{n_test_misma_epoca}} quejas antiguas sirve de referencia sin deriva.
- **Conjunto común:** {{n_test_comun}} quejas recientes balanceadas por mesa, evaluadas con todos los enfoques.
- **Español:** {{n_test_espanol}} quejas reales del conjunto común traducidas con Claude (no revisadas por un traductor profesional).

## Métricas (conjunto común, n = {{n_test_comun}})

{{tabla_principal}}

## Limitaciones

1. **Deriva:** el desempeño baja con el tiempo (Macro-F1 {{f1_deriva_vieja}} en la misma época vs. {{f1_deriva_nueva}}
   en quejas recientes). Hay que reentrenar periódicamente y vigilar la distribución de mesas.
2. **Etiquetas del CFPB:** la mesa sale del producto que eligió el consumidor; muchas quejas tocan dos productos
   (por ejemplo, cobranza que termina en el reporte de crédito). Parte del «error» es ambigüedad de la etiqueta.
3. **Urgencia:** la etiqueta es un proxy (motivo elegido en el formulario) y es ruidosa.
4. **Español:** el clasificador léxico se entrenó solo en inglés y en español acierta {{es_aciertos_tfidf}} de 60
   (frente a {{en_aciertos_tfidf}} en inglés); en la cascada la mayoría de esas quejas va al LLM. Ver el README.
5. **Latencias** medidas en una CPU sin GPU con otros procesos compitiendo por Ollama: son órdenes de magnitud, no
   cifras de producción.
