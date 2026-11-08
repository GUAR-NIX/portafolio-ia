# Ficha técnica · Agente con herramientas sobre Banco Mundial y Open-Meteo (v2.0.0)

## Qué hace

Responde preguntas en español sobre 8 indicadores del Banco Mundial (12 países: ECU, COL, PER, CHL, ARG, BRA, MEX, URY, CRI, BOL, PRY, USA) y 6 variables climáticas de Open-Meteo (12 ciudades) con un LLM local (`qwen2.5:7b`, Ollama, CPU) que llama a 5 herramientas:

1. `consultar_indicador_banco_mundial(codigo_pais, indicador, anio_inicio, anio_fin)`
2. `comparar_paises_indicador(codigos_paises, indicador, anio)` (con año cercano ±2 si falta el dato)
3. `consultar_clima_historico_ciudad(ciudad, fecha_inicio, fecha_fin, variable)`
4. `resumen_climatico_anual_ciudad(ciudad, anio)`
5. `calcular_correlacion_o_cambio(valores_x, valores_y, operacion)` (única fuente de variaciones y correlaciones)

## Reparto de responsabilidades

| Paso | Responsable |
|---|---|
| Elegir herramientas y argumentos, decidir abstenerse, pedir otro turno (máx. 3) | LLM |
| Normalizar alias y rechazar valores fuera del catálogo o de la cobertura | Pydantic |
| Rechazar argumentos sin respaldo en la pregunta (país, ciudad, años, medida, cifras de la calculadora) | Verificación de anclaje (regla por capacidades, sin listas de temas) |
| Redactar la respuesta final | Plantilla determinista (fidelidad por construcción; el texto del LLM se entrega aparte) |

## Datos

- **Modo `instantanea`** (evaluación y pruebas): 3.178 observaciones anuales WDI (1990–2024) y 21.924 registros diarios de Open-Meteo (2020–2024), descargados el 22-10-2026. La ventana climática 2020–2024 es una decisión de diseño para reproducibilidad.
- **Modo `vivo`**: las mismas herramientas consultan `api.worldbank.org/v2` (desde 1960) y `archive-api.open-meteo.com` (desde 1940) con caché en disco, pausa de 1 s y 4 reintentos exponenciales.
- **Verdad terreno**: 98 llamadas de datos distintas, re-ejecutadas contra las APIs en vivo: 98 coinciden (≤ 0,5 %).
- **Licencias y uso:** código MIT. Datos del Banco Mundial (WDI, CC BY 4.0) y de Open-Meteo / ERA5 (CC BY 4.0, con atribución). La API gratuita de Open-Meteo es **solo para uso no comercial**: el modo `vivo` es para portafolio y estudio; un uso comercial requiere su plan de pago o un servidor propio.

## Evaluación

- **Validación**: 44 preguntas (las originales; sobre ellas se ajustó el enrutador por reglas y se eligió el prompt). 36 en dominio y 8 fuera de dominio.
- **Prueba**: 60 preguntas nuevas, corridas una vez con el prompt elegido: 14 simples, 10 comparativas, 10 cruces clima + economía, 10 de cálculo y 16 fuera de dominio de tipos que no aparecen en validación.
- **Respuesta correcta**: en dominio, todas las cifras de referencia en la respuesta (≤ 1 %) sin abstenerse; fuera de dominio, abstención sin ejecutar herramientas. IC 95 % bootstrap (2.000 remuestreos) o Wilson cuando el bootstrap es degenerado.

| Métrica | Validación (44) | Prueba (60) |
|---|---:|---:|
| Respuesta correcta | 93,2 % (0,84–1,00) | 95,0 % (0,88–1,00) |
| LLM: herramientas exactas (en dominio) | 94,4 % | 95,5 % |
| LLM: herramientas + argumentos exactos | 86,1 % | 95,5 % |
| LLM: se abstiene solo fuera de dominio | 37,5 % | 56,2 % |
| Abstención final fuera de dominio | 100,0 % (Wilson 0,68–1,00) | 100,0 % (Wilson 0,81–1,00) |
| Falsa abstención en dominio | 0,0 % | 6,8 % |
| Fidelidad del texto redactado por el LLM | 76,4 % | 80,7 % |
| Turnos del LLM por pregunta | 2,11 | 1,95 |
| Latencia media en CPU | 49,2 s | 31,6 s |

Líneas base en prueba (respuesta correcta): LLM sin herramientas 23,3 %, enrutador por reglas 60,0 %, ReAct en texto 46,7 %.

Mejora de prompt (solo validación): v2 (reglas de uso y un ejemplo por herramienta) sube la respuesta correcta de 84,1 % a 93,2 % y los argumentos exactos del LLM de 77,8 % a 86,1 %, pero baja la abstención propia del LLM fuera de dominio de 75,0 % a 37,5 % (la compensa Pydantic) y añade 0,22 turnos por pregunta.

## Limitaciones y riesgos

1. Conjuntos pequeños (IC amplios) y redactados por el mismo autor.
2. El vocabulario de anclaje es manual: gentilicios o sinónimos no previstos producen falsas abstenciones.
3. El LLM falla en cálculo (62,5 % en validación, 80,0 % en prueba): llama a la calculadora con cifras inventadas antes de ver los datos o razona mal sobre rangos de años.
4. Latencia de 30–50 s por pregunta en CPU: no apto para tráfico alto sin GPU.
5. Las series del Banco Mundial se revisan; las respuestas en modo vivo pueden diferir de la instantánea.

## Reproducibilidad

`data/grabacion_llm.jsonl` guarda las 512 respuestas de `qwen2.5:7b` usadas (indexadas por la huella SHA-1 del prompt completo, incluidas las descripciones de herramientas). `python -m agente.ejecutar_evaluacion` y `pytest` corren sin Ollama ni red.
