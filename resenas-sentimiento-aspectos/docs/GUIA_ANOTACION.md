# Guía de anotación: aspectos y polaridad en reseñas de accesorios y teléfonos

Esta guía es la que seguí para etiquetar a mano las reseñas de `data/etiquetas/` y la misma que recibe el LLM
(resumida) en su prompt. Si dos personas la siguen, deberían poner las mismas etiquetas.

## Qué se anota en cada reseña

1. **Sentimiento global** (`positivo`, `negativo`, `neutro`): la opinión general del autor sobre el producto,
   leyendo título y texto. `neutro` = mixto sin un lado claro o puramente informativo. **No se copia de las
   estrellas**: una reseña de 3★ puede ser negativa y una de 5★ puede quejarse del envío.
2. **Lista de aspectos**: cada aspecto del catálogo del que el autor opina, con
   - `aspecto`: uno de los 12 de abajo;
   - `polaridad`: `positivo`, `negativo` o `neutro` (se menciona sin valorar, o con valoración mixta que se anula);
   - `cita`: el fragmento **literal** de la reseña que lo justifica (lo más corto que se entienda solo).

Un aspecto aparece **una sola vez** por reseña. Si se valora de forma opuesta en dos lugares, se anota la polaridad
que domina; si ninguna domina, `neutro`.

## Catálogo de aspectos

| Aspecto | Incluye | No incluye (va a…) |
|---|---|---|
| `bateria` | duración/autonomía, capacidad de un power bank, "se descarga rápido" | velocidad de carga (`carga`) |
| `carga` | carga rápida o lenta, cable, conector, puerto, carga inalámbrica, "no carga", "carga 2 teléfonos a la vez" | duración (`bateria`) |
| `pantalla` | pantalla o protector: claridad, brillo, sensibilidad táctil, burbujas, polvo debajo, huellas, se raya la pantalla | encaje del protector en los bordes (`ajuste`) |
| `sonido` | audio, volumen, bajos, micrófono, llamadas, cancelación de ruido | conexión Bluetooth (`facilidad_uso` o `calidad` si falla) |
| `calidad` | construcción, materiales, durabilidad, defectos de fábrica, "se rompió", "dejó de funcionar", "se siente barato/sólido" | elogios genéricos sin referencia a la fabricación ("funciona genial", "me encanta" → solo global) |
| `ajuste` | compatibilidad y encaje: tamaño correcto, botones, recortes de cámara, "no le queda", "se suelta" | — |
| `diseno` | apariencia, color, estilo, delgadez/volumen, tacto, agarre, "se ve bonito" | — |
| `proteccion` | si protegió de caídas, golpes o rayones; "mi teléfono sobrevivió una caída" | resistencia del propio producto (`calidad`) |
| `precio` | precio, valor, "vale la pena", "barato", "caro", "por este precio" | — |
| `envio` | entrega, rapidez del envío, empaque, llegó roto, usado o incompleto | — |
| `atencion` | vendedor o fabricante: servicio al cliente, garantía, reembolso, devolución, reemplazo | — |
| `facilidad_uso` | instalación, configuración, emparejamiento, instrucciones, comodidad de uso | — |

## Reglas para casos dudosos

1. **Solo opiniones del autor sobre el producto comprado.** "Mi teléfono anterior tenía mala batería" no cuenta.
2. **Elogio genérico ≠ aspecto.** "Great product", "works as described", "love it" → solo sentimiento global.
   Para `calidad` debe hablar de la fabricación o de la durabilidad ("sturdy", "cheap plastic", "broke after a week").
3. **"Dejó de funcionar"** → `calidad` negativo. Si dice qué falló (no carga, no suena), va **además** a ese aspecto
   solo si lo describe como característica ("stopped charging" → `carga` negativo y `calidad` negativo).
4. **Producto defectuoso al llegar**: "arrived broken/cracked" → `envio` negativo (y `calidad` si culpa al producto).
5. **Devolución**: "I returned it" sin hablar del proceso no es `atencion`; "easy return", "seller refunded me" sí.
6. **Comparaciones**: "better than the original" sobre la carga → `carga` positivo.
7. **Sarcasmo**: se anota la intención real ("great, it broke on day two" → `calidad` negativo).
8. **Fit del protector de pantalla**: "doesn't cover the edges" → `ajuste`; "bubbles" → `pantalla`;
   "hard to apply" → `facilidad_uso`.
9. **Fundas con batería**: la duración extra es `bateria`.
10. **Cita literal**: se copia del título o del texto sin corregir faltas.
11. **Cargador o cable que no vino en la caja** (teléfonos): `carga` negativo; otros faltantes (SIM, piezas,
    unidades del paquete) → `envio` negativo.
12. **Reseñas en español**: mismas reglas; la cita se copia en español.

## Medición del acuerdo

El conjunto de prueba se etiquetó dos veces, en orden distinto y sin mirar la primera pasada. El acuerdo se mide con
F1 entre pasadas para el conjunto de aspectos, kappa de Cohen por aspecto (presente/ausente) y por polaridad de los
aspectos que ambas pasadas marcaron. Ver resultados en el README.
