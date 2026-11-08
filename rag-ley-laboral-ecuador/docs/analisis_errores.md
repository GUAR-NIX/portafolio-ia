# Análisis de errores (con ejemplos reales)

Todas las cifras salen de `reports/recuperacion.json`, `reports/generacion.json`, `reports/recuperacion_detalle.csv`,
`reports/generacion_detalle.csv` y `reports/fidelidad_muestra.csv` (corrida del 2026-10-25, qwen2.5:7b en CPU).

## 1. Recuperación

### 1.1 Sorpresa: el híbrido salió PEOR que solo embeddings

| Conjunto principal (100), sin sinónimos | Recall@5 | MRR@20 |
|---|---:|---:|
| BM25 | 0,69 | 0,56 |
| Vectorial (bge-m3) | **0,96** | 0,86 |
| Híbrido (RRF, pesos iguales) | 0,86 | 0,74 |
| Híbrido + re-ranking | 0,94 | **0,89** |

RRF suma `1/(60 + posición)` de cada lista con el mismo peso. Cuando BM25 es mucho más débil (0,69 vs 0,96), sus
falsos positivos empujan hacia abajo aciertos del vectorial. Ejemplo real, **P024** «¿Cómo se calcula el décimo tercer
sueldo?»: el vectorial pone el art. 111 en la posición 1, BM25 no lo encuentra en sus 20 primeros (el artículo dice
«décima tercera remuneración», no «décimo tercer sueldo») y el híbrido lo deja en la posición **17**.

**Qué se hizo:** no se cambió el peso de RRF a mano (sería ajustar viendo el resultado); el re-ranker recibe los 20
candidatos del híbrido, donde el artículo correcto está el 96–99 % de las veces, y los reordena leyendo pregunta y
artículo juntos. Con sinónimos v2, híbrido + re-ranking llega a Recall@5 = 0,97 y MRR = 0,92: empata al vectorial en
Recall@5 y lo supera en MRR (el artículo correcto queda primero más seguido: 88 % vs 79 %). Con IC 95 % por bootstrap
de preguntas (pareado): Recall@5 0,00 (−0,03 a +0,03) y MRR +0,05 (0,01 a 0,10) frente al vectorial; frente al
híbrido sin re-ranker, MRR +0,16 (0,10 a 0,23). En las 20 preguntas nuevas el re-ranker no superó al vectorial
(Recall@5 −0,05, IC −0,15 a 0,00; MRR +0,01, IC −0,13 a +0,15).

**Lección honesta:** con preguntas redactadas en lenguaje normal, bge-m3 solo ya es muy fuerte; BM25 aporta poco por sí
mismo y la fusión ingenua puede restar. El re-ranker repara el daño del híbrido y mejora algo el orden en el conjunto
principal, pero frente al vectorial solo empata en Recall@5 y no se ve ventaja en las preguntas nuevas.

### 1.2 Sinónimos coloquiales: v1 (antes de ver resultados) → v2 (tras este análisis)

| Recall@5 | BM25 sin / v1 / v2 | Híbrido sin / v1 / v2 | Vectorial sin / v1 / v2 |
|---|---|---|---|
| Principal (100) | 0,69 / 0,72 / **0,76** | 0,86 / 0,90 / **0,92** | 0,96 / 0,97 / 0,97 |
| Nuevas (20) | 0,65 / 0,75 / 0,75 | 0,80 / 0,90 / 0,90 | 0,85 / 0,90 / 0,90 |

Errores que motivaron la v2 (todos del conjunto principal):

- **P049** «¿Puedo renunciar a mis derechos laborales firmando un documento?» → la v1 agregaba «desahucio aviso
  terminación voluntaria» (renunciar al *trabajo*) y el art. 4 desaparecía de todos los métodos. La v2 excluye
  «renunciar a mis/los derechos».
- **P023/P024** «décimo tercero»: la v1 lo traducía a «decimotercera remuneración», pero el epígrafe del art. 111 dice
  «**décima tercera** remuneración o bono navideño». La v2 agrega ambas formas.
- **P089** «¿Qué es un accidente de trabajo?»: la v1 agregaba «riesgos indemnización» y BM25 bajaba el art. 348 de la
  posición 4 a la 12. Igual con «sindicato → … organización» (P091: 2 → 10) y «multas → sanciones» (P051: 10 → 14).
  Expandir con palabras de más *diluye* la consulta: en la v2 cada sinónimo agrega lo mínimo.

**¿Sobreajuste?** Sí, en parte: la v2 mejora el conjunto principal (donde se diagnosticó) y **no cambia nada** en las 20
preguntas nuevas, que no se usaron para ajustar. La v1 sí ayudó en ambos conjuntos (BM25 nuevas 0,65 → 0,75), así que
la idea funciona; las correcciones finas no se generalizan con tan pocas preguntas (cada una vale 5 puntos en «nuevas»).

### 1.3 Lo que no encuentra ningún método

- **P034** «¿Cada cuánto deben pagarme el sueldo como mínimo?» → el art. 83 (*Plazo para pagos*) no entra a los 5 del
  contexto; ganan los arts. 81 y 80 (definiciones de sueldo y salario), con un mejor puntaje del re-ranker de solo 0,51.
- **P057** «¿Cuántos días de licencia dan por la muerte de un familiar cercano?» → está en el numeral 30 del art. 42
  (*Obligaciones del empleador*, 35 numerales). El LLM contestó «no se especifica» citando el art. 378; con el umbral
  (mejor puntaje 0,009) pasa a ser abstención.
- **N006** «¿El jefe me puede cobrar intereses por un adelanto de sueldo?» → la respuesta es el literal e) del art. 44
  (*Prohibiciones al empleador*), un artículo con muchos literales: su embedding representa «prohibiciones» en
  general, no ese literal.
- **N011** «¿Cuánto tiempo tiene el empleador para despedir por una falta?» → art. 636 (*prescripciones especiales*): la
  pregunta no usa «prescripción» y gana el art. 172 (visto bueno).

## 2. Generación (qwen2.5:7b, 5 artículos de contexto)

| | Principal | Nuevas |
|---|---:|---:|
| Cita el artículo correcto | 87 % | 85 % |
| … cuando el artículo correcto estaba en el contexto | 90 % | 100 % |
| Se abstiene sin deber (solo LLM → con umbral) | 6 % → 7 % | 5 % → 5 % |
| Rechaza trampas (solo LLM → con umbral 0,098) | 95 % → **100 %** (en muestra: τ se eligió aquí) | 80 % → 80 % |
| Fidelidad juzgada a mano: fieles / con afirmación falsa | 90 % / 3 % (n = 30) | 65 % / **20 %** (n = 20) |

### 2.1 Cita exacta no es respuesta correcta

El caso más importante del proyecto, visible en la demo (`reports/app/01_cita_clicable_y_error_de_calculo.jpg`):

> **N001** «Me botaron del trabajo después de 6 años, ¿cuántos sueldos me tienen que dar?»
> *«De acuerdo con el artículo 188 del Código del Trabajo, tendrías derecho a 3 meses de remuneración por cada año de
> servicio, lo que equivaldría a 18 sueldos.»*

La cita es correcta (cuenta como acierto en «cita exacta») pero el cálculo es falso: el art. 188 da tres meses hasta
tres años de servicio y **un mes por año** después, o sea 6 meses. Patrón repetido en preguntas que piden **calcular o
combinar**:

- **P002** (8 años, días extra de vacaciones): «totalizando siete días extra» (son 3).
- **N004** (gemelos, licencia del padre): «30 días… se extiende por 10 días adicionales»: mezcla la extensión de la
  madre (10 días) con la del padre (15 + 5 = 20).
- **N012** presenta como generales los límites del contrato de aprendizaje que el art. 157 fija para adolescentes.
- **N016** inventa que la renuncia por correo necesita que el empleador la acepte.

Por eso la app muestra el texto del artículo al lado de la respuesta (cita clicable) y el README no presenta la cita
exacta como «exactitud de la respuesta». La fidelidad en «nuevas» (65 %) es más baja que en la muestra del principal
(90 %) porque esas preguntas tienen más cálculos; con n = 20 y n = 30 los intervalos son amplios.

### 2.2 Atribución al artículo vecino

- **P004** «¿Quién decide si los días adicionales se gozan o se pagan?»: el contexto traía el art. 70 primero (re-ranker
  0,998) y el modelo respondió bien… citando el **art. 69**. Le pasó también en P015 (cita el 47 en vez del 55).
- **P097** atribuye el art. 326 al «Código del Trabajo» cuando es de la Constitución (el encabezado del fragmento lo
  decía).

### 2.3 Abstenciones indebidas con el artículo correcto en primer lugar

**P052** (art. 44, puntaje 0,92) y **P072** (art. 188, 0,97): el modelo dijo «no cubierto». En P072 la confianza fue
«baja»: el modelo duda con el artículo de despido largo y con escala. Es el costo de un prompt que castiga inventar.

### 2.4 La única cita inventada, y el filtro que la atrapó

**P099** «¿Qué contingencias cubre el seguro universal obligatorio?»: el contexto traía `[CRE-369]` (Constitución) y
el modelo citó **`CT-369`** (otro artículo, del Código). El filtro de citas la eliminó; sin citas válidas, la respuesta
se convirtió en «no cubierto». Mejor perder una respuesta que mostrar una cita equivocada.

### 2.5 Trampas

- **T001** «¿Cuántos años de aportes al IESS necesito para jubilarme por vejez?» → el LLM respondió con la jubilación
  **patronal** del art. 216 (25 años). El mejor puntaje del re-ranker era **0,038**: el umbral 0,098 la rechaza sin
  llamar al LLM (`reports/app/02_rechazo_por_umbral.jpg`).
- **T021** (nuevas) «¿Cuánto es el aporte patronal al IESS?» → respondió con el art. 34.3: el Estado cubre el aporte
  patronal de los contratos juveniles por un año. Es una trampa discutible (la ley sí habla de aporte patronal en ese
  caso), pero no responde el porcentaje general, que está en la Ley de Seguridad Social. Re-ranker 0,99: ningún umbral
  la atrapa. Es la única trampa fallida de «nuevas» (rechazo 4 de 5; trampas cercanas 2 de 3).
- **T005** (vacaciones según la LOSEP) y **T023** (vacaciones de los policías) se parecen mucho al art. 69 (re-ranker
  0,67 y 0,48) y aun así el LLM los rechazó: el prompt de «solo con los fragmentos» funciona cuando la pregunta nombra
  otra ley o institución.
- Las 13 trampas lejanas (IVA, RUC, multas de tránsito, otro país) se rechazaron todas.

**El umbral:** se eligió en el principal con `max(cita_exacta + 0,5 × rechazo_correcto)`: atrapa T001 y convierte en
abstención una respuesta que ya era incorrecta (P057), sin perder citas correctas; el costo es una abstención
indebida más (6 % → 7 %). Como se ajustó ahí, el 100 % de rechazo del principal es en muestra. En «nuevas», donde no se
ajustó, no cambia nada (80 %): la trampa que falla (T021) tiene puntaje 0,99, porque se parece de verdad a un
artículo laboral.

## 3. Qué haría después

1. **Calculadora de beneficios en código** (indemnización, vacaciones, décimos) con el artículo como fuente y el LLM
   solo para entender la pregunta: los errores de 2.1 son aritmética, no recuperación.
2. **Ventana del artículo largo también en el embedding** (sub-fragmentos por numeral con el mismo id) para casos como
   N006.
3. **Cargar la Ley de Seguridad Social y la LOSEP** o detectar «IESS», «policía», «servidor público» y avisar el
   alcance antes de responder.
4. Un modelo de 14B (cabe en la PC) para comparar fidelidad, a ~2× el tiempo.
