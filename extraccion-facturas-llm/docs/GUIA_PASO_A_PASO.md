# Guía paso a paso (Windows)

## 0 · Requisitos (una sola vez)

1. **Python 3.10+** (probado con 3.14).
2. **Ollama** → [ollama.com/download](https://ollama.com/download) y el modelo:

```powershell
ollama pull qwen2.5:7b      # 4,7 GB; corre en CPU, sin GPU
```

3. **Windows 10/11 con el idioma español** (viene así en un Windows en español). El OCR usa el motor nativo de Windows
   (`Windows.Media.Ocr`): no hay que instalar Tesseract ni descargar modelos. Para comprobarlo:

```powershell
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages     # debe aparecer es-ES o es-MX
```

Si no aparece: *Configuración → Hora e idioma → Idioma → Español → Opciones → Reconocimiento óptico de caracteres*.
En Linux/Mac el proyecto funciona igual con los PDF digitales; las facturas escaneadas quedan sin texto y la app lo avisa.

## 1 · Todo en un comando

Con Ollama abierto:

```powershell
cd extraccion-facturas-llm        # la carpeta del proyecto, donde lo hayas clonado
powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1                     # 50 facturas con LLM (~2 h en CPU)
powershell -ExecutionPolicy Bypass -File .\evaluar_en_mi_pc.ps1 -NPorPlantilla 4    # 20 facturas (~40 min)
```

Crea `.venv`, genera las 200 facturas, corre las pruebas, extrae con el LLM y calcula métricas y figuras.
La extracción es **reanudable**: si la cortas, vuelve a correr el script y sigue donde quedó (cada llamada al modelo
queda grabada en `data\grabacion_llm.jsonl`).

## 2 · Paso a paso (lo mismo, a mano)

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python generar_datos.py                       # 200 PDF + verdad JSON en data\raw (~10 s)
pytest -q                                     # pruebas, sin Ollama
python -m facturas.texto data\raw\facturas\FAC_061.pdf        # ver el texto (esta está escaneada y girada)
python -m facturas.extractor data\raw\facturas\FAC_002.pdf    # una factura con el LLM → JSON + validaciones
$env:FACTURAS_GRABAR = "1"; python -m facturas.correr_llm     # muestra de evaluación con el LLM
python -m facturas.evaluar                    # métricas → reports\evaluacion.json + figuras
```

## 3 · La app

```powershell
streamlit run app\streamlit_app.py            # http://localhost:8501
```

Sube un PDF o elige una de las 10 facturas de ejemplo. Verás la imagen, el texto extraído (pypdf u OCR), el JSON del
LLM al lado del de la regex, las validaciones de negocio y, en las de ejemplo, si cada campo es correcto.

**Sin Ollama** (para quien clone el repo): en la barra lateral elige *grabado*, o
`$env:FACTURAS_PROVEEDOR = "grabado"; streamlit run app\streamlit_app.py`. Las facturas de ejemplo reproducen las
respuestas grabadas durante la evaluación.

## 4 · Variables útiles

| Variable | Por defecto | Para qué |
|---|---|---|
| `FACTURAS_MODELO` | `qwen2.5:7b` | probar `qwen2.5:3b` (≈2× más rápido) o `qwen2.5:14b` |
| `FACTURAS_PROVEEDOR` | `ollama` | `grabado` = demo sin Ollama |
| `FACTURAS_REINTENTOS` | `1` | reintentos con corrección cuando falla una validación |
| `FACTURAS_GRABAR` | `0` | `1` = guardar/reutilizar cada llamada (evaluación reanudable) |

Si cambias el modelo, la grabación no sirve (la huella incluye el nombre del modelo): borra
`reports\predicciones_llm.jsonl` antes de volver a correr `facturas.correr_llm`.

## 5 · Agregar un formato de factura nuevo

1. Escribe una función `_f_mi_formato(c, f, x)` en `src\facturas\generar.py` y agrégala a `PLANTILLAS` y a `dibujar`.
2. `python generar_datos.py` → `python -m facturas.correr_llm` → `python -m facturas.evaluar`.
3. El LLM no necesita cambios; la regex sí (y eso es justamente lo que mide la plantilla E).

## 6 · Subir a GitHub

1. Crea el repo **`extraccion-facturas-llm`** (público, sin README).
2. En PowerShell, dentro de la carpeta:

```powershell
git init
git add .
git commit -m "Extracción de facturas PDF a JSON validado con LLM local"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/extraccion-facturas-llm.git
git push -u origin main
```

`data\raw` (los 200 PDF, 12 MB) no se sube: se regenera idéntico con `python generar_datos.py`.

---

## 🎤 Preguntas de entrevista (con respuesta)

**1. ¿Por qué un LLM y no solo expresiones regulares?**
La regex es perfecta en los formatos para los que se escribió (100 % de facturas correctas en las digitales de las
plantillas A–D) y cuesta 0 s, pero se rompe cuando cambia el formato: en la plantilla E, que no miré al escribirla,
no saca ni un ítem. En un área contable llegan facturas de cientos de proveedores; mantener una regex por proveedor no
escala. El LLM generaliza a formatos nuevos sin tocar código. Por eso el diseño final es una **cascada**: regex
primero y LLM solo cuando la regex no pasa las validaciones.

**2. ¿Cómo evitas que el modelo invente datos?**
Tres capas. (a) **Salida estructurada**: Ollama recibe el JSON Schema de pydantic y solo puede generar JSON con esos
campos y tipos. (b) **Validaciones de negocio** que no dependen del modelo: dígito verificador del RUC (módulo 10/11),
dígito verificador de la clave de acceso de 49 dígitos (módulo 11) y que esa clave contenga el mismo RUC, fecha y número;
que las líneas sumen el subtotal; que subtotal − descuento + IVA = total; que el IVA sea el 12 % o el 15 % según la
fecha. (c) Lo que no cuadra se le devuelve al modelo una vez y, si sigue mal, se marca **para revisión humana**.
Además mido la **abstención**: en los campos que no están impresos, ¿devuelve null o inventa?

**3. ¿Qué errores NO detectan las validaciones?**
Los de texto libre: nombres. En una factura el modelo intercambió el nombre del emisor y el del cliente; todos los
montos cuadraban y la factura pasó como válida. Por eso reporto los **errores silenciosos** (facturas aprobadas con
algún error) por separado. En producción agregaría una regla cruzada contra el catastro de RUC del SRI (el RUC del
emisor debe corresponder a esa razón social).

**4. ¿Por qué el orden de los campos del esquema importa?**
Ollama genera el JSON en el orden del esquema, token por token. Lo puse en el orden en que aparecen en una factura
(emisor → número → autorización → fecha → cliente → ítems → totales): el modelo lee y escribe en el mismo sentido.
Es la lección del RAG de normativa aduanera (proyecto IA), donde un orden distinto entre ejemplos y esquema costó 40 puntos.

**5. ¿Por qué evaluaste el LLM con 50 facturas y no con las 200?**
Costo: en un CPU compartido con otros procesos cada llamada tarda 1–3 min (≈5 tokens/s de generación; los 49 dígitos
de la clave de acceso son casi un token por dígito). 200 facturas eran ~8 h. Tomé una muestra **estratificada**:
10 por plantilla, sobrerrepresentando escaneadas y con campos faltantes, y procesadas intercaladas para que un corte no
deje plantillas sin evaluar. La regex sí se midió en las 200. Con 50 facturas cada una vale 2 puntos: diferencias
pequeñas no son concluyentes.

**6. ¿Cómo hiciste el OCR sin instalar nada?**
Windows trae un motor de OCR (`Windows.Media.Ocr`, el de la app Recortes) accesible desde Python con `winrt`.
Tarda ~0,3 s por página en CPU. No corrige páginas giradas 90°/180°: si no encuentro palabras típicas de factura
(RUC, TOTAL, IVA…), pruebo las otras tres orientaciones y me quedo con la mejor; acertó en las 7 giradas. Luego
rearmo las filas por coordenada vertical para que una fila de la tabla siga siendo una línea.

**7. ¿Por qué `extraction_mode="layout"` en pypdf?**
El modo normal pone cada celda de la tabla en su propia línea: el modelo ya no sabe qué cantidad va con qué precio.
El modo layout respeta posiciones; comprimo los huecos largos a 4 espacios para ahorrar tokens.

**8. ¿Qué es la clave de acceso y por qué es tan útil?**
Es el número de autorización de la factura electrónica del SRI: 49 dígitos que contienen fecha, tipo de comprobante,
RUC, ambiente, serie, secuencial, código y un dígito verificador módulo 11. Validarla detecta un dígito mal copiado
(OCR) y cruzarla con los otros campos detecta si el modelo tomó el RUC del cliente en vez del emisor.

**9. ¿Por qué datos sintéticos?**
Las facturas reales tienen datos personales y tributarios: no se pueden publicar. Generarlas da la **verdad exacta**
de cada campo (imposible con facturas reales sin etiquetado manual) y permite controlar la dificultad: 5 diseños,
formatos de fecha y número distintos, escaneadas, giradas y con campos faltantes. La limitación: son más limpias que
las reales (sin sellos, firmas, manchas ni letra a mano); las cifras son un techo optimista.

**10. ¿Cómo lo llevarías a producción?**
Cascada regex → LLM; cola de revisión humana para las marcadas; guardar las correcciones humanas como nuevos ejemplos;
GPU o un modelo más pequeño para bajar la latencia de ~2 min a segundos; y leer el XML de la factura electrónica cuando
exista (el SRI lo entrega): el PDF es para los casos en que solo llega el RIDE impreso o escaneado.
