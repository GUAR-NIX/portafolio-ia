# Fuentes de la normativa

Los PDF se bajan con `python descargar_datos.py` (reanudable) a `data/raw/` (no versionado). El script guarda URL,
fecha, tamaño y SHA-256 en [`data/fuentes_descarga.json`](../data/fuentes_descarga.json).

| Documento | Versión usada | Última reforma | Publicado por | URL | Descarga | SHA-256 (inicio) |
|---|---|---|---|---|---|---|
| **Código del Trabajo** | Codificación 2005-017, R.O. Suplemento 167 del 16-XII-2005 | **2026-04-09** (según la portada del propio PDF: «ÚLTIMA REFORMA: 2026-04-09») | Servicio Nacional de Contratación Pública (SERCOP), portal estatal; PDF generado el 2026-08-03 | <https://portal.compraspublicas.gob.ec/sercop/wp-content/uploads/2026/08/CODIGO-DEL-TRABAJO.pdf> | 2026-10-25 | `94d87989…` |
| **Constitución de la República del Ecuador** (20 artículos laborales y de seguridad social) | R.O. 449 del 20-X-2008 | **25-I-2021** («Última modificación: 25-ene.-2021», edición Lexis) | Ministerio de Defensa Nacional | <https://www.defensa.gob.ec/wp-content/uploads/downloads/2021/02/Constitucion-de-la-Republica-del-Ecuador_act_ene-2021.pdf> | 2026-10-25 | `6dd464c3…` |

Artículos de la Constitución incluidos: 33, 34 (trabajo y seguridad social), 43 (mujeres embarazadas), 47–48
(personas con discapacidad), 66 (derechos de libertad), 325–333 (formas de trabajo y su retribución), 367–371
(seguridad social).

## Por qué estas copias

- La copia del Ministerio del Trabajo (<https://www.trabajo.gob.ec/wp-content/uploads/2024/01/CODIGO_DEL_TRABAJO-CT.pdf>,
  Ediciones Legales) tiene como última reforma el Suplemento del R.O. 309 del **12-V-2023**: está desactualizada.
  La del SERCOP es la más reciente que se encontró en un portal del Estado (reformas hasta abril de 2026).
- El portal del SERCOP es lento (~2 KB/s y cortes): el script reanuda con `curl -C -` y reintentos.
- La Ley de Seguridad Social **no** se cargó (era opcional): por eso varias preguntas trampa son sobre el IESS.

## Licencia

Las normas jurídicas del Ecuador son textos oficiales de libre uso (el Código Orgánico de la Economía Social de los
Conocimientos excluye las disposiciones legales de la protección por derecho de autor). Las notas editoriales, concordancias y jurisprudencia que traen los
PDF se separan del texto de la ley y **no** se indexan ni se citan.

## Aviso

El proyecto usa el texto vigente a la fecha de última reforma indicada. Una reforma posterior puede cambiar las
respuestas. **No reemplaza la asesoría de un abogado.**
