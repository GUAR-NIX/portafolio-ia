# Datos de prueba

`muestra_pequena.jsonl` contiene **8 reseñas** (título y texto) de *Amazon Reviews 2023* (McAuley Lab, UC San Diego;
Hou et al., arXiv:2403.03952), categoría *Cell Phones and Accessories*. Son una muestra mínima para que las pruebas
corran sin descargar nada: reproducen la grabación del LLM y comprueban que las citas etiquetadas son literales.

**Licencia:** esos datos se publican para uso académico y de investigación, no bajo MIT. Esta muestra se incluye solo
como fixture de prueba y con atribución; el resto de las 700 reseñas etiquetadas no se versiona (ver el README). Cada
reseña coincide con la huella SHA-1 de `data/etiquetas/indice_muestra.jsonl`, y una prueba lo comprueba.
