# Portafolio de IA aplicada con LLM locales

RAG híbrido, extracción estructurada, clasificación, agentes con herramientas y resúmenes verificados, todo con modelos locales y evaluación honesta.

**Herramientas principales:** Ollama (qwen2.5:7b, bge-m3) · pydantic · FastAPI · Streamlit

Cada carpeta es un proyecto independiente, con su propio README (resultados primero), guía paso a paso, pruebas automáticas y cómo reproducirlo. Los datos crudos no se versionan: cada proyecto trae un script que los descarga desde la fuente oficial.

## Proyectos

| Proyecto | De qué trata |
|---|---|
| [**🌐 Agente de datos abiertos con herramientas: Banco Mundial + Open-Meteo**](agente-datos-abiertos/) | Un LLM local (qwen2.5:7b en Ollama, CPU) responde preguntas sobre economía y clima de América Latina llamando a 5 herramientas conectadas al Banco Mundial (WDI) y a Open-Meteo (ERA5). El proyecto mide por separado lo que decide… |
| [**📨 Enrutamiento de quejas financieras: TF-IDF vs. embeddings vs. LLM, cascada y urgencia**](clasificador-quejas-cfpb/) | ¿Hace falta un LLM local para enrutar quejas financieras a la mesa correcta, o basta un TF-IDF + regresión logística? Este proyecto lo mide con 50.568 quejas reales del CFPB (Consumer Financial Protection Bureau, EE. UU.; textos… |
| [**🧾 Extracción de facturas PDF a JSON validado (LLM local + reglas del SRI)**](extraccion-facturas-llm/) | Convierte facturas ecuatorianas en PDF (digitales, escaneadas o giradas) en un JSON estructurado y validado con reglas de negocio: RUC y clave de acceso con dígito verificador, ítems que suman el subtotal, subtotal − descuento +… |
| [**⚖️ Asistente RAG de la ley laboral del Ecuador (búsqueda híbrida + re-ranking + rechazo)**](rag-ley-laboral-ecuador/) | Responde preguntas de trabajadores y pequeñas empresas (vacaciones, décimos, horas extra, despido, liquidación) citando el artículo exacto del Código del Trabajo (607 artículos, última reforma 2026-04-09) y de la Constitución (20… |
| [**🔎 Sentimiento por aspecto en reseñas de Amazon: reglas vs embeddings vs LLM local**](resenas-sentimiento-aspectos/) | ¿De qué se quejan los clientes de teléfonos y accesorios, y cuánto les cuesta a las marcas en estrellas? Para cada reseña el sistema extrae los aspectos que menciona (batería, carga, pantalla, sonido, calidad, ajuste, diseño,… |
| [**📊 Resúmenes ejecutivos de KPIs con verificación de cada cifra**](resumen-kpis-verificado/) | Un LLM local (qwen2.5:7b en Ollama, CPU) redacta para gerencia el resumen económico de un país de América Latina a partir de los indicadores del Banco Mundial, y un verificador automático comprueba cada número del texto contra… |

## Cómo usar este repositorio

```bash
git clone https://github.com/GUAR-NIX/portafolio-ia.git
cd portafolio-ia/<proyecto>
```

Dentro de cada proyecto, sigue la sección «Cómo reproducir» de su README y `docs/GUIA_PASO_A_PASO.md`.

## Autor

**Erick Guarnizo** · [LinkedIn](https://www.linkedin.com/in/erick-g-571a7b3b4)

## Licencia

El código está bajo licencia [MIT](LICENSE). Los datos de cada proyecto conservan la licencia de su fuente, indicada en su README.
