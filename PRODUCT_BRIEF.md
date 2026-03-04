# Asistente de Consulta Regulatoria SARLAFT

**Inteligencia artificial aplicada a normativa financiera colombiana**

---

## Qué es

Un sistema de consulta regulatoria basado en inteligencia artificial que responde preguntas en lenguaje natural sobre la normativa SARLAFT (Sistema de Administración del Riesgo de Lavado de Activos y Financiación del Terrorismo) de la Superintendencia Financiera de Colombia, citando la sección normativa exacta en cada respuesta.

## Qué problema resuelve

Las fintechs colombianas enfrentan una convergencia regulatoria sin precedentes: SARLAFT, Open Finance (Decreto 1297/2022), Circular 002/2024 de la SFC (supervisión de modelos de IA), Ley 1581 de protección de datos y el Proyecto de Ley 043-2025 sobre IA. La mayoría de las empresas medianas (15-80 empleados) tiene un responsable de compliance pero no un equipo completo dedicado. Las alternativas actuales —buscar en Google, consultar abogados externos o esperar respuesta de la SFC vía elHub— son lentas, costosas o imprecisas.

## Cómo funciona

El sistema utiliza RAG (Retrieval-Augmented Generation) con un diseño específico para documentos legales:

- **Chunking jerárquico:** El corpus normativo se segmenta respetando la estructura de secciones del documento legal (341 segmentos, 10 niveles de profundidad), no por tamaño fijo de tokens. Esto preserva la integridad de cada numeral y permite citas precisas.
- **Retrieval con expansión parent-child:** Cuando una sección es un encabezado con subnumerales, el sistema expande automáticamente el contexto con los hijos relevantes antes de generar la respuesta.
- **Generación con citación obligatoria:** El modelo responde únicamente con base en el contexto recuperado, cita la sección normativa exacta, y declara explícitamente su limitación cuando la información no está en el corpus.

## Métricas de evaluación

Evaluación sobre 30 preguntas reales (3 niveles de dificultad):

| Métrica | Resultado |
|---|---|
| Hit Rate en retrieval (k=5) | **100%** |
| MRR (Mean Reciprocal Rank) | **0.81** |
| Recall (k=5) | **93.1%** |
| Fidelidad (sin alucinación) | **96.5%** |
| Cita normativa presente | **96.5%** |
| Cita de sección correcta | **86.2%** |
| Detección fuera de alcance | **100%** |

## Ejemplo de uso

**Pregunta:** "¿Cuáles son los deberes respecto del funcionario responsable?"

**Respuesta:** El sistema recupera §2.1.2 y expande automáticamente los subnumerales §2.1.2.1 a §2.1.2.5, entregando una respuesta completa con la referencia normativa exacta de la Circular Básica Jurídica de la SFC, Parte I, Título IV, Capítulo IV.

**Pregunta fuera de alcance:** "¿Cuál es la TRM del dólar hoy?"

**Respuesta:** El sistema declara explícitamente que esta información no está dentro de su corpus normativo, sin alucinar una respuesta.

## Cobertura normativa actual

**Incluido:**
- Cuerpo principal de la CBJ SFC — Parte I, Título IV, Capítulo IV (SARLAFT)

**No incluido (en desarrollo):**
- Anexos técnicos SARLAFT (formatos de reporte a la UIAF)
- Normativa complementaria UIAF, decretos y resoluciones sectoriales
- Circular 002/2024 SFC (supervisión de modelos de IA)
- Marco de Open Finance (Decreto 1297/2022, Circular 004/2024)
- Ley 1581/2012 (protección de datos personales)
- Proyecto de Ley 043-2025 (ley de IA)

## Qué NO es

- **No es asesoría jurídica.** Las respuestas se generan automáticamente y deben verificarse contra la normativa vigente antes de tomar decisiones de cumplimiento.
- **No reemplaza un abogado.** Es una herramienta de consulta rápida para acelerar el trabajo de equipos de compliance, no un sustituto del criterio legal profesional.

## Stack técnico

| Componente | Tecnología |
|---|---|
| LLM | Llama 3.1 8B vía Ollama |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector store | ChromaDB |
| Orquestación | LlamaIndex |
| Interfaz | Streamlit |

## Código abierto

El código fuente completo, incluyendo el parser jerárquico, el sistema de retrieval con expansión parent-child, y el evaluador de métricas, está disponible en GitHub:

**https://github.com/manueldazar/sarlaft-rag**

---

*Documento de referencia técnica — Marzo 2026*
*Contacto: Manuel Daza Ramírez — [linkedin.com/in/manueldazaramirez](https://www.linkedin.com/in/manueldazaramirez/)*
