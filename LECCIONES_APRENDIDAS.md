# Lecciones aprendidas y decisiones de diseño

## Proyecto: RAG SARLAFT — Asistente de consulta regulatoria

---

## 1. Chunking de documentos legales jerárquicos

### Problema descubierto: fusión de encabezados con hijos

**Contexto:** El post-procesamiento del parser fusionaba chunks menores a 100 caracteres con su hermano anterior para evitar chunks demasiado pequeños para generar embeddings útiles.

**Fallo:** Secciones como `2.1.2. Deberes respecto del funcionario responsable` (95 chars) fueron fusionadas y desaparecieron del índice. Pero estas secciones son encabezados estructurales que contienen el título semántico de una lista de items hijos (2.1.2.1 a 2.1.2.5). Sin el encabezado, ningún chunk contenía las palabras "deberes" + "funcionario responsable", y el retrieval fallaba para esa consulta.

**Regla implementada:** Los chunks que son padres de otros chunks nunca se fusionan, sin importar su tamaño. Son puntos de entrada semánticos para el retrieval.

**Regla previa confirmada:** Las definiciones (sección 1.x) nunca se fusionan, ya que cada definición es una unidad semántica autocontenida que debe ser recuperable individualmente.

**Lección:** En documentos legales jerárquicos, los encabezados de sección no son "ruido" — son los nodos que conectan el lenguaje natural de una consulta con el contenido técnico de los subnumerales. Eliminarlos rompe la cadena de retrieval.

### Problema descubierto: títulos truncados en definiciones

**Contexto:** El parser extraía los primeros 80 caracteres del texto como título cuando no había encabezado en negrita.

**Fallo:** Definiciones como `1.1. Agentes económicos: son todas las personas naturales o jurídicas...` producían títulos con la definición truncada en lugar del término solo.

**Regla implementada:** Para textos con dos puntos antes de la posición 80, el título se extrae hasta los dos puntos (el término). Para otros, se usan los primeros 80 caracteres.

### Práctica confirmada: chunking semántico > chunking por tamaño fijo

El documento SARLAFT tiene 10 niveles de profundidad jerárquica. Un chunking por tamaño fijo (e.g., 512 tokens con overlap) habría partido secciones lógicas a la mitad y mezclado contenido de numerales diferentes. El chunking por secciones legales preserva la estructura normativa y permite citas precisas.

---

## 2. Retrieval en documentos jerárquicos

### Problema identificado: parent-child retrieval gap

**Contexto:** Tras corregir el chunking, el retriever trae correctamente el encabezado `2.1.2. Deberes respecto del funcionario responsable`. Pero este chunk solo tiene 95 caracteres con el título y la frase introductoria. Los deberes concretos están en chunks hijos separados (2.1.2.1 a 2.1.2.5) que no aparecen en el top-k porque semánticamente son instrucciones específicas ("Designar al funcionario...", "Garantizar que..."), no "deberes".

**Patrón no implementado:** Parent-child retrieval (también llamado "small-to-big" en LlamaIndex). Cuando el retriever recupera un chunk que es un encabezado con hijos, debe expandir el contexto automáticamente incluyendo los chunks hijos.

**Esto es un patrón conocido en la literatura de RAG**, no una lección aprendida. Debió haberse incluido en el diseño original del retriever. Referencias:

- LlamaIndex: Auto-Merging Retriever / Hierarchical Node Parser
- LangChain: Parent Document Retriever
- Concepto general: "small-to-big retrieval" — indexar chunks pequeños para precisión, pero enviar al LLM el contexto ampliado (padre + hijos) para completitud.

**Implementación pendiente:** En el retriever, detectar chunks recuperados que son encabezados (char_count bajo + tienen hijos en el índice) y expandir con sus hijos directos antes de pasar el contexto al LLM.

---

## 3. Reglas de merge del parser (resumen)

| Regla | Justificación |
|---|---|
| Chunks < 100 chars se fusionan con hermano anterior | Chunks muy pequeños producen embeddings poco informativos |
| Definiciones (sección 1.x) nunca se fusionan | Cada definición es una unidad semántica consultable |
| Chunks con hijos nunca se fusionan | Son puntos de entrada semánticos para el retrieval jerárquico |
| Chunks > 3000 chars se dividen en fronteras de párrafo | Evitar exceder la ventana de contexto útil del modelo de embeddings |

---

## 4. Evaluación como herramienta de descubrimiento

La evaluación manual (revisar chunks_preview.txt, probar queries en la interfaz) descubrió problemas que las métricas automáticas no capturaron:

- **Hit Rate 100%** no significó que las respuestas fueran completas — el chunk correcto se recuperaba pero sin el contenido detallado.
- **MRR alto** no garantizó calidad de respuesta — el chunk top era el encabezado, pero el LLM necesitaba los hijos.
- **La prueba manual con la pregunta sobre deberes del funcionario responsable** reveló el gap de parent-child que las 30 preguntas del dataset automático no cubrían.

**Lección:** Las métricas automáticas de retrieval (hit rate, MRR, precision, recall) miden si el sistema encuentra chunks relevantes, no si el LLM recibe contexto suficiente para responder completamente. La validación manual con queries reales es indispensable.

---

## 5. Decisiones de diseño confirmadas

| Decisión | Estado |
|---|---|
| Chunking jerárquico por secciones legales | ✓ Validada — preserva citas y estructura |
| Embeddings sobre `raw_content` sin prefijo | ✓ Validada — el prefijo es para el LLM, no para similarity search |
| Prefijo de contexto jerárquico en cada chunk | ✓ Validada — permite al LLM citar la ruta normativa completa |
| Modelo de embeddings multilingüe generalista | ✓ Aceptable para MVP — limitaciones en vocabulario regulatorio |
| ChromaDB como vector store | ✓ Suficiente para 341 chunks |
| Evaluación con métricas de negocio (citas, faithfulness) | ✓ Crítica — diferencia un proyecto técnico de uno con valor real |
