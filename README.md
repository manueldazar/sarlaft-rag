# Asistente Normativo SARLAFT

Sistema de consulta regulatoria basado en RAG (Retrieval-Augmented Generation) para la normativa SARLAFT de la Superintendencia Financiera de Colombia. Responde preguntas en lenguaje natural sobre el Sistema de Administración del Riesgo de Lavado de Activos y Financiación del Terrorismo, citando la sección normativa exacta de cada respuesta.

**Usuario objetivo:** Oficiales de cumplimiento, equipos legales y fundadores de fintechs colombianas que necesitan resolver consultas regulatorias con precisión y trazabilidad.

---

## Demo

## Demo

### Definición: "¿Qué es la financiación del terrorismo?"
![Definición](docs/images/demo_definicion.png)

### Procedimiento: "¿Cuáles son los deberes respecto del funcionario responsable?"
![Procedimiento](docs/images/demo_procedimiento.png)

### Fuera de scope: "¿Cuál es la TRM del dólar hoy?"
![Fuera de scope](docs/images/demo_fuera_scope.png)

| Caso | Descripción |
|---|---|
| 📖 Definición | "¿Qué es la financiación del terrorismo?" → §1.16 con cita exacta |
| 📋 Procedimiento | "¿Cuáles son los deberes respecto del funcionario responsable?" → §2.1.2 + expansión hijos §2.1.2.1–2.1.2.5 |
| 🚫 Fuera de scope | "¿Cuál es la TRM del dólar hoy?" → declara limitación sin alucinar |

---

## Métricas de evaluación

Evaluación sobre 30 preguntas (29 in-corpus + 1 fuera de scope), 3 niveles de dificultad (fácil / medio / difícil).

### Retrieval (k=5)

| Métrica | Resultado |
|---|---|
| Hit Rate@5 | **100%** |
| MRR | **0.81** |
| Recall@5 | **93.1%** |
| Precision@3 | 49.4% |
| Context Relevance@5 | 67.0% |

### Generación (Llama 3.1 8B, promedio 9.3s/query)

| Métrica | Resultado |
|---|---|
| Cita normativa presente | **96.5%** |
| Cita de sección correcta | **86.2%** |
| Faithfulness (sin alucinación) | **96.5%** |
| Detección out-of-scope | **100%** |

---

## Arquitectura

```
Consulta del usuario
        │
        ▼
┌───────────────────┐
│  Embedding query  │  paraphrase-multilingual-MiniLM-L12-v2
└───────────────────┘
        │
        ▼
┌───────────────────┐
│     ChromaDB      │  Búsqueda por similitud coseno → top-k chunks
│  (341 chunks)     │
└───────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Parent-child expansion       │  Si un chunk recuperado es un encabezado
│  (retriever.py)               │  con hijos, se expanden automáticamente
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Construcción de prompt       │  Contexto con prefijo jerárquico
│  (generator.py)               │  [Fuente:][Sección:][Ruta:]
└───────────────────────────────┘
        │
        ▼
┌───────────────────┐
│  Ollama           │  Llama 3.1 8B Q4_K_M · temp=0.1 · local
└───────────────────┘
        │
        ▼
   Respuesta con cita normativa exacta
```

### Chunking jerárquico

El corpus principal (CBJ SFC, Parte I, Título IV, Capítulo IV) tiene **10 niveles de profundidad jerárquica**. El parser produce **341 chunks** respetando la estructura de secciones del documento legal.

Cada chunk incluye:
- `section_id`: identificador jerárquico (e.g., `4.2.2.2.1.4`)
- `hierarchy_path`: ruta completa desde la raíz del documento
- `depth`: nivel de profundidad (0–9)
- `chunk_type`: `definition` | `section` | `intro`
- `raw_content`: texto sin prefijo (usado para embeddings)
- `content_with_prefix`: texto con prefijo de contexto (enviado al LLM)

**Por qué no se usó chunking por tamaño fijo:** Un chunking de 512 tokens con overlap habría partido secciones normativas a la mitad y mezclado el contenido de numerales distintos. La estructura legal del SARLAFT es el dato, no un obstáculo; preservarla permite citas precisas y trazables.

---

## Decisiones de diseño

### 1. Chunking semántico por secciones legales

El parser segmenta por secciones del documento, no por tokens. Reglas de post-procesamiento:

| Regla | Razón |
|---|---|
| Chunks < 100 chars se fusionan con el hermano anterior | Embeddings poco informativos |
| Definiciones (§1.x) nunca se fusionan | Cada definición es una unidad consultable |
| Chunks con hijos nunca se fusionan | Son puntos de entrada semánticos para el retrieval |
| Chunks > 3000 chars se dividen en fronteras de párrafo | Ventana efectiva del modelo de embeddings |

La segunda regla se descubrió necesaria después del primer deployment: el chunk `§2.1.2. Deberes respecto del funcionario responsable` (95 chars) era absorbido por el hermano anterior, haciendo que ningún chunk contuviera los términos de esa consulta. Ver [LECCIONES_APRENDIDAS.md](LECCIONES_APRENDIDAS.md).

### 2. Separación raw_content / content_with_prefix

Los embeddings se generan sobre `raw_content` (texto limpio, sin prefijo). El prefijo jerárquico `[Fuente: CBJ SFC][Sección: 4.2.2.2.1][Ruta: ...]` se inyecta únicamente en el prompt al LLM. Esto evita que el ruido de metadatos degrade la similitud semántica en el espacio vectorial.

### 3. Parent-child retrieval

Documentos legales tienen un patrón recurrente: una sección contiene solo el título y la frase introductoria de una lista; el contenido detallado está distribuido en los subnumerales hijos. El sistema detecta estos encabezados (criterio: `char_count < 200` o el texto termina con `:`) y expande el contexto con los hijos directos antes de construir el prompt.

Esta es una implementación del patrón *small-to-big retrieval* (cf. LlamaIndex Auto-Merging Retriever). La expansión no consume slots del top-k original.

### 4. Prompt de generación con temperatura baja

`temperature=0.1` para maximizar determinismo en respuestas jurídicas. El prompt del sistema instruye al LLM a: (a) responder solo con base en el contexto recuperado, (b) citar la sección normativa exacta, (c) declarar explícitamente la limitación cuando la información no está en el corpus.

---

## Stack técnico

| Componente | Tecnología | Versión |
|---|---|---|
| Lenguaje | Python | 3.11+ |
| LLM | Llama 3.1 8B Q4_K_M vía Ollama | ollama 0.4.7 |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers 3.3.1 |
| Vector store | ChromaDB (cosine similarity) | 0.6.3 |
| Orquestación | LlamaIndex | 0.12.4 |
| Interfaz | Streamlit | 1.42.0 |

**Hardware de desarrollo:**
- CPU: Intel Core i7-12700H
- GPU: NVIDIA RTX 3070 Ti Laptop (8 GB VRAM)
- RAM: 32 GB
- OS: WSL2 Ubuntu 24.04 / Windows 11

---

## Estructura del proyecto

```
rag-fintech/
├── doc/
│   ├── 01/                   # CBJ SARLAFT principal + 8 anexos (.docx) + 6 formatos (.xls)
│   ├── 02/                   # Normativa primaria (Ley 526/1999, Decreto 663/1993, etc.)
│   ├── 03/                   # Resoluciones y circulares UIAF / Supersociedades / DIAN
│   └── 04/                   # Marco GAFI (FATF Recommendations 2012)
├── src/
│   ├── parser/
│   │   └── sarlaft_parser.py         # Parser jerárquico del documento principal
│   ├── indexer/
│   │   └── index_chunks.py           # Embeddings + ingesta en ChromaDB
│   ├── retriever/
│   │   └── retriever.py              # Retrieval con parent-child expansion
│   ├── generator/
│   │   └── generator.py              # Pipeline RAG + integración Ollama
│   ├── evaluator/
│   │   └── evaluator.py              # Métricas de retrieval y generación
│   └── app/
│       └── app.py                    # Interfaz Streamlit
├── data/
│   ├── chunks/
│   │   ├── chunks.json               # 341 chunks con metadata completa
│   │   └── parsing_report.txt        # Estadísticas del parser
│   └── chroma_db/                    # Persistencia de ChromaDB (gitignore)
├── eval/
│   ├── questions.json                # 30 preguntas de evaluación con secciones esperadas
│   └── results/                      # JSONs de resultados por timestamp
├── requirements.txt
├── .env                              # OLLAMA_HOST, CHROMA_DB_PATH, EMBEDDING_MODEL
├── LECCIONES_APRENDIDAS.md
└── README.md
```

---

## Instalación y uso

**Requisitos previos:** Python 3.11+, [Ollama](https://ollama.com) instalado y corriendo.

```bash
# 1. Clonar el repositorio
git clone <repo-url> && cd rag-fintech

# 2. Entorno virtual e instalación de dependencias
python -m venv .venv
source .venv/bin/activate        # WSL/Linux
pip install -r requirements.txt

# 3. Descargar el modelo LLM
ollama pull llama3.1:8b

# 4. Indexar el corpus (una sola vez; ~2 min en RTX 3070 Ti)
python -m src.indexer.index_chunks

# 5. Levantar la interfaz
streamlit run src/app/app.py
# → http://localhost:8501
```

Para re-indexar desde cero (e.g., tras actualizar chunks.json):

```bash
python -m src.indexer.index_chunks --reset
```

---

## Evaluación

El sistema incluye un evaluador en dos fases ejecutable por separado:

```bash
python -m src.evaluator.evaluator          # Solo retrieval (~10s)
python -m src.evaluator.evaluator --full   # Retrieval + generación (~5 min)
python -m src.evaluator.evaluator --report # Último reporte guardado
```

### Dataset

30 preguntas en `eval/questions.json`, cubren 10 temas: definiciones, etapas, elementos, gobernanza, conocimiento del cliente, reportes, sanciones, transferencias, metodología y capacitación. Incluye 1 pregunta fuera de corpus para medir la tasa de declaración de limitación.

Distribución por dificultad: 10 fáciles · 16 medias · 3 difíciles.

### Métricas de retrieval

- **Precision@k**: fracción de chunks recuperados que son relevantes para la pregunta.
- **Recall@k**: fracción de las secciones esperadas cubiertas en los top-k chunks.
- **MRR**: rango recíproco del primer chunk relevante (mide si el mejor chunk llega primero).
- **Hit Rate**: porcentaje de preguntas con al menos un chunk relevante en top-k.

La relevancia es jerárquica: un chunk de §4.2.2.2 es relevante para una pregunta que espera §4.2.2, y viceversa.

### Métricas de generación

- **Cita normativa**: ¿el LLM incluye al menos una referencia de sección en su respuesta?
- **Cita correcta**: ¿la sección citada corresponde a alguna de las secciones esperadas?
- **Faithfulness**: ¿el LLM solo referencia secciones que estaban en el contexto recuperado?
- **Detección out-of-scope**: ¿el LLM declara limitación para preguntas fuera del corpus?

---

## Limitaciones

**Corpus MVP.** El sistema indexa únicamente el cuerpo principal de la CBJ SFC (Parte I, Título IV, Capítulo IV — SARLAFT). Los 8 anexos técnicos (formatos de reporte a la UIAF) y la normativa complementaria (UIAF, decretos, resoluciones sectoriales) no están incluidos.

**Modelo 8B.** Llama 3.1 8B presenta alucinaciones ocasionales: en la evaluación, 1 de 30 queries obtuvo una cita de sección que no estaba en el contexto recuperado. Modelos más grandes reducen este comportamiento.

**No es asesoría jurídica.** Las respuestas se generan automáticamente y deben verificarse contra la normativa vigente antes de tomar decisiones de cumplimiento.

---

## Roadmap

- [ ] Indexar los 8 anexos técnicos de la CBJ SARLAFT (formatos ROS, RTE, etc.)
- [ ] Incorporar circular UIAF y normativa sectorial relevante (doc/03)
- [ ] Fine-tuning del modelo de embeddings con vocabulario regulatorio colombiano
- [ ] API REST (FastAPI) para integración con sistemas de compliance existentes
- [ ] Evaluación con modelos más grandes (Llama 3.1 70B vía API) para comparación de faithfulness

---

## Licencia

Este proyecto es de portafolio. El corpus normativo pertenece a la Superintendencia Financiera de Colombia y es de acceso público.
