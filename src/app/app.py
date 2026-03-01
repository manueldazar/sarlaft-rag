"""
Fase 6 — Interfaz Streamlit del Asistente Normativo SARLAFT.

Inicio:
    streamlit run src/app/app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — permite importar desde la raíz del proyecto
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.generator import RAGPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

APP_TITLE    = "Asistente Normativo SARLAFT"
APP_SUBTITLE = "Consulta inteligente de la normativa SFC sobre prevención de lavado de activos y financiación del terrorismo"

DISCLAIMER = (
    "⚠️ **Herramienta de apoyo — No sustituye asesoría jurídica profesional.** "
    "Las respuestas se generan automáticamente a partir de la Circular Básica Jurídica de la SFC "
    "(Parte I, Título IV, Capítulo IV — SARLAFT). Verificá siempre la normativa vigente y "
    "consultá con un profesional en derecho financiero antes de tomar decisiones de cumplimiento."
)

EXAMPLE_QUERIES = [
    "¿Qué es una operación sospechosa y cuándo debe reportarse a la UIAF?",
    "¿Cuáles son las etapas del SARLAFT?",
    "¿Qué funciones tiene el oficial de cumplimiento?",
    "¿Cuándo aplican procedimientos simplificados de conocimiento del cliente?",
    "¿Qué son las personas expuestas políticamente (PEP)?",
    "¿Qué reportes deben enviarse a la UIAF?",
    "¿Qué son las sanciones financieras dirigidas?",
]

CHUNK_TYPE_LABELS = {
    "definition": "📖 Definición",
    "section":    "📄 Sección",
    "intro":      "📝 Introducción",
}

# ---------------------------------------------------------------------------
# Cache del pipeline — se carga una sola vez por sesión del servidor
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo y base de datos normativa…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------

def render_chunk_card(chunk, container):
    """Renderiza un chunk recuperado como tarjeta expandible."""
    type_label = CHUNK_TYPE_LABELS.get(chunk.chunk_type, chunk.chunk_type)
    short_path = chunk.hierarchy_path[:55] + "…" if len(chunk.hierarchy_path) > 55 else chunk.hierarchy_path

    if chunk.expanded:
        label    = f"  ↳ §{chunk.section_id}"
        expanded = False   # hijos arrancan colapsados por defecto
    else:
        score_pct = chunk.score * 100
        label     = f"[{chunk.rank}] §{chunk.section_id}  —  {score_pct:.0f}% relevancia"
        expanded  = (chunk.rank == 1)

    with container.expander(label, expanded=expanded):
        if chunk.expanded:
            st.caption(f"↳ Expansión de §{chunk.expanded_from}")
        else:
            # Barra de relevancia solo para chunks del top-k original
            st.progress(chunk.score, text=f"Similitud coseno: {chunk.score:.3f}")

        # Metadata en columnas compactas
        m1, m2 = st.columns(2)
        m1.caption(f"**Tipo:** {type_label}")
        m2.caption(f"**Profundidad:** {chunk.depth}")
        m1.caption(f"**Actualización:** {chunk.last_updated}")
        m2.caption(f"**Caracteres:** {chunk.char_count:,}")

        st.caption(f"**Ruta:** {short_path}")

        # Texto del chunk (primeros 400 chars para no saturar el panel)
        preview = chunk.raw_content.strip()
        if len(preview) > 400:
            st.markdown(f"> {preview[:400]}…")
        else:
            st.markdown(f"> {preview}")


def render_sources_panel(chunks, container):
    """Panel completo de fuentes en la columna derecha."""
    container.markdown(f"### 📋 Fuentes consultadas")
    container.caption(f"{len(chunks)} fragmento{'s' if len(chunks) != 1 else ''} de la normativa")
    container.divider()
    for chunk in chunks:
        render_chunk_card(chunk, container)


def score_color(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    if score >= 0.55:
        return "🟡"
    return "🔴"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Renderiza el sidebar y devuelve la configuración seleccionada."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")

        k = st.slider(
            "Fragmentos a consultar (k)",
            min_value=3,
            max_value=10,
            value=5,
            help="Número de fragmentos normativos recuperados para generar la respuesta. "
                 "Mayor k = más contexto, respuestas más lentas.",
        )

        st.divider()

        st.markdown("### 💡 Consultas de ejemplo")
        st.caption("Pulsa para cargar en el área de consulta:")
        example_choice = None
        for q in EXAMPLE_QUERIES:
            if st.button(q[:52] + "…" if len(q) > 52 else q, use_container_width=True, key=f"ex_{q[:20]}"):
                example_choice = q

        st.divider()

        st.markdown("### ℹ️ Sobre el sistema")
        st.markdown(
            "**Corpus:** CBJ SFC, Parte I, Título IV, Capítulo IV — SARLAFT  \n"
            "**Chunks:** 327 fragmentos jerárquicos  \n"
            "**Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2`  \n"
            "**Modelo:** Llama 3.1 8B (vía Ollama)  \n"
            "**Vector DB:** ChromaDB (cosine similarity)  \n"
            "**Métricas:** Hit Rate@5 = 100% · MRR = 0.82"
        )

        st.divider()

        st.warning(DISCLAIMER)

    return {"k": k, "example_choice": example_choice}


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS mínimo para mejorar legibilidad
    st.markdown(
        """
        <style>
        .stExpander { border-left: 3px solid #e8a800; }
        .disclaimer-box {
            background: #fff3cd;
            border: 1px solid #e8a800;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    config = render_sidebar()
    k = config["k"]
    example_choice = config["example_choice"]

    # Estado de la sesión
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    if example_choice:
        st.session_state.query_input = example_choice

    # ---------------------------------------------------------------------------
    # Encabezado
    # ---------------------------------------------------------------------------
    st.markdown(f"# ⚖️ {APP_TITLE}")
    st.caption(APP_SUBTITLE)

    st.markdown(
        f'<div class="disclaimer-box">{DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ---------------------------------------------------------------------------
    # Formulario de consulta
    # ---------------------------------------------------------------------------
    with st.form("query_form", clear_on_submit=False):
        query_input = st.text_area(
            "Tu consulta:",
            value=st.session_state.query_input,
            height=100,
            placeholder="Ej: ¿Qué es una operación sospechosa y cuándo debe reportarse a la UIAF?",
            label_visibility="visible",
        )
        submitted = st.form_submit_button(
            "🔍  Consultar",
            type="primary",
            use_container_width=False,
        )

    # Actualizar session_state con lo que escribió el usuario
    if query_input:
        st.session_state.query_input = query_input

    # ---------------------------------------------------------------------------
    # Procesamiento y resultados
    # ---------------------------------------------------------------------------
    if not submitted or not query_input.strip():
        if not submitted:
            st.info("Escribe tu consulta y pulsa **Consultar** para obtener una respuesta.")
        return

    query = query_input.strip()
    pipeline = get_pipeline()

    # Retrieval (síncrono) + abrir conexión streaming
    with st.spinner("Consultando normativa…"):
        token_iter, retrieval = pipeline.stream(query, k=k)

    st.divider()

    # Layout de resultados: respuesta (izq.) | fuentes (der.)
    col_answer, col_sources = st.columns([3, 2], gap="large")

    # ---------------------------------------------------------------------------
    # Panel de fuentes — se renderiza ANTES del streaming para que aparezca
    # junto con los primeros tokens de la respuesta
    # ---------------------------------------------------------------------------
    if retrieval.chunks:
        render_sources_panel(retrieval.chunks, col_sources)
    else:
        col_sources.info("No se encontraron fragmentos relevantes para esta consulta.")

    # ---------------------------------------------------------------------------
    # Respuesta con streaming
    # ---------------------------------------------------------------------------
    with col_answer:
        st.markdown("### Respuesta")

        t0 = time.perf_counter()

        # Filtrar tokens vacíos que Ollama puede emitir al inicio/fin
        def clean_stream(gen):
            for token in gen:
                if token:
                    yield token

        answer = st.write_stream(clean_stream(token_iter))
        elapsed = time.perf_counter() - t0

        # Métricas de la respuesta
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("⏱️ Tiempo", f"{elapsed:.1f}s")
        m2.metric("📄 Fragmentos", k)
        m3.metric(
            "🎯 Top score",
            f"{retrieval.chunks[0].score:.2f}" if retrieval.chunks else "—",
        )

        if retrieval.chunks:
            top_score = retrieval.chunks[0].score
            quality_label = (
                "Alta relevancia" if top_score >= 0.75
                else "Relevancia moderada" if top_score >= 0.55
                else "Relevancia baja — verificar manualmente"
            )
            st.caption(
                f"{score_color(top_score)} {quality_label}  ·  "
                f"Modelo: `llama3.1:8b`  ·  k={k}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
