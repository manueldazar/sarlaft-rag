"""
Fase 4 — Generación con Ollama + Llama 3.1 8B.

Pipeline completo:
    Query → Retriever (ChromaDB) → Top-k chunks → Prompt → Ollama → Respuesta con citas

Uso programático:
    from src.generator.generator import RAGPipeline
    pipeline = RAGPipeline()
    response = pipeline.query("¿Qué es una operación sospechosa?")
    print(response)

Uso desde terminal:
    python src/generator/generator.py --query "¿Qué es el beneficiario final?"
    python src/generator/generator.py --query "..." --k 7 --stream
    python src/generator/generator.py --batch          # 5 queries de demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv
import ollama

# Añadir raíz del proyecto al path para importar retriever
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever.retriever import Retriever, RetrievedChunk, RetrievalResult

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_K     = int(os.getenv("RAG_K", "5"))

# ---------------------------------------------------------------------------
# Prompt del sistema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Eres un asistente especializado en regulación SARLAFT (Sistema de Administración \
del Riesgo de Lavado de Activos y Financiación del Terrorismo) para entidades \
vigiladas por la Superintendencia Financiera de Colombia (SFC).

REGLAS OBLIGATORIAS:
1. Responde ÚNICAMENTE con base en los fragmentos normativos proporcionados en el \
contexto. No uses conocimiento externo ni inferencias más allá de lo que dice el texto.
2. Cita siempre la fuente normativa exacta: documento, parte, título, capítulo y \
numeral o sección correspondiente.
3. Si la información no es suficiente para responder con certeza, responde \
exactamente: "No tengo información suficiente en la normativa consultada para \
responder esta pregunta."
4. Responde en español colombiano formal con registro jurídico-financiero.
5. No inventes definiciones, plazos, porcentajes ni obligaciones que no estén \
explícitamente en el contexto.
6. Si varios fragmentos aportan información complementaria, intégralos en una \
respuesta cohesiva citando cada fuente.
"""

# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """Resultado completo del pipeline RAG."""
    query: str
    answer: str
    chunks_used: list[RetrievedChunk]
    model: str
    elapsed_seconds: float
    k: int

    def __str__(self) -> str:
        lines = [
            f"Pregunta: {self.query}",
            "=" * 70,
            "",
            self.answer,
            "",
            "-" * 70,
            f"Fuentes recuperadas ({len(self.chunks_used)} chunks, k={self.k}, "
            f"modelo={self.model}, {self.elapsed_seconds:.1f}s):",
        ]
        for chunk in self.chunks_used:
            lines.append(
                f"  [{chunk.rank}] score={chunk.score:.3f} | {chunk.section_id} | "
                f"{chunk.hierarchy_path[:60]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline RAG
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orquesta retrieval + generación.
    Se instancia una sola vez y se reutiliza.
    """

    def __init__(self, model: str = OLLAMA_MODEL, k: int = DEFAULT_K):
        self.model = model
        self.k = k
        self._retriever = Retriever()
        self._client = ollama.Client(host=OLLAMA_HOST)
        self._verify_model()

    def _verify_model(self):
        """Confirma que el modelo está disponible en Ollama."""
        try:
            models = [m.model for m in self._client.list().models]
            # ollama incluye tags, e.g. "llama3.1:8b"
            available = [m.split(":")[0] for m in models] + models
            target = self.model.split(":")[0]
            if self.model not in available and target not in available:
                print(
                    f"[WARN] Modelo '{self.model}' no encontrado en Ollama. "
                    f"Disponibles: {models}"
                )
        except Exception as e:
            print(f"[WARN] No se pudo verificar modelos Ollama: {e}")

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        k: Optional[int] = None,
        chunk_type: Optional[str] = None,
        section_prefix: Optional[str] = None,
    ) -> RAGResponse:
        """
        Ejecuta el pipeline RAG completo y devuelve la respuesta.

        Args:
            user_query:     Pregunta en lenguaje natural.
            k:              Número de chunks a recuperar (default: self.k).
            chunk_type:     Filtro opcional por tipo de chunk.
            section_prefix: Filtro opcional por prefijo de sección.

        Returns:
            RAGResponse con la respuesta del LLM y metadata del retrieval.
        """
        k = k or self.k
        t0 = time.perf_counter()

        retrieval = self._retriever.retrieve(
            user_query,
            k=k,
            chunk_type=chunk_type,
            section_prefix=section_prefix,
        )

        if not retrieval.chunks:
            return RAGResponse(
                query=user_query,
                answer="No tengo información suficiente en la normativa consultada "
                       "para responder esta pregunta.",
                chunks_used=[],
                model=self.model,
                elapsed_seconds=time.perf_counter() - t0,
                k=k,
            )

        prompt = self._build_user_prompt(user_query, retrieval)

        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.1, "num_predict": 1024},
        )

        answer = response.message.content.strip()
        elapsed = time.perf_counter() - t0

        return RAGResponse(
            query=user_query,
            answer=answer,
            chunks_used=retrieval.chunks,
            model=self.model,
            elapsed_seconds=elapsed,
            k=k,
        )

    def stream(
        self,
        user_query: str,
        k: Optional[int] = None,
    ) -> tuple[Iterator[str], RetrievalResult]:
        """
        Versión streaming: devuelve un iterador de tokens y el retrieval.
        Útil para la interfaz Streamlit de la Fase 6.

        Returns:
            (token_iterator, retrieval_result)
        """
        k = k or self.k

        retrieval = self._retriever.retrieve(user_query, k=k)

        if not retrieval.chunks:
            def _empty():
                yield "No tengo información suficiente en la normativa consultada " \
                      "para responder esta pregunta."
            return _empty(), retrieval

        prompt = self._build_user_prompt(user_query, retrieval)

        stream_response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.1, "num_predict": 1024},
            stream=True,
        )

        def _token_gen():
            for chunk in stream_response:
                yield chunk.message.content

        return _token_gen(), retrieval

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(query: str, retrieval: RetrievalResult) -> str:
        """
        Construye el prompt de usuario con el contexto de los chunks recuperados.
        Usa content_with_prefix (que incluye [Fuente:][Sección:]) para dar
        contexto completo al LLM.
        """
        context_blocks = []
        for chunk in retrieval.chunks:
            context_blocks.append(
                f"--- Fragmento {chunk.rank} (relevancia: {chunk.score:.2f}) ---\n"
                f"{chunk.content_with_prefix}"
            )

        context = "\n\n".join(context_blocks)

        return (
            f"CONTEXTO NORMATIVO:\n\n{context}\n\n"
            f"{'='*60}\n\n"
            f"PREGUNTA: {query}\n\n"
            f"Responde con base EXCLUSIVAMENTE en los fragmentos anteriores. "
            f"Incluye la referencia normativa exacta en tu respuesta."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "¿Qué es una operación sospechosa y cuándo debe reportarse?",
    "¿Cuáles son las funciones del oficial de cumplimiento?",
    "¿Qué es el beneficiario final según el SARLAFT?",
    "¿Qué reportes deben enviarse a la UIAF?",
    "¿Cuáles son las etapas del SARLAFT?",
]


def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG SARLAFT — Fase 4")
    parser.add_argument("--query", "-q", type=str, help="Pregunta a responder")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Chunks a recuperar (default: {DEFAULT_K})")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help=f"Modelo Ollama (default: {OLLAMA_MODEL})")
    parser.add_argument("--stream", action="store_true", help="Streaming de tokens")
    parser.add_argument("--batch", action="store_true", help="Correr queries de demo")
    parser.add_argument("--filter-type", choices=["definition", "section", "intro"])
    parser.add_argument("--section-prefix", type=str)
    args = parser.parse_args()

    if not args.query and not args.batch:
        parser.print_help()
        sys.exit(0)

    pipeline = RAGPipeline(model=args.model, k=args.k)

    if args.batch:
        print(f"\n{'='*70}")
        print(f"  DEMO BATCH — {len(DEMO_QUERIES)} preguntas")
        print(f"  Modelo: {args.model}  |  k={args.k}")
        print(f"{'='*70}\n")
        for i, q in enumerate(DEMO_QUERIES, 1):
            print(f"\n[{i}/{len(DEMO_QUERIES)}] {q}")
            print("-" * 70)
            response = pipeline.query(q, k=args.k)
            print(response.answer)
            print(f"\n  [Fuentes: {len(response.chunks_used)} chunks | {response.elapsed_seconds:.1f}s]")
            for c in response.chunks_used:
                print(f"    · score={c.score:.3f} | {c.section_id} | {c.hierarchy_path[:55]}")
            print()
        return

    if args.stream:
        print(f"\nPregunta: {args.query}\n{'='*70}\n")
        token_iter, retrieval = pipeline.stream(args.query, k=args.k)
        for token in token_iter:
            print(token, end="", flush=True)
        print(f"\n\n{'─'*70}")
        print(f"Fuentes ({len(retrieval.chunks)} chunks):")
        for c in retrieval.chunks:
            print(f"  · score={c.score:.3f} | {c.section_id} | {c.hierarchy_path[:55]}")
    else:
        response = pipeline.query(
            args.query,
            k=args.k,
            chunk_type=args.filter_type,
            section_prefix=args.section_prefix,
        )
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
