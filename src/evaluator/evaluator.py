"""
Fase 5 — Evaluación del sistema RAG SARLAFT.

Métricas técnicas (retrieval):
  - Precision@k (k=3, k=5): fracción de chunks recuperados que son relevantes.
  - Recall@k    (k=3, k=5): fracción de secciones esperadas cubiertas en top-k.
  - MRR         (k=5):      Mean Reciprocal Rank del primer chunk relevante.
  - Context relevance:      score promedio de similitud coseno de los top-k chunks.

Métricas de negocio (generación, requiere --full):
  - Tasa de cita normativa presente: ¿el LLM cita alguna sección en su respuesta?
  - Tasa de cita correcta: ¿cita una de las secciones esperadas?
  - Faithfulness (groundedness): ¿el LLM solo referencia secciones que estaban en el contexto?
  - Tasa "no info": para preguntas fuera de corpus, ¿declara correctamente que no sabe?
  - Tiempo promedio de respuesta.

Uso:
    python src/evaluator/evaluator.py                   # solo retrieval (rápido)
    python src/evaluator/evaluator.py --full            # retrieval + generación (lento)
    python src/evaluator/evaluator.py --report          # muestra último reporte guardado
    python src/evaluator/evaluator.py --k 3 5 7         # evaluar con k=[3,5,7]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever.retriever import Retriever

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
RESULTS_DIR    = PROJECT_ROOT / "eval" / "results"
DEFAULT_K      = [3, 5]

NO_INFO_PHRASES = [
    "no tengo información suficiente",
    "no encuentro información",
    "no hay información",
    "no se menciona",
    "no está contemplado",
]

# ---------------------------------------------------------------------------
# Helpers de relevancia
# ---------------------------------------------------------------------------

def normalize_sid(sid: str) -> str:
    """Elimina trailing dot y sufijos de split como '[p1]'."""
    return re.sub(r'\[p\d+\]$', '', sid).rstrip('.')


def is_relevant(chunk_section_id: str, expected_sections: list[str]) -> bool:
    """
    Un chunk es relevante si su sección es igual, un hijo o un padre
    de alguna sección esperada.

    Ejemplos:
        chunk "1.17."   vs expected ["1.17"]    → True  (exacto)
        chunk "4.1.1."  vs expected ["4.1"]     → True  (hijo)
        chunk "4.1."    vs expected ["4.1.1"]   → True  (padre, crédito parcial)
    """
    c = normalize_sid(chunk_section_id)
    for exp in expected_sections:
        e = exp.rstrip('.')
        if c == e or c.startswith(e + '.') or e.startswith(c + '.'):
            return True
    return False


def first_relevant_rank(section_ids: list[str], expected: list[str]) -> Optional[int]:
    """Devuelve el rank (1-based) del primer chunk relevante, o None."""
    for i, sid in enumerate(section_ids, 1):
        if is_relevant(sid, expected):
            return i
    return None


# ---------------------------------------------------------------------------
# Métricas de generación
# ---------------------------------------------------------------------------

SECTION_RE = re.compile(r'\b([1-9]\d*(?:\.\d+)+\.?)\b')

def extract_cited_sections(text: str) -> list[str]:
    """Extrae todas las referencias de sección del texto (ej. '4.2.7.2.1')."""
    return [m.group(1).rstrip('.') for m in SECTION_RE.finditer(text)]


def has_citation(answer: str) -> bool:
    return bool(SECTION_RE.search(answer))


def correct_citation(answer: str, expected_sections: list[str]) -> bool:
    """¿El LLM citó al menos una de las secciones esperadas (o un hijo/padre)?"""
    cited = extract_cited_sections(answer)
    return any(is_relevant(c + '.', expected_sections) for c in cited)


def faithfulness(answer: str, retrieved_section_ids: list[str]) -> tuple[bool, list[str]]:
    """
    ¿El LLM solo citó secciones que estaban en el contexto recuperado?
    Devuelve (is_faithful, hallucinated_sections).
    Una sección citada se considera hallucinated si no es hijo ni padre
    de ninguna sección recuperada.
    """
    cited = extract_cited_sections(answer)
    hallucinated = []
    for c in cited:
        if not is_relevant(c + '.', [normalize_sid(s) for s in retrieved_section_ids]):
            hallucinated.append(c)
    return len(hallucinated) == 0, hallucinated


def declared_no_info(answer: str) -> bool:
    lower = answer.lower()
    return any(p in lower for p in NO_INFO_PHRASES)


# ---------------------------------------------------------------------------
# Evaluador
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, k_values: list[int] = None):
        self.k_values   = k_values or DEFAULT_K
        self.k_max      = max(self.k_values)
        self._retriever = Retriever()

    # ------------------------------------------------------------------
    # Evaluación de retrieval
    # ------------------------------------------------------------------

    def evaluate_retrieval(self, questions: list[dict]) -> list[dict]:
        """
        Para cada pregunta calcula métricas de retrieval.
        Solo corre el retriever (rápido).
        """
        results = []
        in_corpus = [q for q in questions if q.get("is_in_corpus", True)]

        print(f"\n[Retrieval] Evaluando {len(in_corpus)} preguntas con k_max={self.k_max}...")

        for q in in_corpus:
            retrieval = self._retriever.retrieve(q["question"], k=self.k_max)
            # Use only ANN-retrieved (base) chunks for metrics; expanded children
            # would distort the top-k window and make P@k / R@k incomparable.
            base = retrieval.base_chunks
            section_ids = [c.section_id for c in base]
            scores      = [c.score for c in base]
            expected    = q.get("expected_sections", [])

            row: dict = {
                "id":       q["id"],
                "question": q["question"],
                "topic":    q.get("topic", ""),
                "difficulty": q.get("difficulty", ""),
                "expected_sections": expected,
                "retrieved_sections": section_ids,
                "retrieved_scores":   scores,
            }

            # MRR
            frr = first_relevant_rank(section_ids, expected)
            row["mrr"]          = (1.0 / frr) if frr else 0.0
            row["first_hit_rank"] = frr

            for k in self.k_values:
                top_ids    = section_ids[:k]
                top_scores = scores[:k]
                relevant_flags = [is_relevant(s, expected) for s in top_ids]

                p_at_k = sum(relevant_flags) / k if k else 0.0
                r_at_k = (
                    len(set(
                        e for e in expected
                        if any(is_relevant(s, [e]) for s in top_ids)
                    )) / len(expected)
                    if expected else 0.0
                )

                row[f"precision_at_{k}"] = round(p_at_k, 4)
                row[f"recall_at_{k}"]    = round(r_at_k, 4)
                row[f"context_relevance_at_{k}"] = round(
                    sum(top_scores) / len(top_scores), 4
                ) if top_scores else 0.0

            results.append(row)
            _print_retrieval_row(row, self.k_values)

        return results

    # ------------------------------------------------------------------
    # Evaluación de generación
    # ------------------------------------------------------------------

    def evaluate_generation(self, questions: list[dict]) -> list[dict]:
        """
        Para cada pregunta corre el pipeline completo (retrieval + LLM).
        Lento: ~10-60s por pregunta según el modelo.
        """
        from src.generator.generator import RAGPipeline
        pipeline = RAGPipeline(k=self.k_max)

        results = []
        print(f"\n[Generation] Evaluando {len(questions)} preguntas "
              f"(modelo={pipeline.model}, k={pipeline.k})...")

        for q in questions:
            expected = q.get("expected_sections", [])
            in_corpus = q.get("is_in_corpus", True)

            t0 = time.perf_counter()
            rag_resp = pipeline.query(q["question"])
            elapsed = time.perf_counter() - t0

            answer = rag_resp.answer
            ret_sids = [c.section_id for c in rag_resp.chunks_used]

            faithful, hallucinated = faithfulness(answer, ret_sids)

            row = {
                "id":               q["id"],
                "question":         q["question"],
                "topic":            q.get("topic", ""),
                "is_in_corpus":     in_corpus,
                "answer":           answer,
                "elapsed_seconds":  round(elapsed, 2),
                "has_citation":     has_citation(answer),
                "correct_citation": correct_citation(answer, expected) if expected else None,
                "faithful":         faithful,
                "hallucinated_sections": hallucinated,
                "declared_no_info": declared_no_info(answer),
                "no_info_correct":  (declared_no_info(answer) == (not in_corpus)),
                "retrieved_sections": ret_sids,
            }
            results.append(row)
            _print_generation_row(row)

        return results

    # ------------------------------------------------------------------
    # Agregación y reporte
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_retrieval(results: list[dict], k_values: list[int]) -> dict:
        n = len(results)
        if n == 0:
            return {}

        metrics: dict = {"n_questions": n}

        for k in k_values:
            metrics[f"precision_at_{k}"] = round(
                sum(r.get(f"precision_at_{k}", 0) for r in results) / n, 4
            )
            metrics[f"recall_at_{k}"] = round(
                sum(r.get(f"recall_at_{k}", 0) for r in results) / n, 4
            )
            metrics[f"context_relevance_at_{k}"] = round(
                sum(r.get(f"context_relevance_at_{k}", 0) for r in results) / n, 4
            )

        metrics["mrr"] = round(sum(r["mrr"] for r in results) / n, 4)
        metrics["hit_rate"] = round(
            sum(1 for r in results if r["mrr"] > 0) / n, 4
        )

        # Por dificultad
        for diff in ["easy", "medium", "hard"]:
            sub = [r for r in results if r.get("difficulty") == diff]
            if sub:
                metrics[f"mrr_{diff}"] = round(sum(r["mrr"] for r in sub) / len(sub), 4)

        return metrics

    @staticmethod
    def aggregate_generation(results: list[dict]) -> dict:
        n = len(results)
        if n == 0:
            return {}

        in_corpus  = [r for r in results if r["is_in_corpus"]]
        out_corpus = [r for r in results if not r["is_in_corpus"]]

        metrics: dict = {
            "n_questions":         n,
            "n_in_corpus":         len(in_corpus),
            "n_out_of_corpus":     len(out_corpus),
            "avg_response_time_s": round(
                sum(r["elapsed_seconds"] for r in results) / n, 2
            ),
        }

        if in_corpus:
            metrics["citation_rate"] = round(
                sum(1 for r in in_corpus if r["has_citation"]) / len(in_corpus), 4
            )
            with_expected = [r for r in in_corpus if r["correct_citation"] is not None]
            metrics["correct_citation_rate"] = round(
                sum(1 for r in with_expected if r["correct_citation"]) / len(with_expected), 4
            ) if with_expected else None
            metrics["faithfulness_rate"] = round(
                sum(1 for r in in_corpus if r["faithful"]) / len(in_corpus), 4
            )

        if out_corpus:
            metrics["no_info_detection_rate"] = round(
                sum(1 for r in out_corpus if r["declared_no_info"]) / len(out_corpus), 4
            )

        return metrics

    @staticmethod
    def print_retrieval_report(metrics: dict, k_values: list[int]):
        print(f"\n{'='*60}")
        print("  REPORTE DE EVALUACIÓN — RETRIEVAL")
        print(f"{'='*60}")
        print(f"  Preguntas evaluadas : {metrics.get('n_questions', '?')}")
        print(f"  Hit Rate@{max(k_values)}          : {metrics.get('hit_rate', 0):.1%}")
        print(f"  MRR                 : {metrics.get('mrr', 0):.4f}")
        for k in k_values:
            print(f"\n  --- k={k} ---")
            print(f"  Precision@{k}         : {metrics.get(f'precision_at_{k}', 0):.4f}")
            print(f"  Recall@{k}            : {metrics.get(f'recall_at_{k}', 0):.4f}")
            print(f"  Context Relevance@{k} : {metrics.get(f'context_relevance_at_{k}', 0):.4f}")
        print(f"\n  --- MRR por dificultad ---")
        for diff in ["easy", "medium", "hard"]:
            key = f"mrr_{diff}"
            if key in metrics:
                print(f"  MRR ({diff:6s})       : {metrics[key]:.4f}")
        print(f"{'='*60}\n")

    @staticmethod
    def print_generation_report(metrics: dict):
        print(f"\n{'='*60}")
        print("  REPORTE DE EVALUACIÓN — GENERACIÓN")
        print(f"{'='*60}")
        print(f"  Preguntas evaluadas : {metrics.get('n_questions', '?')}")
        print(f"  Tiempo prom. resp.  : {metrics.get('avg_response_time_s', 0):.1f}s")
        print(f"\n  --- Métricas de negocio (in-corpus) ---")
        print(f"  Cita normativa      : {metrics.get('citation_rate', 0):.1%}")
        print(f"  Cita correcta       : {(metrics.get('correct_citation_rate') or 0):.1%}")
        print(f"  Faithfulness        : {metrics.get('faithfulness_rate', 0):.1%}")
        if metrics.get("no_info_detection_rate") is not None:
            print(f"\n  --- Métricas out-of-corpus ---")
            print(f"  Detección 'no info' : {metrics.get('no_info_detection_rate', 0):.1%}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Helpers de impresión
# ---------------------------------------------------------------------------

def _print_retrieval_row(row: dict, k_values: list[int]):
    k_main = max(k_values)
    hit = "✓" if row["mrr"] > 0 else "✗"
    p = row.get(f"precision_at_{k_main}", 0)
    r = row.get(f"recall_at_{k_main}", 0)
    print(
        f"  {hit} {row['id']} | {row['difficulty']:6s} | "
        f"P@{k_main}={p:.2f} R@{k_main}={r:.2f} MRR={row['mrr']:.2f} | "
        f"{row['question'][:55]}"
    )


def _print_generation_row(row: dict):
    citation = "C✓" if row["has_citation"] else "C✗"
    faithful = "F✓" if row["faithful"] else f"F✗({','.join(row['hallucinated_sections'][:2])})"
    no_info  = "NI✓" if row.get("no_info_correct") else "NI✗"
    print(
        f"  {row['id']} | {citation} {faithful} {no_info} | "
        f"{row['elapsed_seconds']:.1f}s | {row['question'][:50]}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluador RAG SARLAFT — Fase 5")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Incluir evaluación de generación (lento: ~30-90min para 30 preguntas)",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=DEFAULT_K,
        help=f"Valores de k a evaluar (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Mostrar el último reporte guardado en eval/results/",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(QUESTIONS_PATH),
        help=f"Ruta al archivo de preguntas (default: {QUESTIONS_PATH})",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Mostrar último reporte
    if args.report:
        reports = sorted(RESULTS_DIR.glob("summary_*.json"), reverse=True)
        if not reports:
            print("[ERROR] No hay reportes guardados en eval/results/")
            sys.exit(1)
        with open(reports[0]) as f:
            summary = json.load(f)
        print(f"\nÚltimo reporte: {reports[0].name}\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    # Cargar preguntas
    questions_path = Path(args.questions)
    if not questions_path.exists():
        sys.exit(f"[ERROR] No se encontró {questions_path}")
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[OK] Cargadas {len(questions)} preguntas desde {questions_path.name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluator = Evaluator(k_values=args.k)

    # --- Retrieval ---
    retrieval_results = evaluator.evaluate_retrieval(questions)
    retrieval_metrics = Evaluator.aggregate_retrieval(retrieval_results, args.k)
    evaluator.print_retrieval_report(retrieval_metrics, args.k)

    # Guardar resultados de retrieval
    with open(RESULTS_DIR / f"retrieval_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=2, ensure_ascii=False)

    summary = {
        "timestamp": timestamp,
        "n_questions": len(questions),
        "k_values": args.k,
        "retrieval": retrieval_metrics,
    }

    # --- Generación (opcional) ---
    if args.full:
        gen_results = evaluator.evaluate_generation(questions)
        gen_metrics = Evaluator.aggregate_generation(gen_results)
        evaluator.print_generation_report(gen_metrics)

        with open(RESULTS_DIR / f"generation_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(gen_results, f, indent=2, ensure_ascii=False)

        summary["generation"] = gen_metrics

    # Guardar summary
    with open(RESULTS_DIR / f"summary_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Resultados guardados en eval/results/ (ts={timestamp})")


if __name__ == "__main__":
    main()
