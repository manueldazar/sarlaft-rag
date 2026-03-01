"""
Fase 3.2 — Retrieval desde ChromaDB.

Recibe una query en lenguaje natural, genera su embedding y devuelve los
k chunks más similares con su metadata y score de relevancia.

Uso programático:
    from src.retriever.retriever import Retriever
    r = Retriever()
    results = r.retrieve("¿Qué es una operación sospechosa?", k=5)

Uso desde terminal (validación manual):
    python src/retriever/retriever.py --query "¿Qué es lavado de activos?" --k 5
    python src/retriever/retriever.py --query "oficial de cumplimiento" --filter-type definition
    python src/retriever/retriever.py --batch                      # corre las 20 queries de prueba
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "data/chroma_db"))
if not CHROMA_DB_PATH.is_absolute():
    CHROMA_DB_PATH = PROJECT_ROOT / CHROMA_DB_PATH

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sarlaft_chunks")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """Un chunk recuperado con su score de similitud y metadata."""
    rank: int
    chunk_id: str
    score: float                  # similitud coseno [0, 1], mayor = más relevante
    raw_content: str              # texto sin prefijo (usado para embeddings)
    content_with_prefix: str      # texto con [Fuente:][Sección:] para el LLM
    source: str
    section_id: str
    section_title: str
    hierarchy_path: str
    depth: int
    chunk_type: str
    last_updated: str
    char_count: int
    parent_section: str
    # Expansión parent-child
    expanded: bool = False        # True si fue agregado por expansión de un chunk padre
    expanded_from: str = ""       # section_id del padre que originó esta expansión

    def __str__(self) -> str:
        if self.expanded:
            prefix = f"  ↳ [exp §{self.expanded_from}]"
        else:
            prefix = f"[{self.rank}]"
        return (
            f"{prefix} score={self.score:.3f} | {self.section_id} "
            f"| tipo={self.chunk_type} | depth={self.depth}\n"
            f"    ruta: {self.hierarchy_path[:70]}\n"
            f"    texto: {self.raw_content[:120].strip()}..."
        )


@dataclass
class RetrievalResult:
    """Resultado completo de una búsqueda."""
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    filters_applied: dict = field(default_factory=dict)

    @property
    def top(self) -> Optional[RetrievedChunk]:
        return self.chunks[0] if self.chunks else None

    @property
    def base_chunks(self) -> list[RetrievedChunk]:
        """Solo los chunks del top-k original, sin los expandidos."""
        return [c for c in self.chunks if not c.expanded]

    def __str__(self) -> str:
        header = f"Query: «{self.query}»"
        if self.filters_applied:
            header += f"  [filtros: {self.filters_applied}]"
        lines = [header, "=" * len(header)]
        for chunk in self.chunks:
            lines.append(str(chunk))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Encapsula el modelo de embeddings y la conexión a ChromaDB.
    Se instancia una sola vez y se reutiliza para múltiples queries.
    """

    def __init__(self):
        if not CHROMA_DB_PATH.exists():
            sys.exit(
                f"[ERROR] No se encontró la base de datos ChromaDB en: {CHROMA_DB_PATH}\n"
                "Ejecuta primero: python src/indexer/index_chunks.py"
            )

        self._model = SentenceTransformer(EMBEDDING_MODEL)

        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self._collection = client.get_collection(COLLECTION_NAME)

        count = self._collection.count()
        if count == 0:
            sys.exit("[ERROR] La colección ChromaDB está vacía. Re-ejecuta el indexador.")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        chunk_type: Optional[str] = None,
        min_depth: Optional[int] = None,
        max_depth: Optional[int] = None,
        section_prefix: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Busca los k chunks más relevantes para la query.

        Args:
            query:          Pregunta en lenguaje natural.
            k:              Número de resultados a devolver.
            chunk_type:     Filtrar por tipo (e.g., "definition", "section", "intro").
            min_depth:      Profundidad mínima en la jerarquía (0 = raíz).
            max_depth:      Profundidad máxima en la jerarquía.
            section_prefix: Filtrar secciones que empiecen con este prefijo (e.g., "4.2").

        Returns:
            RetrievalResult con los chunks ordenados por relevancia descendente.
        """
        query = query.strip()
        if not query:
            raise ValueError("La query no puede estar vacía.")

        # Construir filtros where de ChromaDB
        where_clause = _build_where(chunk_type, min_depth, max_depth, section_prefix)

        # Embedding de la query
        query_embedding = self._model.encode(query).tolist()

        # Recuperar más resultados de los necesarios si hay filtros.
        # section_prefix se filtra en Python (ChromaDB no soporta startswith),
        # y where_clause reduce el espacio desde ChromaDB — ambos requieren over-fetch.
        has_any_filter = bool(where_clause) or bool(section_prefix)
        n_results = k if not has_any_filter else min(k * 10, self._collection.count())

        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where_clause:
            kwargs["where"] = where_clause

        raw = self._collection.query(**kwargs)

        chunks = []
        for i, (doc, meta, dist) in enumerate(
            zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0])
        ):
            # Aplicar filtro de section_prefix aquí (ChromaDB no soporta startswith)
            if section_prefix and not meta.get("section_id", "").startswith(section_prefix):
                continue

            chunks.append(
                RetrievedChunk(
                    rank=len(chunks) + 1,
                    chunk_id=raw["ids"][0][i],
                    score=1.0 - dist,           # distancia coseno → similitud
                    raw_content=doc,
                    content_with_prefix=meta.get("content_with_prefix", doc),
                    source=meta.get("source", ""),
                    section_id=meta.get("section_id", ""),
                    section_title=meta.get("section_title", ""),
                    hierarchy_path=meta.get("hierarchy_path", ""),
                    depth=int(meta.get("depth", 0)),
                    chunk_type=meta.get("chunk_type", ""),
                    last_updated=meta.get("last_updated", ""),
                    char_count=int(meta.get("char_count", 0)),
                    parent_section=meta.get("parent_section", ""),
                )
            )
            if len(chunks) == k:
                break

        # ---------------------------------------------------------------
        # Expansión parent-child
        # Para cada chunk que sea un encabezado de sección (char_count bajo
        # o contenido termina en ':'), buscar hijos directos en ChromaDB y
        # agregarlos al resultado sin contar contra el límite k.
        # ---------------------------------------------------------------
        expanded: list[RetrievedChunk] = []
        seen_ids: set[str] = {c.chunk_id for c in chunks}

        for chunk in chunks:
            expanded.append(chunk)
            if _is_header(chunk):
                for child in self._fetch_children(chunk):
                    if child.chunk_id not in seen_ids:
                        expanded.append(child)
                        seen_ids.add(child.chunk_id)

        # Re-numerar ranks secuencialmente en la lista final
        for i, c in enumerate(expanded, 1):
            c.rank = i

        filters_applied = {}
        if chunk_type:
            filters_applied["chunk_type"] = chunk_type
        if min_depth is not None:
            filters_applied["min_depth"] = min_depth
        if max_depth is not None:
            filters_applied["max_depth"] = max_depth
        if section_prefix:
            filters_applied["section_prefix"] = section_prefix

        return RetrievalResult(query=query, chunks=expanded, filters_applied=filters_applied)

    def _fetch_children(self, parent: RetrievedChunk) -> list[RetrievedChunk]:
        """
        Una sola query a ChromaDB por chunk expandido:
        trae todos los chunks en depth = parent.depth + 1 y filtra en Python
        los que sean hijos directos (section_id empieza con parent_sid + ".").
        """
        parent_sid  = _normalize_sid(parent.section_id)
        target_depth = parent.depth + 1

        raw = self._collection.get(
            where={"depth": {"$eq": target_depth}},
            include=["documents", "metadatas"],
        )

        children: list[RetrievedChunk] = []
        for doc, meta, cid in zip(raw["documents"], raw["metadatas"], raw["ids"]):
            sid       = meta.get("section_id", "")
            sid_clean = _normalize_sid(sid)
            if sid_clean.startswith(parent_sid + "."):
                children.append(RetrievedChunk(
                    rank=0,                          # se asigna al final
                    chunk_id=cid,
                    score=parent.score,              # hereda el score del padre
                    raw_content=doc,
                    content_with_prefix=meta.get("content_with_prefix", doc),
                    source=meta.get("source", ""),
                    section_id=sid,
                    section_title=meta.get("section_title", ""),
                    hierarchy_path=meta.get("hierarchy_path", ""),
                    depth=int(meta.get("depth", 0)),
                    chunk_type=meta.get("chunk_type", ""),
                    last_updated=meta.get("last_updated", ""),
                    char_count=int(meta.get("char_count", 0)),
                    parent_section=meta.get("parent_section", ""),
                    expanded=True,
                    expanded_from=parent.section_id,
                ))

        children.sort(key=lambda c: c.section_id)
        return children


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _normalize_sid(sid: str) -> str:
    """Elimina trailing dot y sufijos de split como '[p1]'."""
    return re.sub(r'\[p\d+\]$', '', sid).rstrip('.')


def _is_header(chunk: RetrievedChunk) -> bool:
    """
    Un chunk es candidato a expansión cuando es un encabezado de sección con hijos:
    - char_count bajo (< 200): el chunk es solo un título o intro de lista, O
    - el contenido termina en ':' o 'deben:': patrón típico de lista enumerable en la norma.
    """
    content = chunk.raw_content.strip()
    return (
        chunk.char_count < 200
        or content.endswith(":")
        or content.endswith("deben:")
    )


def _build_where(
    chunk_type: Optional[str],
    min_depth: Optional[int],
    max_depth: Optional[int],
    section_prefix: Optional[str],
) -> Optional[dict]:
    """Construye el filtro `where` de ChromaDB."""
    conditions = []

    if chunk_type:
        conditions.append({"chunk_type": {"$eq": chunk_type}})

    if min_depth is not None and max_depth is not None:
        conditions.append({"depth": {"$gte": min_depth}})
        conditions.append({"depth": {"$lte": max_depth}})
    elif min_depth is not None:
        conditions.append({"depth": {"$gte": min_depth}})
    elif max_depth is not None:
        conditions.append({"depth": {"$lte": max_depth}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BATCH_QUERIES = [
    "¿Qué es lavado de activos según el SARLAFT?",
    "¿Cuáles son las etapas del SARLAFT?",
    "¿Qué es un beneficiario final?",
    "¿Cuáles son los factores de riesgo del SARLAFT?",
    "¿Qué funciones tiene el oficial de cumplimiento?",
    "¿Qué son las personas expuestas políticamente?",
    "¿Cuándo se aplican procedimientos simplificados de conocimiento del cliente?",
    "¿Qué es una operación sospechosa?",
    "¿Qué reportes deben enviarse a la UIAF?",
    "¿Qué son las sanciones financieras dirigidas?",
    "¿Qué es la financiación del terrorismo?",
    "¿Qué obligaciones tienen las entidades vigiladas frente a la SFC?",
    "¿Qué es el segmentación de clientes en el SARLAFT?",
    "¿Cuáles son los elementos del SARLAFT?",
    "¿Qué es el conocimiento del cliente?",
    "¿Qué son las señales de alerta?",
    "¿Qué es una operación inusual?",
    "¿Qué responsabilidades tiene la junta directiva en el SARLAFT?",
    "¿Cuándo se exonera a un cliente de reportes a la UIAF?",
    "¿Qué es el enfoque basado en riesgo en LA/FT?",
]


def main():
    parser = argparse.ArgumentParser(description="Retriever SARLAFT — Fase 3.2")
    parser.add_argument("--query", "-q", type=str, help="Query a buscar")
    parser.add_argument("--k", type=int, default=5, help="Número de resultados (default: 5)")
    parser.add_argument(
        "--filter-type",
        choices=["definition", "section", "intro"],
        help="Filtrar por tipo de chunk",
    )
    parser.add_argument("--min-depth", type=int, help="Profundidad mínima")
    parser.add_argument("--max-depth", type=int, help="Profundidad máxima")
    parser.add_argument("--section-prefix", type=str, help='Prefijo de sección (e.g., "4.2")')
    parser.add_argument(
        "--batch",
        action="store_true",
        help=f"Corre las {len(BATCH_QUERIES)} queries de validación",
    )
    args = parser.parse_args()

    if not args.query and not args.batch:
        parser.print_help()
        sys.exit(0)

    retriever = Retriever()

    if args.batch:
        print(f"\n{'='*70}")
        print(f"  VALIDACIÓN BATCH — {len(BATCH_QUERIES)} queries de prueba")
        print(f"  k={args.k} resultados por query")
        print(f"{'='*70}\n")
        for i, q in enumerate(BATCH_QUERIES, 1):
            result = retriever.retrieve(q, k=args.k)
            print(f"\n--- [{i:02d}/{len(BATCH_QUERIES)}] {result.query} ---")
            for chunk in result.chunks:
                print(f"  {chunk}")
            print()
    else:
        result = retriever.retrieve(
            args.query,
            k=args.k,
            chunk_type=args.filter_type,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            section_prefix=args.section_prefix,
        )
        print(f"\n{result}\n")


if __name__ == "__main__":
    main()
