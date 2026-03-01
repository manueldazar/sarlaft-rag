"""
Fase 3.1 — Indexación de chunks en ChromaDB.

Lee chunks.json, genera embeddings con paraphrase-multilingual-MiniLM-L12-v2
y los persiste en ChromaDB.

Uso:
    python src/indexer/index_chunks.py
    python src/indexer/index_chunks.py --reset   # borra colección existente antes de indexar
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", "data/chunks/chunks.json"))
if not CHUNKS_PATH.is_absolute():
    CHUNKS_PATH = PROJECT_ROOT / CHUNKS_PATH

CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "data/chroma_db"))
if not CHROMA_DB_PATH.is_absolute():
    CHROMA_DB_PATH = PROJECT_ROOT / CHROMA_DB_PATH

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sarlaft_chunks")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Tamaño de lote para generación de embeddings — ajustar según VRAM/RAM
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_chunks(path: Path) -> list[dict]:
    """Carga y valida chunks.json."""
    if not path.exists():
        sys.exit(f"[ERROR] No se encontró chunks.json en: {path}")

    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or not chunks:
        sys.exit("[ERROR] chunks.json está vacío o tiene formato incorrecto.")

    # Verificar campos requeridos en el primer chunk
    required = {"chunk_id", "raw_content", "source", "section_id",
                 "hierarchy_path", "depth", "chunk_type"}
    missing = required - set(chunks[0].keys())
    if missing:
        sys.exit(f"[ERROR] Campos faltantes en chunks.json: {missing}")

    print(f"[OK] Cargados {len(chunks)} chunks desde {path}")
    return chunks


def build_metadata(chunk: dict) -> dict:
    """
    Extrae metadata para ChromaDB.
    ChromaDB solo acepta str, int, float o bool — no None ni listas.
    """
    return {
        "source":            str(chunk.get("source", "")),
        "section_id":        str(chunk.get("section_id", "")),
        "section_title":     str(chunk.get("section_title", "")),
        "hierarchy_path":    str(chunk.get("hierarchy_path", "")),
        "hierarchy_path_ids": str(chunk.get("hierarchy_path_ids", "")),
        "depth":             int(chunk.get("depth", 0)),
        "char_count":        int(chunk.get("char_count", 0)),
        "last_updated":      str(chunk.get("last_updated", "")),
        "parent_section":    str(chunk.get("parent_section", "")),
        "chunk_type":        str(chunk.get("chunk_type", "")),
        # Guardar el content con prefijo para usarlo en la Fase 4 (generación)
        "content_with_prefix": str(chunk.get("content", "")),
    }


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """Genera embeddings en lotes con barra de progreso."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generando embeddings"):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(embs.tolist())
    return all_embeddings


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Indexa chunks en ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Elimina la colección existente antes de indexar",
    )
    args = parser.parse_args()

    # 1. Cargar chunks
    chunks = load_chunks(CHUNKS_PATH)

    # 2. Cargar modelo de embeddings
    print(f"[INFO] Cargando modelo: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("[OK] Modelo cargado")

    # 3. Conectar a ChromaDB (persistente)
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[INFO] Colección '{COLLECTION_NAME}' eliminada")
        except Exception:
            pass  # No existía

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # distancia coseno para similitud semántica
    )

    # Evitar duplicados: solo indexar chunks que no existen aún
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print(f"[OK] Todos los chunks ya están indexados ({len(existing_ids)} en colección).")
        _print_stats(collection)
        return

    print(f"[INFO] Chunks nuevos a indexar: {len(new_chunks)} "
          f"(ya existían: {len(existing_ids)})")

    # 4. Preparar datos
    ids        = [c["chunk_id"]   for c in new_chunks]
    documents  = [c["raw_content"] for c in new_chunks]  # sin prefijo para embedding
    metadatas  = [build_metadata(c) for c in new_chunks]

    # 5. Generar embeddings (usa raw_content, no el content con prefijo)
    embeddings = generate_embeddings(documents, model)

    # 6. Insertar en ChromaDB en lotes
    print("[INFO] Insertando en ChromaDB...")
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Insertando lotes"):
        collection.add(
            ids=ids[i : i + BATCH_SIZE],
            documents=documents[i : i + BATCH_SIZE],
            embeddings=embeddings[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    print(f"\n[OK] Indexación completada.")
    _print_stats(collection)


def _print_stats(collection):
    count = collection.count()
    print(f"\n--- Estadísticas de la colección '{COLLECTION_NAME}' ---")
    print(f"  Total chunks indexados: {count}")
    print(f"  Ruta ChromaDB:          {CHROMA_DB_PATH}")
    print(f"  Modelo de embeddings:   {EMBEDDING_MODEL}")
    print("-" * 52)


if __name__ == "__main__":
    main()
