import argparse
import json
import time
from pathlib import Path

import chromadb

from src.embeddings import EmbeddingModel


CHUNKS_FILE = Path("data/processed/chunks.jsonl")
CHROMA_PATH = Path("chroma_db")

COLLECTION_NAME = "parallax_rag_chunks"

BATCH_SIZE = 32


def load_chunks(path: Path):
    """
    Load chunk records from JSONL.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}"
        )

    chunks = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not record.get("chunk_id"):
                continue

            if not record.get("text"):
                continue

            chunks.append(record)

    return chunks


def get_or_create_collection(
    client,
    reset: bool = False,
):
    """
    Create the ChromaDB collection.

    If reset=True, remove the existing collection first.
    """

    if reset:
        try:
            client.delete_collection(
                name=COLLECTION_NAME
            )

            print(
                f"Deleted existing collection: "
                f"{COLLECTION_NAME}"
            )

        except Exception:
            # Collection may not exist.
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "ArXiv ML chunks for "
                "Parallax RAG project"
            ),
            "embedding_model": "all-MiniLM-L6-v2",
            "distance_metric": "cosine",
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate embeddings and build "
            "the ChromaDB vector store."
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the collection.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PARALLAX RAG - VECTOR STORE BUILD")
    print("=" * 60)

    chunks = load_chunks(CHUNKS_FILE)

    if not chunks:
        raise RuntimeError(
            "No valid chunks found."
        )

    print(
        f"Loaded {len(chunks):,} chunks."
    )

    print()
    print("Connecting to ChromaDB...")

    CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = get_or_create_collection(
        client,
        reset=args.reset,
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    embedding_model = EmbeddingModel()

    print(
        f"Embedding dimension: "
        f"{embedding_model.dimension}"
    )

    total_embedding_time = 0.0
    total_chunks = len(chunks)

    build_start = time.perf_counter()

    for start in range(
        0,
        total_chunks,
        BATCH_SIZE,
    ):

        batch = chunks[
            start:start + BATCH_SIZE
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        ids = [
            item["chunk_id"]
            for item in batch
        ]

        metadata = [
            {
                "document_id": item[
                    "document_id"
                ],
                "chunk_index": int(
                    item["chunk_index"]
                ),
                "title": item.get(
                    "title",
                    "",
                ),
                "source": item.get(
                    "source",
                    "unknown",
                ),
            }
            for item in batch
        ]

        embeddings, elapsed = (
            embedding_model.encode(
                texts,
                batch_size=BATCH_SIZE,
            )
        )

        total_embedding_time += elapsed

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadata,
        )

        processed = min(
            start + BATCH_SIZE,
            total_chunks,
        )

        print(
            f"Indexed {processed:,}/"
            f"{total_chunks:,} chunks | "
            f"Batch embedding time: "
            f"{elapsed:.2f}s"
        )

    total_build_time = (
        time.perf_counter() - build_start
    )

    final_count = collection.count()

    average_embedding_time = (
        total_embedding_time / total_chunks
    )

    print()
    print("=" * 60)
    print("VECTOR STORE BUILD COMPLETE")
    print("=" * 60)

    print(
        f"Chunks processed:       "
        f"{total_chunks:,}"
    )

    print(
        f"ChromaDB records:        "
        f"{final_count:,}"
    )

    print(
        f"Embedding model:         "
        f"{embedding_model.model_name}"
    )

    print(
        f"Embedding dimension:     "
        f"{embedding_model.dimension}"
    )

    print(
        f"Total embedding time:    "
        f"{total_embedding_time:.2f}s"
    )

    print(
        f"Average per chunk:       "
        f"{average_embedding_time:.4f}s"
    )

    print(
        f"Total build time:        "
        f"{total_build_time:.2f}s"
    )

    if total_build_time > 0:
        print(
            f"Overall throughput:      "
            f"{total_chunks / total_build_time:.2f} "
            f"chunks/sec"
        )

    print(
        f"ChromaDB location:       "
        f"{CHROMA_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()