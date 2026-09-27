import json
import time
from pathlib import Path

from src.chunking import recursive_split


INPUT = Path("data/processed/clean_corpus.jsonl")
OUTPUT = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.perf_counter()

    document_count = 0
    chunk_count = 0

    chunk_lengths = []

    print("=" * 60)
    print("PARALLAX RAG - DOCUMENT CHUNKING")
    print("=" * 60)
    print(f"Input:         {INPUT}")
    print(f"Output:        {OUTPUT}")
    print(f"Chunk size:    {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print()

    with (
        INPUT.open("r", encoding="utf-8") as infile,
        OUTPUT.open("w", encoding="utf-8") as outfile,
    ):

        for line in infile:
            document = json.loads(line)

            document_id = document["id"]
            title = document.get("title", "")
            source = document.get("source", "")
            text = document.get("text", "")

            chunks = recursive_split(
                text,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            for chunk_index, chunk_text in enumerate(chunks):

                chunk_id = (
                    f"{document_id}_chunk_{chunk_index}"
                )

                chunk_record = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "title": title,
                    "source": source,
                    "text": chunk_text,
                }

                outfile.write(
                    json.dumps(
                        chunk_record,
                        ensure_ascii=False,
                    ) + "\n"
                )

                chunk_count += 1
                chunk_lengths.append(len(chunk_text))

            document_count += 1

            if document_count % 500 == 0:
                print(
                    f"Processed documents: "
                    f"{document_count:,} | "
                    f"Chunks: {chunk_count:,}"
                )

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(f"Documents processed: {document_count:,}")
    print(f"Chunks created:      {chunk_count:,}")

    if chunk_lengths:
        average_length = (
            sum(chunk_lengths) / len(chunk_lengths)
        )

        print(
            f"Average chunk size:  "
            f"{average_length:.2f} characters"
        )

        print(
            f"Minimum chunk size:  "
            f"{min(chunk_lengths)} characters"
        )

        print(
            f"Maximum chunk size:  "
            f"{max(chunk_lengths)} characters"
        )

    print(f"Processing time:      {elapsed:.2f} seconds")
    print(f"Output:               {OUTPUT}")

    if document_count:
        print(
            f"Average chunks/document: "
            f"{chunk_count / document_count:.2f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()