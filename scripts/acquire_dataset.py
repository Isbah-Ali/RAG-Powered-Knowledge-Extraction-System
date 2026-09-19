from datasets import load_dataset
import json
from pathlib import Path


OUTPUT = Path("data/raw/arxiv_5500.jsonl")
TARGET = 5500


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PARALLAX RAG - DATASET ACQUISITION")
    print("=" * 60)

    print("\nConnecting to Hugging Face...")
    print("Dataset: CShorten/ML-ArXiv-Papers")
    print("Using streaming mode...")
    print()

    dataset = load_dataset(
        "CShorten/ML-ArXiv-Papers",
        split="train",
        streaming=True,
    )

    count = 0

    with OUTPUT.open("w", encoding="utf-8") as f:

        for row in dataset:

            title = str(row.get("title", "")).strip()

            abstract = str(
                row.get(
                    "abstract",
                    row.get("text", "")
                )
            ).strip()

            # Skip records without useful text
            if not abstract or len(abstract) < 20:
                continue

            document = {
                "id": f"arxiv_{count}",
                "title": title,
                "text": f"{title}\n\n{abstract}",
                "source": "ArXiv",
            }

            f.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                ) + "\n"
            )

            count += 1

            if count % 500 == 0:
                print(f"Collected {count}/{TARGET} documents")

            if count >= TARGET:
                break

    print()
    print("=" * 60)
    print(f"SUCCESS: Saved {count} documents")
    print(f"Location: {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()