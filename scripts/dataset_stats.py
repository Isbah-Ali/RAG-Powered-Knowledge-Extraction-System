import json
from pathlib import Path


FILE = Path("data/processed/clean_corpus.jsonl")


def main():

    documents = []

    with FILE.open("r", encoding="utf-8") as f:

        for line in f:
            documents.append(json.loads(line))

    if not documents:
        print("No documents found.")
        return

    lengths = [
        len(document["text"])
        for document in documents
    ]

    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    print(f"Documents:          {len(documents)}")
    print(
        f"Average characters: "
        f"{sum(lengths) / len(lengths):.2f}"
    )
    print(f"Minimum characters: {min(lengths)}")
    print(f"Maximum characters: {max(lengths)}")

    print("=" * 50)


if __name__ == "__main__":
    main()