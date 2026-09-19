import json
from pathlib import Path

from src.preprocessing import clean_text, is_english


INPUT = Path("data/raw/arxiv_5500.jsonl")
OUTPUT = Path("data/processed/clean_corpus.jsonl")


def main():

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    removed = 0

    print("=" * 60)
    print("PARALLAX RAG - DATA PREPROCESSING")
    print("=" * 60)

    with INPUT.open("r", encoding="utf-8") as infile, \
         OUTPUT.open("w", encoding="utf-8") as outfile:

        for line in infile:

            total += 1

            document = json.loads(line)

            original_text = document.get("text", "")

            # Clean text
            text = clean_text(original_text)

            # Remove invalid documents
            if not text:
                removed += 1
                continue

            if len(text) < 50:
                removed += 1
                continue

            if not is_english(text):
                removed += 1
                continue

            cleaned_document = {
                "id": document["id"],
                "title": document["title"],
                "text": text,
                "source": document["source"],
            }

            outfile.write(
                json.dumps(
                    cleaned_document,
                    ensure_ascii=False
                ) + "\n"
            )

            kept += 1

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Total documents:   {total}")
    print(f"Documents kept:    {kept}")
    print(f"Documents removed: {removed}")

    print()
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()