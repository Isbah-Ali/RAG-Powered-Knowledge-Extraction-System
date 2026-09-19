import sys


def check_import(name):
    try:
        __import__(name)
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")


print("=" * 50)
print("PARALLAX RAG - ENVIRONMENT VERIFICATION")
print("=" * 50)

print("Python:", sys.version)
print()

# Required packages
packages = [
    "pandas",
    "datasets",
    "spacy",
    "pytest",
    "sentence_transformers",
    "chromadb",
]

for package in packages:
    check_import(package)

print()

# PyTorch / CUDA check
try:
    import torch

    print("[PASS] torch")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: CPU-only environment")

except Exception as e:
    print("[INFO] torch check:", e)

print()
print("=" * 50)
print("VERIFICATION COMPLETE")
print("=" * 50)