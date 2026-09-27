import time
from statistics import mean, median

from src.retrieval import SemanticRetriever


QUERIES = [
    "deep learning neural networks",
    "transformer models for natural language processing",
    "reinforcement learning and decision making",
    "computer vision image classification",
    "optimization methods for machine learning",
    "generative models and representation learning",
    "graph neural networks",
    "machine learning model evaluation",
    "attention mechanisms in neural networks",
    "unsupervised learning techniques",
]

TOP_K = 5


def main():
    print("=" * 70)
    print("PARALLAX RAG - RETRIEVAL PERFORMANCE BENCHMARK")
    print("=" * 70)

    retriever = SemanticRetriever()

    latencies = []

    print()
    print(
        f"Testing {len(QUERIES)} queries "
        f"with top_k={TOP_K}"
    )
    print()

    for index, query in enumerate(
        QUERIES,
        start=1,
    ):

        start = time.perf_counter()

        results = retriever.search(
            query,
            top_k=TOP_K,
        )

        latency = (
            time.perf_counter() - start
        )

        latencies.append(latency)

        print(
            f"{index:02d}. "
            f"{latency * 1000:.2f} ms | "
            f"{len(results)} results | "
            f"{query}"
        )

    print()

    if not latencies:
        print("No benchmark results.")
        return

    average_latency = mean(latencies)
    median_latency = median(latencies)
    minimum_latency = min(latencies)
    maximum_latency = max(latencies)

    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"Average latency: "
        f"{average_latency * 1000:.2f} ms"
    )

    print(
        f"Median latency:  "
        f"{median_latency * 1000:.2f} ms"
    )

    print(
        f"Minimum latency: "
        f"{minimum_latency * 1000:.2f} ms"
    )

    print(
        f"Maximum latency: "
        f"{maximum_latency * 1000:.2f} ms"
    )

    print(
        f"Queries tested:  "
        f"{len(QUERIES)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()