import time
from typing import List, Tuple

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Wrapper around Sentence Transformers with timing support.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.model_name = model_name

        print(
            f"Loading embedding model: "
            f"{self.model_name}"
        )

        load_start = time.perf_counter()

        self.model = SentenceTransformer(
            self.model_name
        )

        self.load_time = time.perf_counter() - load_start

        print(
            f"Model loaded in "
            f"{self.load_time:.2f} seconds"
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> Tuple[List[List[float]], float]:
        """
        Generate embeddings and return embeddings + elapsed time.
        """

        if not texts:
            return [], 0.0

        start_time = time.perf_counter()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        elapsed = time.perf_counter() - start_time

        return embeddings.tolist(), elapsed

    @property
    def dimension(self) -> int:
        """
        Return embedding vector dimension.
        """

        return self.model.get_sentence_embedding_dimension()