from typing import Any, Dict, List

import chromadb

from src.embeddings import EmbeddingModel


DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_COLLECTION_NAME = "parallax_rag_chunks"


class SemanticRetriever:
    """
    Semantic retrieval over the ChromaDB vector store.
    """

    def __init__(
        self,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.client = chromadb.PersistentClient(
            path=chroma_path
        )

        try:
            self.collection = (
                self.client.get_collection(
                    name=collection_name
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection "
                f"'{collection_name}' was not found. "
                f"Run build_vector_store.py first."
            ) from exc

        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else EmbeddingModel()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        collection_count = (
            self.collection.count()
        )

        if collection_count == 0:
            return []

        top_k = min(
            top_k,
            collection_count,
        )

        query_embedding, _ = (
            self.embedding_model.encode(
                [query],
                batch_size=1,
            )
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding[0]
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        output = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            output.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return output