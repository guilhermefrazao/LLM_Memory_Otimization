from .base import Retriever
from .store import ChromaVectorStore

class NaiveRetriever(Retriever):
    """Realiza busca vetorial simples."""

    def __init__(self, vector_store: ChromaVectorStore, k: int):
        self.vector_store = vector_store
        self.k = k
        print("NaiveRetriever pronto")

    def get_context(self, query: str) -> list[str]:
        print("Executando get_context")
        return self.vector_store.similarity_search(query, k=self.k)

    async def aget_context(self, query: str) -> list[str]:
        print("Executando aget_context")
        return await self.vector_store.asimilarity_search(query, k=self.k)
