from .base import Retriever
from .store import ChromaVectorStore
from .models import RerankerModel

class RerankerRetriever(Retriever):
    """Realiza busca vetorial + Reranker."""

    def __init__(self, 
                 vector_store: ChromaVectorStore, 
                 reranker_model: RerankerModel, 
                 initial_k: int, 
                 final_k: int):
        self.vector_store = vector_store
        self.reranker_model = reranker_model
        self.initial_k = initial_k
        self.final_k = final_k
        print("RerankerRetriever pronto")

    def get_context(self, query: str) -> list[str]:
        print("Executando get_context")

        initial_docs = self.vector_store.similarity_search(query, k=self.initial_k)  # Busca larga
        ranked_docs = self.reranker_model.rank(query, initial_docs)  # Rerank
        return ranked_docs[:self.final_k]  # Retorna os maiores ranks

    async def aget_context(self, query: str) -> list[str]:
        print("Executando aget_context")
        initial_docs = await self.vector_store.asimilarity_search(query, k=self.initial_k)  # Busca larga
        ranked_docs = await self.reranker_model.arank(query, initial_docs)  # Rerank
        return ranked_docs[:self.final_k]  # Retorna os maiores ranks
