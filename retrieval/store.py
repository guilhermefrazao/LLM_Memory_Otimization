from chromadb.api import AsyncClientAPI, ClientAPI


class ChromaVectorStore:
    def __init__(
        self,
        client: ClientAPI | None = None,
        embed_model = None,
        async_client: AsyncClientAPI | None = None,
    ):
        if embed_model is None:
            raise ValueError("embed_model is required")

        self.collection = None
        if client is not None:
            self.collection = client.get_or_create_collection("memories")
        self.embed_model = embed_model
        self.async_client = async_client
        self.async_collection = None

    async def ainit_collection(self):
        if self.async_client is None:
            raise ValueError("Async client not initialized")
        self.async_collection = await self.async_client.get_or_create_collection("memories")

    def _debug_results(self, r):
        print(r)

    def similarity_search(self, query, k, max_distance=2.0):
        if self.collection is None:
            raise ValueError("Client not initialized")

        emb = self.embed_model.embed_text(query)
        r = self.collection.query(
            query_embeddings=[emb], n_results=k, include=["documents", "distances"]
        )
        results = []
        if (
            not r["distances"]
            or not r["distances"][0]
            or not r["documents"]
            or not r["documents"][0]
        ):
            return results

        for i in range(len(r["distances"][0])):
            if r["distances"][0][i] > max_distance:
                continue
            results.append(
                r["documents"][0][i]
            )
        return results

    async def asimilarity_search(self, query, k, max_distance=2.0):
        if self.async_collection is None:
            raise ValueError("Async client not initialized")

        emb = await self.embed_model.aembed_text(query)
        result = await self.async_collection.query(
            query_embeddings=[emb], n_results=k, include=["documents", "distances"]
        )
        self._debug_results(result)
        results = []
        if (
            not result["distances"]
            or not result["distances"][0]
            or not result["documents"]
            or not result["documents"][0]
        ):
            return results

        for i in range(len(result["distances"][0])):
            if result["distances"][0][i] > max_distance:
                continue
            results.append(
                result["documents"][0][i]
            )

        return results
