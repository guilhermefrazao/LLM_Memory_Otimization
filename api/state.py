## Instanciar os retrivers no contexto do app

import os
from contextlib import asynccontextmanager
import sys

from chromadb import AsyncHttpClient
from chromadb.api import AsyncClientAPI
from fastapi import FastAPI

from retrieval.models import HFEmbeddingModel
from retrieval.naive import NaiveRetriever
from retrieval.reranker import RerankerRetriever, RerankerModel
from retrieval.store import ChromaVectorStore

NAIVE_TOP_K = int(os.getenv("NAIVE_TOP_K", 5))
RERANKER_INITAL_TOP_K = int(os.getenv("RERANKER_INITAL_TOP_K", 5))
RERANKER_FINAL_TOP_K = int(os.getenv("RERANKER_FINAL_TOP_K", 5))

_chroma_client: AsyncClientAPI | None = None
_store: ChromaVectorStore | None = None
_reranker_model: RerankerModel | None = None
_naive: NaiveRetriever | None = None
_reranker: RerankerRetriever | None = None


async def _init_globals():
    global _store
    global _chroma_client
    global _reranker_model
    global _naive
    global _reranker

    if _chroma_client is None:
        try:
            _chroma_client = await AsyncHttpClient(
                host="localhost",
                port=8010,
            )
            await _chroma_client.heartbeat()
        except Exception as e:
            print(f"Verifique se o chroma está rodando: {e}")
            print("Para iniciar ele em modo server rode: uv run chroma run --path ./memory/db --host 0.0.0.0 --port 8010")
            sys.exit(1)

    if _store is None:
        _store = ChromaVectorStore(
            embed_model=HFEmbeddingModel(),
            async_client=_chroma_client,
        )
        await _store.ainit_collection()

    if _reranker_model is None:
        _reranker_model = RerankerModel()

    if _naive is None:
        _naive = NaiveRetriever(_store, NAIVE_TOP_K)

    if _reranker is None:
        _reranker = RerankerRetriever(
            _store,
            _reranker_model,
            RERANKER_INITAL_TOP_K,
            RERANKER_FINAL_TOP_K,
        )

def get_reranker_retriver() -> RerankerRetriever:
    global _reranker
    if _reranker is None:
        raise Exception("Reranker Retriever not initialized")

    return _reranker


def get_naive_retriver() -> NaiveRetriever:
    global _naive
    if _naive is None:
        raise Exception("Naive Retriever not initialized")

    return _naive


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _init_globals()
    yield
