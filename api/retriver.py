from fastapi import Depends, HTTPException
from fastapi.routing import APIRouter

from api.state import get_naive_retriver, get_reranker_retriver
from retrieval.naive import NaiveRetriever

router = APIRouter(prefix="/retriver")


@router.get("/naive")
async def naive_retriver(
    query: str,
    retriver: NaiveRetriever = Depends(get_naive_retriver),
):
    return await retriver.aget_context(query)


@router.get("/reranker")
async def reranker_retriver(
    query: str,
    retriver: NaiveRetriever = Depends(get_reranker_retriver),
):
    return await retriver.aget_context(query)


@router.get("/describe")
async def describe(
    naive: NaiveRetriever = Depends(get_naive_retriver),
):
    if not naive.vector_store.async_collection:
        raise HTTPException(status_code=404, detail="Vector store not found")
    return {
        "configuration": naive.vector_store.async_collection.configuration_json,
        "count": await naive.vector_store.async_collection.count(),
    }

