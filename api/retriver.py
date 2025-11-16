from fastapi import Depends
from fastapi.routing import APIRouter

from api.state import get_naive_retriver, get_reranker_retriver
from retrieval.naive import NaiveRetriever

router = APIRouter()


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
