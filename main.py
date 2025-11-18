from models.mamba import generate_answer_mamba
from models.transformers_model import generate_answer_transformers
from models.xlstm import generate_answer_xlstm
from writing.memory import MemoryRepository
from retrieval.store import ChromaVectorStore
from data.PerLTQA.Dataset.dataset import PerLTMem, PerLTQA
from retrieval.models import HFEmbeddingModel, RerankerModel
from retrieval.naive import NaiveRetriever
from retrieval.reranker import RerankerRetriever
from models.utils.json_utils import write_json

import argparse
import time
import json
import random
from chromadb import PersistentClient

parser = argparse.ArgumentParser()

parser.add_argument("--reranker", action="store_true", help="Usa o reranker")
parser.add_argument("--naiverag", action="store_true", help="Usa o naive RAG")
parser.add_argument("--embeddings", action="store_true", help="Gerando os embeddings")
parser.add_argument("--mamba", action="store_true", help="Chamando o modelo mamba")
parser.add_argument("--transformers", action="store_true", help="Chamando o modelo de trasnformers")
parser.add_argument("--xlstm", action="store_true", help="Chamando o modelo xlstm")

args = parser.parse_args()


def find_rand(list: list):
    random_number = random.randint(0, len(list) - 1)
    chosen_data = list[random_number]
    return chosen_data, random_number


def generate_embeddings(documents: str, client):
    emb_model = HFEmbeddingModel()
    embeddings_docs = emb_model.embed_text(documents)
    MemoryRepository(client).add_memory(chat_id="PerLQTA_dataset", content=documents, category=0, embeddings=embeddings_docs)
    return embeddings_docs


def dataset_PerLQTA():
    """
    Carrega o dataset PerLTQA e retorna uma pergunta, memória e fatos do personagem.
    Garante que o personagem escolhido tenha tanto memórias quanto perguntas.
    """
    with open("data/PerLTQA/Dataset/en/qa_dataset_extraido.json", 'r', encoding="utf-8") as f:
        content = f.read()
        context = json.loads(content)   

    return context



if __name__ == "__main__":
    start_time = time.perf_counter()
    client = PersistentClient(path="memory/db")

    vector_store = ChromaVectorStore(
    client=client,
    embed_model=HFEmbeddingModel(),
    )

    context = dataset_PerLQTA()

    answer_list = []
    answer_dict = {"resposta_do_model": "", "resposta_correta": "", "contexto": ""}

    for i, j in enumerate(context):
        initial_prompt = context[i]["pergunta"]  

        if args.naiverag:
            rag = NaiveRetriever(vector_store=vector_store, k=5).get_context(initial_prompt)[0]

        elif args.reranker:
            ranked_docs, rag = RerankerRetriever(vector_store, RerankerModel().model, 20, 5).get_context(initial_prompt)


        else:
            rag = ""

        if args.mamba:
            answer = generate_answer_mamba(question=initial_prompt, base_context=rag)
        
        elif args.transformers:
            answer = generate_answer_transformers(query=initial_prompt)

        elif args.xlstm:
            answer = generate_answer_xlstm(query=initial_prompt)

        answer_dict = {
            "resposta_do_model": answer,
            "resposta_correta": context[i]["resposta_correta"],
            "contexto": rag
        }

        answer_list.append(answer_dict)

        print(f"\niteration: {i}")

    end_time = time.perf_counter()
    write_json(answer_list, "output/model_answer_reranker.json")

    print("Total_time: ", (end_time - start_time) / 60)