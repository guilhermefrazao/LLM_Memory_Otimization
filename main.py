from models.mamba import generate_answer_mamba
from models.transformers import generate_answer_transformers
from models.xlstm import generate_answer_xlstm
from writing.memory import MemoryRepository
from retrieval.store import ChromaVectorStore
from data.PerLTQA.Dataset.dataset import PerLTMem, PerLTQA
from retrieval.models import HFEmbeddingModel, RerankerModel
from retrieval.naive import NaiveRetriever
from retrieval.reranker import RerankerRetriever

import argparse
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


#TODO: Adicionar a lógica dos outros datasets para realizar a inferência com outros tipos de memória
#TODO: Escalar a implementação da avaliação para o dataset, por enquanto somente avalia 1 pergunta e resposta por código rodado.
#TODO: Após realizar a avaliação, salvar os resultados dentro de algum arquivo, separado por modelo utilizado.
#TODO: Verificar a qualidade do RAG feito (Recupera documentos similares e relevantes?).
#TODO: Analisar se com base na arquitetura do modelo o resultado é satisfatório?


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
    with open("data/PerLTQA/Dataset/en/qa_dataset_extraido.json", 'r', encoding="utf-8") as f:
        content = f.read()
        context = json.loads(content)   

    chosen_data, random_character = find_rand(context)

    initial_prompt = chosen_data["pergunta"]    

    sample_mem =  chosen_data["resposta_correta"]

    return initial_prompt, sample_mem, context, random_character



if __name__ == "__main__":
    client = PersistentClient(path="memory/db")

    vector_store = ChromaVectorStore(
    client=client,
    embed_model=HFEmbeddingModel(),
    )

    #Foi criado somente o processamento com 1 dos datasets.
    initial_prompt, sample_mem, context, random_character = dataset_PerLQTA()

    if args.embeddings:
        embeddings = generate_embeddings(json.dumps(context), client)

    if args.naiverag:
        rag = NaiveRetriever(vector_store=vector_store, k=5).get_context(initial_prompt)

    elif args.reranker:
        rag = RerankerRetriever(vector_store, RerankerModel().model, 20, 5).get_context(initial_prompt)

    else:
        rag = []

    if args.mamba:
        answer = generate_answer_mamba(question=initial_prompt, base_context=rag)
    
    elif args.transformers:
        answer = generate_answer_transformers(question=initial_prompt)

    elif args.xlstm:
        answer = generate_answer_xlstm(question=initial_prompt)