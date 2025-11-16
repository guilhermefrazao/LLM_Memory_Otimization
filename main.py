from models.mamba import generate_answer_mamba
from models.transformers_model import generate_answer_transformers
from models.xlstm import generate_answer_xlstm
from writing.memory import MemoryRepository
from retrieval.store import ChromaVectorStore
from evaluate.ragas import evaluate_ragas
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
    return chosen_data


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
    # load PerLT_Mem dataset
    dataset_mem = PerLTMem()
    dataset_qa = PerLTQA()

    character_data = dataset_qa.read_json_data("data/PerLTQA/Dataset/en/perltqa_en.json")
    character_facts = dataset_mem.read_json_data("data/PerLTQA/Dataset/en/perltmem_en.json")

    character_names_mem = set(dataset_mem.extract_character_names())
    character_names_qa = set(dataset_qa.extract_character_names())
    
    # Encontra personagens que existem em AMBOS os datasets
    common_characters = list(character_names_mem.intersection(character_names_qa))
    
    if not common_characters:
        print("AVISO: Nenhum personagem comum entre os datasets de memória e perguntas!")
        # Fallback: usa qualquer personagem de QA
        common_characters = list(character_names_qa)
    
    # Escolhe um personagem aleatório que tenha dados em ambos
    character_name = find_rand(common_characters)
    print(f"-> Personagem escolhido: {character_name}")
    
    # Extrai memórias e perguntas
    samples_Mem = dataset_mem.extract_sample(character_name)
    samples_QA = dataset_qa.extract_sample(character_name)
    
    # Verifica se há perguntas de perfil disponíveis
    if samples_QA and "profile" in samples_QA and samples_QA["profile"]:
        question = find_rand(samples_QA["profile"])
        initial_prompt = question["Question"]
        ground_truth = question.get("Answer", "")  # Pega a resposta esperada se existir
    else:
        print(f"AVISO: Personagem {character_name} não tem perguntas de perfil!")
        initial_prompt = f"Tell me about {character_name}"
        ground_truth = ""
    
    # Se samples_Mem estiver vazio, usa string vazia
    if not samples_Mem:
        samples_Mem = ""
        ground_truth = ""
    
    return initial_prompt, ground_truth, character_facts



if __name__ == "__main__":
    client = PersistentClient(path="memory/db")

    vector_store = ChromaVectorStore(
    client=client,
    embed_model=HFEmbeddingModel(),
    )

    answer = ""
    rag = ""

    #Foi criado somente o processamento com 1 dos datasets.
    initial_prompt, sample_mem, character_facts = dataset_PerLQTA()

    if args.embeddings:
        embeddings = generate_embeddings(json.dumps(character_facts), client)

    if args.naiverag:
        rag = NaiveRetriever(vector_store=vector_store, k=5).get_context(initial_prompt)[0]

    elif args.reranker:
        rag = RerankerRetriever(vector_store, RerankerModel().model, 20, 5).get_context(initial_prompt)[0]

    prompt = initial_prompt + rag

    if args.mamba:
        answer = generate_answer_mamba(question=prompt)
    
    elif args.transformers:
        answer = generate_answer_transformers(query=prompt)

    elif args.xlstm:
        answer = generate_answer_xlstm(query=prompt)

    # ragas espera uma lista de listas em "contexts" (retrieved_contexts),
    # ou seja, [[doc1, doc2, ...]] por pergunta.
    contexts = [[rag]] if rag else [[]]

    result = evaluate_ragas(
        questions=[initial_prompt],
        ground_truths=[sample_mem],
        contexts=contexts,
        answers=[answer],
    )