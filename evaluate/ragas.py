import os
import json
import torch
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_similarity,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

def _get_llm_and_embeddings():
    """
    Configura o LLM e embeddings para o ragas.
    Tenta usar OpenAI se a chave estiver disponível, senão usa HuggingFace local.
    """

    from openai import OpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory
    import os

    # Certifique-se que sua chave está no ambiente ou passe diretamente
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Maneira correta e nativa do Ragas v2+
    llm = llm_factory(model="gpt-4o-mini", client=openai_client)
    embeddings = embedding_factory(type="openai", model="text-embedding-3-small", client=openai_client)

    return llm, embeddings


def write_json(data, file_path):
    """Função auxiliar para salvar os resultados."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Resultados salvos em: {file_path}")
    except Exception as e:
        print(f"Erro ao salvar JSON em {file_path}: {e}")

def evaluate_ragas(questions, ground_truths, contexts, answers, title="mamba"):
    """
    Executa o RAGAs no dataset fornecido.
    """
    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts,
        "answer": answers
    })

    print(f"\n==== Avaliando: {title} ====\n")

    llm, embeddings = _get_llm_and_embeddings()

    metrics = [
        answer_correctness,
        answer_similarity,
    ]


    if llm:
        answer_correctness.llm = llm
    if embeddings:
        answer_correctness.embeddings = embeddings
        answer_similarity.embeddings = embeddings

    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )

        print("\n--- Resultados da Avaliação ---")


        return result.scores[0]

    except Exception as e:
        print(f"Erro durante avaliação RAGAS: {e}")
        return {
            "answer_correctness": 0.0,
            "answer_similarity": 0.0,
        }