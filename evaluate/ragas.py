from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
    answer_relevancy,
)
import os

load_dotenv()

def _get_llm_and_embeddings():
    """
    Configura o LLM e embeddings para o ragas.
    Tenta usar OpenAI se a chave estiver disponível, senão usa HuggingFace local.
    """
    try:
        # Tenta usar OpenAI se a chave estiver configurada
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            
            llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
            embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
            print("-> Usando OpenAI para avaliação RAGAS")
            return llm, embeddings
    except Exception as e:
        print(f"-> Não foi possível usar OpenAI: {e}")
    
    # Fallback: usa HuggingFace local
    try:
        from langchain_community.llms import HuggingFacePipeline
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from transformers import pipeline
        
        print("-> Carregando modelo HuggingFace local para avaliação RAGAS...")
        
        # Usa um modelo pequeno e rápido para avaliação
        pipe = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256,
            device_map="auto"
        )
        
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        llm = LangchainLLMWrapper(hf_llm)
        
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        
        print("-> Modelo HuggingFace carregado para RAGAS")
        return llm, embeddings
        
    except Exception as e:
        print(f"-> Erro ao carregar HuggingFace: {e}")
        print("-> RAGAS rodará sem LLM customizado (pode gerar erros)")
        return None, None


def evaluate_ragas(questions, ground_truths, contexts, answers, title=""):
    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts,
        "answer": answers
    })
    
    print(f"\n==== Avaliando: {title} ====\n")
    
    # Configura LLM e embeddings
    llm, embeddings = _get_llm_and_embeddings()
    
    # Define as métricas a usar
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_correctness,
        answer_relevancy,
    ]
    
    # Configura o LLM e embeddings nas métricas se disponíveis
    if llm and embeddings:
        for metric in metrics:
            if hasattr(metric, 'llm'):
                metric.llm = llm
            if hasattr(metric, 'embeddings'):
                metric.embeddings = embeddings
    
    try:
        result = evaluate(
            dataset,
            metrics=metrics,
        )
        print(result)
        return result
    except Exception as e:
        print(f"Erro durante avaliação RAGAS: {e}")
        return {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "answer_relevancy": 0.0,
            "error": str(e)
        }
