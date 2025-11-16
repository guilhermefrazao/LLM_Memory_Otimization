import json

from evaluate.ragas import evaluate_ragas


if __name__ == "__main__":
    with open("data/PerLTQA/Dataset/en/qa_dataset_extraido.json", 'r', encoding="utf-8") as f:
        content_dataset = f.read()
        context_dataset = json.loads(content_dataset)  

    with open("data/PerLTQA/Dataset/en/meus_dados_salvos.json", 'r', encoding="utf-8") as f:
        content_generated = f.read()
        context_generated = json.loads(content_generated)  
    
    for j, i in enumerate(context_generated):
        answer = i["resposta_do_model"]
        rag = []
        sample_mem = i["resposta_correta"]
        initial_prompt = context_dataset[j]["pergunta"]
        result = evaluate_ragas(questions=[initial_prompt], ground_truths=[sample_mem], contexts=[rag], answers=[answer])