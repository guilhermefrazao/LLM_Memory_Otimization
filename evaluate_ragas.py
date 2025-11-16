import json

from evaluate.ragas import evaluate_ragas

def read_from_json(file_path: str):
    with open(file_path, 'r', encoding="utf-8") as f:
        content = f.read()
        context = json.loads(content) 

    return context

if __name__ == "__main__":
    context_dataset = read_from_json("data/PerLTQA/Dataset/en/qa_dataset_extraido.json")

    context_generated = read_from_json("data/PerLTQA/Dataset/en/meus_dados_salvos.json")
    
    for j, i in enumerate(context_generated):
        answer = i["resposta_do_model"]
        rag = []
        sample_mem = i["resposta_correta"]
        initial_prompt = context_dataset[j]["pergunta"]
        result = evaluate_ragas(questions=[initial_prompt], ground_truths=[sample_mem], contexts=[rag], answers=[answer])