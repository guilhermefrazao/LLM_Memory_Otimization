import json

from evaluate.ragas import evaluate_ragas
import wandb

def wandb_init(title):
    wandb.init(
    project="ragas-memory-eval",
    name=title,
    config={
        "metric_list": ["answer_correctness", "answer_similarity"]
    }
)

def read_from_json(file_path: str):
    with open(file_path, 'r', encoding="utf-8") as f:
        content = f.read()
        context = json.loads(content) 

    return context

if __name__ == "__main__":
    context_dataset = read_from_json("data/PerLTQA/Dataset/en/qa_dataset_extraido.json")

    context_generated = read_from_json("data/PerLTQA/Dataset/en/meus_dados_salvos.json")

    wandb_init(title="mamba_baseline")

    result_score = []
    
    for j, i in enumerate(context_generated):
        answer = i["resposta_do_model"]
        rag = []
        sample_mem = i["resposta_correta"]
        initial_prompt = context_dataset[j]["pergunta"]
        result = evaluate_ragas(questions=[initial_prompt], ground_truths=[sample_mem], contexts=[rag], answers=[answer])
        result_score.append(result)

    for metric_name, values in result_score.items():
            # loga estatísticas básicas
            wandb.log({
                f"{metric_name}/mean": float(sum(values) / len(values)),
                f"{metric_name}/min": float(min(values)),
                f"{metric_name}/max": float(max(values)),
            })

            # loga distribuição (histograma)
            wandb.log({
                f"{metric_name}/distribution": wandb.Histogram(values)
            })

    wandb.finish()