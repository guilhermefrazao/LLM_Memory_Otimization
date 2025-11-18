import json

from evaluate.ragas import evaluate_ragas
import json
import wandb
from statistics import median
import pandas as pd

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

    context_generated = read_from_json("output/model_answer_reranker.json")

    wandb_init(title="mamba_reranker")

    result_score = []

    for j, i in enumerate(context_generated):
        answer = i["resposta_do_model"]
        rag = []
        sample_mem = i["resposta_correta"]
        initial_prompt = context_dataset[j]["pergunta"]
        result = evaluate_ragas(questions=[initial_prompt], ground_truths=[sample_mem], contexts=[rag], answers=[answer])
        result_score.append(result)

    df_results = pd.DataFrame(result_score)

    print(df_results)

    for metric_name in df_results.columns:

        values = df_results[metric_name].dropna().tolist()

        if not values:
            print(f"Aviso: Métrica '{metric_name}' não tem valores para logar.")
            continue

        print(f"A logar estatísticas para: {metric_name}")


        wandb.log({
            f"{metric_name}/mean": float(sum(values) / len(values)),
            f"{metric_name}/min": float(min(values)),
            f"{metric_name}/max": float(max(values)),
            f"{metric_name}/median": float(median(values)),
        })

        wandb.log({
            f"{metric_name}/distribution": wandb.Histogram(values)
        })

    wandb.finish()