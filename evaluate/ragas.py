from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
    answer_relevancy,
)

import os
import json


def evaluate_ragas(questions, ground_truths, contexts, answers, title="mamba"):
    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts,
        "answer": answers
    })
    
    print(f"\n==== Avaliando: {title} ====\n")
    
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_correctness,
            answer_relevancy,
        ]
    )

    os.makedirs("output", exist_ok=True)

    scores = result.scores

    filename = f"output/ragas_result_{title or 'default'}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)

    print(f"Resultado salvo em: {filename}")
    
    print(scores)
    
    return result
