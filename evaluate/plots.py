import matplotlib.pyplot as plt

def plot_ragas_boxplot(scores):
    plt.figure(figsize=(8,5))
    plt.boxplot([
        scores["answer_correctness"], 
        scores["answer_similarity"]
    ], labels=["Correctness", "Similarity"])
    plt.title("Distribuição das métricas RAGAS")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_ragas_scatter(scores):
    plt.figure(figsize=(6,6))
    plt.scatter(scores["answer_correctness"], scores["answer_similarity"])
    plt.xlabel("Correctness")
    plt.ylabel("Similarity")
    plt.title("Correlação entre métricas RAGAS")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_ragas_hist(scores):
    plt.figure(figsize=(10,4))
    
    plt.hist(scores["answer_correctness"], bins=10, alpha=0.6, label="Correctness")
    plt.hist(scores["answer_similarity"], bins=10, alpha=0.6, label="Similarity")

    plt.title("Histograma das métricas RAGAS")
    plt.xlabel("Score")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
