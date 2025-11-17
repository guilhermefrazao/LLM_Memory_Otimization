🚀 Como executar o projeto

Você pode rodar o pipeline com diferentes modos via linha de comando usando flags do argparse.

Login no hugging face:

```bash
huggingface-cli login
```

Rodar o pipeline no modo padrão:

```bash
python main.py
```


⚙️ Rodando com argumentos

Você pode ativar diferentes estratégias do RAG adicionando flags:

🔹 RAG Naive

```bash
python main.py --naiverag
```

🔹 Reranker

```bash
python main.py --reranker
```

🔹 Embeddings (Gerar novamente os embeddings dos datasets.)

```bash
python main.py --embeddings
```

🔹 Combinação de opções

Se quiser combinar várias etapas, basta passar múltiplas flags:

```bash
python main.py --naiverag --reranker --embeddings
```


# Datasets 

Datasets utilizados para treinar modelos de Linguagem para tarefas relacionadas com a memória das LLMs.

1. **PerLTQA** 
Sobre - Dataset de QA focado em memória de longo prazo, que inclui memória semântica (Envolve fatos sobre o mundo e fatos pessoais/relacionamentos) e memória episódica (Histórico pessoal de dialogos e experiências),

Paper - https://arxiv.org/pdf/2402.16288
Repositório - https://github.com/Elvin-Yiming-Du/PerLTQA


2. **locomo**
Sobre - Dataset de conversas muito longo, é Multimodal e aprensenta várias conversas com diversos meses de diferença entre elas.

Paper - https://arxiv.org/pdf/2402.17753
Repositório - https://github.com/snap-research/LoCoMo

3. **LoCoGen**
Sobre - Datase para memória de Longo prazo.

Paper - https://aclanthology.org/2025.findings-acl.1014.pdf
Repositório - https://github.com/JamesLLMs/LoCoGen

# Rag

**Naive RAG** 
**Rag Rerank**


# Models

Modelos treinados, otimizados para memória.

1. **Transformers-like**
2. **x-LSMT**
3. **Mamba**


# Evaluation
**RAG Evaluation com Ragas**

Para avaliar a qualidade do pipeline de RAG, utilizamos o Ragas, um framework projetado especificamente para medir o desempenho de sistemas de Retrieval-Augmented Generation.
Ele analisa tanto a etapa de recuperação (retrieval) quanto a qualidade da resposta gerada (generation).

A função abaixo realiza toda a avaliação usando um conjunto de perguntas, respostas esperadas, contextos recuperados e respostas do modelo.

# API para inferência
Para aproveitar o caráter assíncrono do FastAPI, é necessário utilizar um client assíncrono do chromadb, que só está disponível no modo HTTP.
Para isso, rode um server do chroma:
```bash
# Instalar dependências
uv sync
# Rode o servidor
uv run chroma run --path ./memory/db --host 0.0.0.0 --port 8010
```
Ou, sem uv:
```bash
python -m venv .venv
# Linux
.venv/bin/activate
# Windows
.venv\Scripts\activate.bat

# Instalar dependências
pip install -r requirements.txt

# Rode o servidor
chroma run --path ./memory/db --host 0.0.0.0 --port 8010
```
Rodar a API:
```bash
uv run api/main.py
```
Acesse a doc OpenAPI em http://localhost:8009/docs

# Estrutura das pastas
```.
├── data    # Datasets para treino
│   ├── LoCoGen
│   ├── locomo 
│   └── PerLTQA
├── memory    # Memórias dos chats <Sujeito à mudança>
│   ├── memory_chat_1  # Exemplo de um chat
│   ├── faiss.index    # Banco vetorial
│   └── lookup.json    # Banco map
├── retrieval
│   ├── base.py      # Interface Retriever
│   ├── models.py    # Embedding e Reranker importados
│   ├── naive.py     # NaiveRetriever
│   ├── reranker.py  # RerankerRetriever
│   └── store.py     # Busca vetorial <Sujeito à mudança>
└── writing
    └── memory_writer.py    # Salva textos na memória <Place-holder>```
