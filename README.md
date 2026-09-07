# Multi-Tool RAG & Stateful AI Agents with LangChain and LangGraph

An end-to-end practical guide and implementation of Retrieval-Augmented Generation (RAG) and stateful multi-tool conversational agents using LangChain, LangGraph, Groq, and Hugging Face embeddings.

## Project Structure

```markdown
.
│
├── data/ # Local PDF knowledge bases
│ ├── agriculture.pdf # Agricultural domain document
│ ├── dengue.pdf # Dengue fever health document
│ └── m2m_strategy_and_objectives_development.pdf # NASA Moon to Mars strategy
│
├── RAG_course.ipynb # Main tutorial notebook
├── pyproject.toml # Project metadata and PEP 621 dependencies (uv / pip compatible)
├── requirements.txt # Standard pip requirements
├── .env.example # Environment variable template
├── .gitignore # Git ignore configuration
└── README.md # Project documentation
```

## Features

1. **Direct LLM Invocations**: Zero-shot querying with Groq-hosted open LLMs (such as openai/gpt-oss-20b or LLaMA models).
2. **Document Ingestion & Chunking**: Local document loading via PyPDFLoader.
3. **Dense Vector Embeddings & Similarity Search**: Embeddings via mixedbread-ai/mxbai-embed-large-v1 stored in InMemoryVectorStore.
4. **LCEL RAG Pipeline**: Composable RAG chain with ChatPromptTemplate, retriever, and StrOutputParser.
5. **Multi-Domain Tool Routing**: Dynamic multi-tool ReAct agent routing questions to NASA, Agriculture, or Dengue knowledge bases.
6. **Stateful Graph Workflows (LangGraph)**: Cyclic graph architecture featuring tool nodes, conditional routing, and MemorySaver checkpoints for multi-turn chat persistence.

## Getting Started

### 1. Prerequisites

- Python >= 3.10
- A [Groq API Key](https://console.groq.com/)

### 2. Installation

Using uv:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

Or using standard pip:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the sample environment file and set your Groq API key:

```bash
cp .env.example .env
```

Edit .env:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Running the Notebook

Start Jupyter Notebook or JupyterLab:

```bash
jupyter notebook RAG_course.ipynb
```

You can also run this notebook directly in Google Colab: Colab secrets (userdata.get('GROQ_API_KEY')) are automatically detected.
