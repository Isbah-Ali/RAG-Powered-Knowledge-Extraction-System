# RAG-Powered Knowledge Extraction System

A production-oriented **Retrieval-Augmented Generation (RAG) knowledge extraction system** designed to process large collections of research documents, retrieve relevant information through semantic search, and generate context-grounded responses using Large Language Models (LLMs).

The system combines data ingestion, preprocessing, recursive document chunking, semantic embeddings, persistent vector storage, similarity-based retrieval, and LLM-powered generation in a modular pipeline. It emphasizes data quality, reproducibility, testing, measurable performance, and maintainable architecture.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [Retrieval and RAG Generation](#retrieval-and-rag-generation)
- [Benchmarking and Testing](#benchmarking-and-testing)
- [Current Status](#current-status)
- [Engineering Principles](#engineering-principles)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview

The goal is to build an end-to-end knowledge extraction system that answers questions over a large collection of machine learning research papers.

The system follows a modular RAG pipeline:

1. **Data ingestion:** Acquire research documents from Hugging Face datasets.
2. **Preprocessing:** Normalize Unicode, remove HTML markup, clean whitespace, and filter documents by language.
3. **Document chunking:** Split long documents into smaller, overlapping text segments while preserving context.
4. **Embedding generation:** Convert chunks into dense semantic vectors using Sentence Transformers.
5. **Vector storage:** Store embeddings, document text, and metadata in ChromaDB.
6. **Semantic retrieval:** Retrieve the most relevant chunks for a user's query.
7. **Context construction:** Format retrieved evidence into a structured prompt.
8. **LLM generation:** Generate responses grounded in the retrieved research context.

## Key Features

- **Scalable data processing:** Handles thousands of research documents through a structured ingestion and preprocessing pipeline.
- **Modular preprocessing:** Independently testable text-cleaning and language-filtering functions.
- **Recursive chunking:** Configurable chunk size and overlap to preserve contextual continuity.
- **Semantic embeddings:** Uses the `all-MiniLM-L6-v2` Sentence Transformer model.
- **Persistent vector database:** Stores embeddings and metadata in ChromaDB.
- **Semantic top-K retrieval:** Retrieves relevant research passages based on vector similarity.
- **Grounded generation:** Supplies retrieved evidence to an LLM instead of relying exclusively on the model's internal knowledge.
- **Source metadata preservation:** Retains document IDs, titles, source information, and chunk indices.
- **Error handling:** Accounts for invalid queries, empty retrieval results, and external API failures.
- **Performance benchmarking:** Measures embedding throughput and retrieval latency.
- **Automated testing:** Uses pytest to validate preprocessing, chunking, and retrieval behavior.

## Architecture

```text
Research Documents (ArXiv)
          |
          v
    Data Ingestion
          |
          v
   Text Preprocessing
          |
          v
  Recursive Chunking
          |
          v
 Sentence Transformer
     Embeddings
          |
          v
      ChromaDB
    Vector Store
          |
          v
  Semantic Retrieval
          |
          v
   Retrieved Context
          |
          v
    Prompt Builder
          |
          v
       LLM API
          |
          v
  Grounded Response
   + Source Metadata
```

The retrieval and generation components are separated so that retrieval quality, response quality, and system performance can be evaluated independently.

## Dataset

The initial corpus uses the **ML-ArXiv-Papers** dataset available through Hugging Face.

A subset of approximately **5,500 machine learning research documents** is used for the initial implementation.

Documents are stored in JSONL format with structured fields such as:

```json
{
  "id": "arxiv_0",
  "title": "Example Research Paper",
  "text": "Title and abstract content...",
  "source": "ArXiv"
}
```

### Data Processing Pipeline

```text
Hugging Face Dataset
         |
         v
  Dataset Acquisition
         |
         v
   Raw JSONL Files
         |
         v
    Preprocessing
         |
         v
   Cleaned Corpus
         |
         v
  Document Chunking
         |
         v
    Vector Indexing
```

Raw documents, processed datasets, generated chunks, and vector database files are stored locally and excluded from Git version control to keep the repository lightweight.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, PyArrow |
| Machine Learning | PyTorch, scikit-learn |
| NLP | Sentence Transformers, spaCy, langdetect |
| Text Processing | BeautifulSoup4, Unicode normalization, regular expressions |
| Dataset Acquisition | Hugging Face Datasets |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| LLM Integration | DeepSeek, OpenRouter, Groq |
| Testing | pytest |
| Development Tools | Git, GitHub, VS Code |

## Project Structure

```text
RAG-Powered-Knowledge-Extraction-System/
│
├── data/
│   ├── raw/
│   │   └── arxiv_5500.jsonl
│   └── processed/
│       ├── clean_corpus.jsonl
│       └── chunks.jsonl
│
├── chroma_db/
│
├── scripts/
│   ├── acquire_dataset.py
│   ├── dataset_stats.py
│   ├── preprocess_dataset.py
│   ├── create_chunks.py
│   ├── build_vector_store.py
│   ├── benchmark_retrieval.py
│   └── verify_env.py
│
├── src/
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── retrieval.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_chunking.py
│   └── test_retrieval.py
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

*Note: The structure represents the core repository components. Generation-related modules may be added or reorganized as development progresses.*

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version supported by the project's dependencies
- Git
- An LLM API key for generation functionality

### 1. Clone the Repository

```bash
git clone https://github.com/Isbah-Ali/RAG-Powered-Knowledge-Extraction-System.git
cd RAG-Powered-Knowledge-Extraction-System
```

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the Environment

```bash
python scripts/verify_env.py
```

The verification script checks core dependencies and detects available PyTorch/CUDA support.

## Running the Pipeline

Execute the following stages in order.

### 1. Acquire the Dataset

```bash
python scripts/acquire_dataset.py
```

The acquisition script uses Hugging Face streaming to retrieve the configured document subset without unnecessarily downloading the entire source dataset.

Output:

```text
data/raw/arxiv_5500.jsonl
```

### 2. Preprocess Documents

```bash
python -m scripts.preprocess_dataset
```

The preprocessing pipeline performs:

- Unicode NFKC normalization
- HTML removal using BeautifulSoup
- Whitespace normalization
- Minimum-content validation
- English-language detection

Output:

```text
data/processed/clean_corpus.jsonl
```

The script reports document-processing statistics, including the number of documents retained and removed.

### 3. Generate Document Chunks

```bash
python -m scripts.create_chunks
```

The chunking module uses recursive text splitting to preserve semantic boundaries where possible.

**Current configuration:**

| Parameter | Value |
|---|---|
| Chunk size | 500 characters |
| Chunk overlap | 75 characters |
| Splitting strategy | Recursive |
| Chunk identifiers | Deterministic |

Each chunk retains metadata such as its document ID, title, source, and chunk index.

Output:

```text
data/processed/chunks.jsonl
```

### 4. Build the Vector Store

```bash
python -m scripts.build_vector_store --reset
```

This process:

1. Loads the generated chunks.
2. Initializes the Sentence Transformer embedding model.
3. Generates embeddings in batches.
4. Records embedding performance statistics.
5. Stores vectors, text, and metadata in ChromaDB.
6. Reports the resulting collection size.

The database is stored locally under `chroma_db/`.

**Warning:** The `--reset` option rebuilds the vector collection. Use it when you intentionally want to recreate the index.

## Retrieval and RAG Generation

### Semantic Retrieval

The retrieval component is implemented in `src/retrieval.py`.

When a user submits a query, the system:

1. Validates the query and retrieval parameters.
2. Converts the query into an embedding.
3. Performs a similarity search in ChromaDB.
4. Retrieves the top-K relevant chunks.
5. Returns text, metadata, and similarity-distance information.

This approach retrieves semantically relevant passages even when the query does not use the exact wording found in a document.

The retrieval layer handles common edge cases, including empty queries, invalid `top_k` values, empty collections, and requests exceeding the available number of results.

### LLM-Powered Generation

The RAG generation pipeline extends semantic retrieval by supplying retrieved passages to an external LLM API.

```text
User Question
      |
      v
Semantic Retrieval
      |
      v
Relevant Research Chunks
      |
      v
Context Construction
      |
      v
Grounded Prompt
      |
      v
LLM API
      |
      v
Generated Answer
```

The prompt instructs the model to prioritize the retrieved evidence and avoid presenting unsupported claims as facts.

Source metadata is retained throughout retrieval and generation to support source-aware responses.

The system also accounts for common external API failures, including missing credentials, request errors, timeouts, rate limits, and unexpected responses.

**Important:** Retrieval-grounded prompting can reduce unsupported answers, but it does not guarantee that hallucinations will never occur. Quantitative groundedness and answer-quality evaluation remains a separate engineering requirement.

### API Configuration

Configure the API key for your chosen LLM provider through environment variables. Never hard-code credentials or commit them to GitHub.

The exact variable name and provider configuration depend on the generation implementation.

## Benchmarking and Testing

### Retrieval Benchmark

Run the retrieval benchmark:

```bash
python -m scripts.benchmark_retrieval
```

The benchmark evaluates representative machine-learning queries covering areas such as:

- Deep learning
- Transformer models
- Computer vision
- Reinforcement learning
- Generative models
- Graph neural networks
- Attention mechanisms
- Model evaluation

The script measures retrieval latency and reports summary statistics such as average, median, minimum, and maximum latency.

These measurements establish a baseline for comparing retrieval configurations and future optimizations.

### Unit Tests

Run the test suite:

```bash
pytest -v
```

Tests cover the core pipeline components.

**Preprocessing**
- Unicode normalization
- HTML removal
- Whitespace cleanup
- Language detection and filtering
- End-to-end text cleaning

**Chunking**
- Empty and whitespace-only inputs
- Short and long documents
- Multiple chunk generation
- Chunk-size constraints
- Context overlap
- Invalid configuration values
- Invalid input types

**Retrieval**
- Query validation
- Retrieval edge cases
- Empty or unavailable collections

The test suite can be extended as additional API, evaluation, and generation components are implemented.

## Reproducibility and Data Management

The project uses a Python virtual environment and `requirements.txt` to manage dependencies.

Generated datasets and vector indexes are intentionally excluded from version control.

Typical pipeline reconstruction commands:

```bash
python -m scripts.preprocess_dataset
python -m scripts.create_chunks
python -m scripts.build_vector_store --reset
python -m scripts.benchmark_retrieval
```

Run dataset acquisition first if the required local data files are not available.

The generation component additionally requires a configured LLM API credential.

The `.gitignore` file should exclude:

```text
.venv/
.env
.env.*
data/raw/
data/processed/
chroma_db/
hf_cache/
__pycache__/
.pytest_cache/
```

## Current Status

The project is being developed incrementally toward a complete RAG knowledge extraction system.

| Component | Status |
|---|---|
| Environment setup and verification | Completed |
| Dataset acquisition | Completed |
| Text preprocessing | Completed |
| Recursive document chunking | Completed |
| Sentence Transformer embeddings | Completed |
| ChromaDB vector storage | Completed |
| Semantic retrieval | Completed |
| LLM integration and grounded generation | Completed |
| Retrieval latency benchmarking | Implemented |
| Preprocessing, chunking, and retrieval tests | Implemented |
| Topic modeling, sentiment analysis, and NER | Planned |
| FastAPI service layer | Planned |
| Formal retrieval and generation evaluation | Planned |
| End-to-end performance benchmarking | Planned |
| Final deployment and optimization | Planned |

*Status reflects the documented implementation and should be updated as additional features are completed and verified.*

## Engineering Principles

The system follows several engineering principles:

- **Modularity:** Data acquisition, preprocessing, chunking, embeddings, retrieval, and generation are separated into distinct components.
- **Reproducibility:** Dependencies and execution steps are documented.
- **Testability:** Core processing and retrieval functions are tested independently.
- **Data quality:** Raw, cleaned, and chunked datasets remain separate.
- **Deterministic identification:** Stable chunk IDs support consistent vector indexing.
- **Measurable performance:** Embedding and retrieval operations are benchmarked.
- **Grounded generation:** Retrieved evidence is supplied to the LLM as context.
- **Separation of concerns:** Retrieval and generation can be evaluated independently.
- **Data security:** Credentials and large generated artifacts are excluded from version control.

## Future Improvements

Planned extensions include:

- Topic modeling, sentiment analysis, and Named Entity Recognition (NER)
- Metadata-aware and hybrid keyword-semantic retrieval
- Retrieval reranking and query expansion
- FastAPI endpoints and structured error responses
- Precision@K and Recall@K retrieval evaluation
- Groundedness, answer relevance, and source-attribution evaluation
- End-to-end latency and throughput benchmarking
- Structured logging and observability
- Improved documentation and deployment readiness

These improvements will be prioritized according to their measurable impact on retrieval quality, reliability, and system performance.

## License

Developed for educational and research purposes.
