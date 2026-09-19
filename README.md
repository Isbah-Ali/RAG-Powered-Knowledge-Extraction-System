# RAG-Powered Knowledge Extraction System

A production-oriented **Retrieval-Augmented Generation (RAG)** knowledge extraction system developed as part of the **Parallax Labs AI/ML Engineer Internship**.

The system is designed to ingest thousands of real-world research documents, preprocess and normalize their content, transform documents into semantic vector representations, perform similarity-based retrieval, and generate grounded responses using Large Language Models (LLMs).

The project is being developed incrementally across a six-week engineering roadmap, with emphasis on **data quality, modular architecture, retrieval performance, evaluation, reproducibility, and production-ready API design**.

---

## Project Status

| Phase  | Focus                                               | Status      |
| ------ | --------------------------------------------------- | ----------- |
| Week 1 | Environment, data ingestion & preprocessing         | ✅ Completed |
| Week 2 | Chunking, embeddings, ChromaDB & semantic retrieval | 🔄 Upcoming |
| Week 3 | LLM integration & grounded generation               | ⏳ Planned   |
| Week 4 | Topic modeling, sentiment & NER                     | ⏳ Planned   |
| Week 5 | FastAPI, evaluation & benchmarking                  | ⏳ Planned   |
| Week 6 | Documentation, architecture & final optimization    | ⏳ Planned   |

---

# 1. Project Overview

The objective of this project is to build an end-to-end knowledge extraction system capable of answering questions over a large collection of real-world documents.

The final architecture will follow the general pipeline:

```text
                    ┌─────────────────────┐
                    │   Real-World Data   │
                    │ ArXiv / Reddit /    │
                    │ Wikipedia Sources   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Ingestion    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Text Preprocessing  │
                    │ • Unicode           │
                    │ • HTML Removal      │
                    │ • Whitespace        │
                    │ • Language Filter   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Document Chunking   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Embeddings       │
                    │ Sentence Transformers│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     ChromaDB        │
                    │  Vector Database    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retrieval  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM Generation    │
                    │ DeepSeek / OpenRouter│
                    │       / Groq        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Grounded Response   │
                    │ + Source Context    │
                    └─────────────────────┘
```

---

# 2. Week 1 — Environment, Data Ingestion & Preprocessing

Week 1 establishes the data and engineering foundation required for the remainder of the project.

### Completed objectives

* Initialized an isolated Python virtual environment
* Installed and verified required ML/NLP dependencies
* Configured dataset and package caches outside the system drive
* Created a modular project structure
* Configured Git version control
* Added a `.gitignore` for datasets, environments, caches and secrets
* Acquired a 5,500-document subset of ML research papers
* Stored raw data in JSONL format
* Implemented reusable text preprocessing functions
* Applied Unicode normalization
* Removed HTML markup
* Normalized whitespace
* Applied English-language filtering
* Generated a validated processed corpus
* Implemented unit tests using `pytest`
* Generated dataset statistics
* Locked the Python environment using `requirements.txt`

---

# 3. Dataset

The initial corpus is based on the **ML-ArXiv-Papers** dataset available through Hugging Face.

The dataset contains machine-learning-related research papers originating from ArXiv.

For Week 1, a controlled subset of **5,500 documents** is acquired rather than downloading the complete dataset.

Each document is represented using structured JSONL records containing fields such as:

```json
{
    "id": "arxiv_0",
    "title": "Example Research Paper",
    "text": "Title and abstract content...",
    "source": "ArXiv"
}
```

### Dataset strategy

The project intentionally keeps raw and processed datasets outside version control.

```text
External Dataset
      │
      ▼
Streaming Acquisition
      │
      ▼
5,500 Raw Documents
      │
      ▼
Local JSONL Storage
      │
      ▼
Preprocessing
      │
      ▼
Validated Clean Corpus
```

Large datasets are excluded from GitHub through `.gitignore` to keep the repository lightweight and reproducible.

---

# 4. Data Preprocessing Pipeline

The preprocessing layer is implemented as a modular Python component rather than embedding cleaning logic directly inside the ingestion script.

### Processing stages

```text
Raw Document
     │
     ▼
Unicode Normalization
     │
     ▼
HTML Removal
     │
     ▼
Whitespace Normalization
     │
     ▼
Minimum Content Validation
     │
     ▼
English Language Detection
     │
     ▼
Clean Document
```

The core preprocessing module is located at:

```text
src/preprocessing.py
```

Implemented operations include:

### Unicode normalization

Normalizes Unicode representations using NFKC normalization.

### HTML stripping

Removes HTML markup using BeautifulSoup while preserving readable text.

### Whitespace normalization

Converts repeated spaces, tabs and newlines into normalized whitespace.

### Language filtering

Uses language detection to retain English-language documents suitable for the initial corpus.

### Modular design

The cleaning functions are independently testable:

```python
normalize_unicode()
strip_html()
clean_whitespace()
is_english()
clean_text()
```

This design allows individual preprocessing stages to be modified or extended without restructuring the complete ingestion pipeline.

---

# 5. Project Structure

```text
parallax-rag-knowledge-system/
│
├── data/
│   ├── raw/
│   │   └── arxiv_5500.jsonl
│   │
│   └── processed/
│       └── clean_corpus.jsonl
│
├── scripts/
│   ├── __init__.py
│   ├── acquire_dataset.py
│   ├── dataset_stats.py
│   ├── preprocess_dataset.py
│   └── verify_env.py
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py
│
├── tests/
│   └── test_preprocessing.py
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

> **Note:** `data/raw/`, `data/processed/`, `.venv/`, Hugging Face caches and other generated artifacts are intentionally excluded from version control.

---

# 6. Environment & Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* PyArrow

### NLP / Machine Learning

* Sentence Transformers
* spaCy
* langdetect

### Data Acquisition

* Hugging Face Datasets

### Text Processing

* BeautifulSoup4
* Unicode normalization
* Regular expressions

### Vector Database

* ChromaDB

### Testing

* pytest

### Development

* Visual Studio Code
* Git
* GitHub
* Python virtual environment

### Planned LLM / RAG Components

* DeepSeek
* OpenRouter
* Groq
* Sentence Transformers
* ChromaDB
* FastAPI

---

# 7. Environment Verification

The project includes an environment verification script:

```text
scripts/verify_env.py
```

It validates the availability of the core Python dependencies and checks the available PyTorch/CUDA environment.

Run:

```bash
python scripts/verify_env.py
```

The verification process checks packages including:

```text
pandas
datasets
spacy
pytest
sentence-transformers
chromadb
torch
```

GPU availability is detected dynamically rather than assumed.

---

# 8. Dataset Acquisition

The dataset acquisition process is implemented in:

```text
scripts/acquire_dataset.py
```

The script uses Hugging Face streaming functionality to avoid downloading the complete source dataset unnecessarily.

Run:

```bash
python scripts/acquire_dataset.py
```

The resulting raw corpus is written to:

```text
data/raw/arxiv_5500.jsonl
```

---

# 9. Data Preprocessing

Run the preprocessing pipeline using:

```bash
python -m scripts.preprocess_dataset
```

The cleaned corpus is generated at:

```text
data/processed/clean_corpus.jsonl
```

The preprocessing stage reports:

* Total documents processed
* Documents retained
* Documents removed

---

# 10. Dataset Statistics

Basic corpus statistics can be generated using:

```bash
python scripts/dataset_stats.py
```

The statistics include:

* Number of documents
* Average document length
* Minimum document length
* Maximum document length

These measurements provide an initial validation of corpus quality before downstream embedding and retrieval stages.

---

# 11. Testing

Unit tests are implemented using `pytest`.

Run the complete test suite:

```bash
pytest -v
```

The current Week 1 tests cover:

* Unicode normalization
* HTML removal
* Whitespace normalization
* English-language detection
* Non-English filtering
* End-to-end text cleaning

The preprocessing pipeline is designed so that future transformations can be added with corresponding tests.

---

# 12. Reproducibility

The project uses a Python virtual environment and a locked dependency file:

```text
requirements.txt
```

Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The repository also includes:

```text
.gitignore
```

to prevent large datasets, generated files, virtual environments, cache directories and secrets from being committed.

---

# 13. Engineering Principles

The project is being developed with the following engineering principles:

### Modular Architecture

Data acquisition, preprocessing, testing and future retrieval/generation components are separated into independent modules.

### Reproducibility

Environment dependencies and execution procedures are documented.

### Testability

Core preprocessing functions are independently unit tested.

### Data Quality

Raw and processed corpora are separated, allowing the preprocessing pipeline to be evaluated independently.

### Scalability

The ingestion and processing architecture is designed to operate on thousands of documents and can later be extended to larger corpora.

### Version Control

Source code and configuration are version-controlled while large generated datasets remain outside Git.

---

# 14. Planned RAG Architecture

The upcoming stages will extend the Week 1 foundation into a complete RAG system.

```text
                 Clean Corpus
                      │
                      ▼
               Document Chunking
                      │
                      ▼
              Sentence Embeddings
                      │
                      ▼
                  ChromaDB
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       User Query        Metadata Filters
             │                 │
             └────────┬────────┘
                      ▼
             Semantic Retrieval
                      │
                      ▼
              Retrieved Context
                      │
                      ▼
                Prompt Builder
                      │
                      ▼
                     LLM
                      │
                      ▼
             Grounded Response
                      │
                      ▼
             Sources / Evidence
```

---

# 15. Evaluation Strategy

The final system will be evaluated beyond simple response generation.

Planned evaluation dimensions include:

### Retrieval

* Precision@K
* Recall@K
* Retrieval latency
* Semantic relevance

### Generation

* Groundedness
* Hallucination rate
* Answer relevance
* Off-topic query handling

### System Performance

* End-to-end latency
* API response time
* Retrieval performance
* Error handling

### NLP Analysis

* Topic modeling quality
* Sentiment classification
* Named Entity Recognition

---

# 16. Six-Week Development Roadmap

## Week 1 — Foundation

**Environment, ingestion and preprocessing**

* Environment setup
* Dataset acquisition
* Data cleaning
* Language filtering
* Unit testing
* Dataset validation

**Status: Completed**

---

## Week 2 — Retrieval Infrastructure

**Chunking, embeddings and vector search**

Planned components:

* Document chunking
* Sentence Transformer embeddings
* ChromaDB
* Vector indexing
* Semantic similarity search
* Retrieval latency measurement

---

## Week 3 — RAG Generation

**LLM integration and grounded responses**

Planned components:

* LLM API integration
* Prompt engineering
* Context injection
* Source-aware responses
* Error handling
* Hallucination mitigation
* Off-topic query handling
* End-to-end latency measurement

---

## Week 4 — NLP Intelligence

**Document-level NLP analysis**

Planned components:

* Topic modeling
* Sentiment analysis
* Named Entity Recognition
* Metadata enrichment
* Filtered retrieval
* Manual validation

---

## Week 5 — API & Evaluation

**Production-oriented service layer**

Planned components:

* FastAPI endpoint
* Structured logging
* HTTP error handling
* Retrieval evaluation
* Generation evaluation
* Precision@K
* Recall@K
* Latency benchmarking
* API unit tests

---

## Week 6 — Production Readiness

**Documentation, reproducibility and final optimization**

Planned components:

* Architecture documentation
* Type hints
* Docstrings
* Benchmarking
* Reproducible setup
* Final system evaluation
* Project cleanup
* Final technical documentation

---

# 17. Current Deliverables

### Week 1 Deliverables

* [x] Python virtual environment
* [x] Dependency installation
* [x] Environment verification
* [x] Git repository initialization
* [x] `.gitignore`
* [x] Hugging Face dataset acquisition
* [x] 5,500-document raw corpus
* [x] Modular preprocessing pipeline
* [x] Clean validated corpus
* [x] Language filtering
* [x] Unit tests
* [x] Dataset statistics
* [x] `requirements.txt`
* [x] Project documentation

---

# 18. Security & Data Management

No API keys or secrets are stored in the repository.

Environment files are excluded using:

```text
.env
.env.*
```

Large datasets and generated artifacts are also excluded from Git version control.

The project therefore separates:

```text
Source Code
     │
     ├── GitHub
     │
     ▼
Version Controlled

Large Data
     │
     ├── Local Storage
     │
     ▼
Git-Ignored
```

---

# 19. Future Extensions

Potential future extensions include:

* Multi-source document ingestion
* Hybrid keyword + semantic retrieval
* Metadata-aware retrieval
* Reranking
* Query expansion
* Conversational memory
* Retrieval caching
* Streaming LLM responses
* Advanced RAG evaluation
* Observability and structured logging
* Production deployment
* REST API access
* Interactive user interface

---

# 20. Author

**AI/ML Engineer — Parallax Labs Internship**

This repository documents the engineering progression of a complete RAG-powered knowledge extraction system from raw document ingestion through retrieval, generation, evaluation and API deployment.

---

## License

This project is developed for educational and internship purposes as part of the Parallax Labs AI/ML Engineer Internship.
