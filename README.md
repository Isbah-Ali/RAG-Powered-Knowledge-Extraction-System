# RAG Powered Knowledge Extraction System
A production-oriented **Retrieval-Augmented Generation (RAG) knowledge extraction system** developed as part of the **Parallax Labs AI/ML Engineer Internship**.

The system is designed to ingest thousands of real-world research documents, preprocess and normalize their content, intelligently chunk documents, transform text into semantic vector representations, index those representations in a vector database, perform similarity-based retrieval, and generate grounded responses using Large Language Models (LLMs).

The project is being developed incrementally across a six-week engineering roadmap, with emphasis on **data quality, modular architecture, retrieval quality, measurable performance, evaluation, reproducibility, maintainability, and production-ready API design**.

Rather than treating each weekly requirement as an isolated assignment, the implementation is designed so that every stage becomes a foundation for the final end-to-end RAG system.

---

## Project Status

| Phase  | Focus                                               | Status      |
| ------ | --------------------------------------------------- | ----------- |
| Week 1 | Environment, data ingestion & preprocessing         | ✅ Completed |
| Week 2 | Chunking, embeddings, ChromaDB & semantic retrieval | ✅ Completed |
| Week 3 | LLM integration & grounded generation               | ✅ Completed |
| Week 4 | Topic modeling, sentiment & NER                     | ⏳ Planned   |
| Week 5 | FastAPI, evaluation & benchmarking                  | ⏳ Planned   |
| Week 6 | Documentation, architecture & final optimization    | ⏳ Planned   |

---

# 1. Project Overview

The objective of this project is to build an end-to-end knowledge extraction system capable of retrieving and answering questions over a large collection of real-world research documents.

The system is constructed as a modular retrieval and generation pipeline:

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
                    │ Recursive Splitting │
                    │ + Context Overlap   │
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
                    │  Persistent Vector  │
                    │       Store         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retrieval  │
                    │      Top-K Search   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Prompt Builder    │
                    │ Retrieved Context   │
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

Week 1 established the data and engineering foundation required for downstream retrieval and generation.

### Completed Objectives

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

For Week 1, a controlled subset of **5,500 documents** is acquired rather than downloading the complete source dataset.

Each document is represented using structured JSONL records containing fields such as:

```json
{
  "id": "arxiv_0",
  "title": "Example Research Paper",
  "text": "Title and abstract content...",
  "source": "ArXiv"
}
```

### Dataset Strategy

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
      │
      ▼
Document Chunking
```

Large datasets are excluded from GitHub through `.gitignore` to keep the repository lightweight and reproducible.

---

# 4. Data Preprocessing Pipeline

The preprocessing layer is implemented as a modular Python component rather than embedding cleaning logic directly inside the ingestion script.

### Processing Stages

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

### Unicode Normalization

Normalizes Unicode representations using NFKC normalization.

### HTML Stripping

Removes HTML markup using BeautifulSoup while preserving readable text.

### Whitespace Normalization

Converts repeated spaces, tabs and newlines into normalized whitespace.

### Language Filtering

Uses language detection to retain English-language documents suitable for the initial corpus.

### Modular Design

The cleaning functions are independently testable:

```text
normalize_unicode()
strip_html()
clean_whitespace()
is_english()
clean_text()
```

This design allows individual preprocessing stages to be modified or extended without restructuring the complete ingestion pipeline.

---

# 5. Week 2 — Document Chunking, Embeddings & Semantic Retrieval

Week 2 extends the cleaned corpus into a functional vector retrieval system.

The objective is to transform long documents into retrieval-friendly semantic units, encode those units as vectors, persist them in ChromaDB, and retrieve relevant passages based on semantic similarity.

The Week 2 architecture is:

```text
Clean Corpus
     │
     ▼
Recursive Chunking
     │
     ▼
Chunk Metadata
     │
     ▼
Sentence Transformer
     │
     ▼
Dense Embeddings
     │
     ▼
ChromaDB
     │
     ▼
Query Embedding
     │
     ▼
Semantic Similarity Search
     │
     ▼
Top-K Retrieved Chunks
```

---

## 5.1 Document Chunking

Document chunking is implemented in:

```text
src/chunking.py
```

The system uses a recursive splitting strategy rather than simply cutting documents at fixed character positions.

The splitter prioritizes increasingly fine-grained boundaries:

```text
Paragraph
    ↓
Line
    ↓
Sentence
    ↓
Word
    ↓
Character
```

This approach attempts to preserve semantic coherence while ensuring chunks remain within the configured size constraints.

### Current Configuration

```text
Chunk size:    500 characters
Chunk overlap: 75 characters
```

The overlap preserves contextual information between neighboring chunks and reduces the probability that important information is lost at chunk boundaries.

### Chunk Metadata

Each generated chunk contains structured metadata:

```json
{
  "chunk_id": "arxiv_0_chunk_0",
  "document_id": "arxiv_0",
  "chunk_index": 0,
  "title": "Example Research Paper",
  "source": "ArXiv",
  "text": "..."
}
```

Deterministic chunk IDs make the indexing pipeline reproducible and allow ChromaDB records to be safely updated using `upsert`.

### Chunk Output

Generated chunks are stored locally in:

```text
data/processed/chunks.jsonl
```

The generated dataset remains excluded from version control.

---

## 5.2 Chunking Tests

Chunking behavior is independently tested using:

```text
tests/test_chunking.py
```

The test suite covers:

* Empty input
* Whitespace-only input
* Short documents
* Long documents
* Multiple chunk generation
* Maximum chunk size
* Empty chunk prevention
* Context overlap
* Zero-overlap configuration
* Invalid chunk sizes
* Invalid overlap values
* Non-string input
* Default configuration validation

This ensures that chunking behavior is validated independently before embeddings are generated.

---

## 5.3 Semantic Embeddings

Embedding functionality is implemented in:

```text
src/embeddings.py
```

The project uses **Sentence Transformers** to convert text chunks into dense semantic vectors.

### Current Embedding Model

```text
all-MiniLM-L6-v2
```

The model provides a lightweight embedding pipeline suitable for local development while maintaining practical semantic retrieval capability.

The embedding wrapper records:

* Model loading time
* Embedding generation time
* Number of processed texts
* Embedding dimensionality
* Embedding throughput

Embeddings are normalized before being stored, allowing cosine-based similarity comparisons to operate consistently.

---

## 5.4 Embedding Performance

Embedding performance is measured rather than treated as an opaque preprocessing step.

The vector-store construction process records:

```text
Embedding model
Embedding dimension
Total embedding time
Average embedding time per chunk
Overall indexing throughput
```

This provides a measurable baseline that can later be used when experimenting with:

* Different embedding models
* Different batch sizes
* CPU vs GPU execution
* Larger corpora
* Retrieval optimizations

---

# 6. ChromaDB Vector Store

ChromaDB is used as the persistent vector database for the project.

The vector-store construction pipeline is implemented in:

```text
scripts/build_vector_store.py
```

The persistent database is stored locally at:

```text
chroma_db/
```

and excluded from Git version control.

### Collection

The primary collection is:

```text
parallax_rag_chunks
```

Each indexed record contains:

```text
Chunk ID
Document text
Dense embedding
Document metadata
```

Metadata includes:

```text
document_id
chunk_index
title
source
```

The embedding model is explicitly controlled by the application rather than allowing the vector database to silently select an embedding model.

This makes the retrieval pipeline more reproducible and makes future embedding-model comparisons possible.

---

## 6.1 Vector Store Construction

The vector database can be rebuilt using:

```bash
python -m scripts.build_vector_store --reset
```

The `--reset` option allows the collection to be safely rebuilt from the current chunk dataset.

The indexing pipeline processes chunks in batches to avoid unnecessarily loading the entire embedding workload into memory at once.

---

# 7. Semantic Retrieval

Semantic retrieval is implemented in:

```text
src/retrieval.py
```

The retrieval pipeline performs:

```text
User Query
     │
     ▼
Query Embedding
     │
     ▼
ChromaDB Similarity Search
     │
     ▼
Top-K Results
     │
     ▼
Text + Metadata + Distance
```

The retriever returns the retrieved text together with metadata and similarity distance.

This preserves the connection between a retrieved passage and its originating document.

Example metadata returned with a result:

```text
Document ID
Chunk index
Paper title
Source
Similarity distance
```

This metadata supports source-aware RAG responses and citation generation.

---

## 7.1 Retrieval Edge Cases

The retrieval layer includes defensive handling for common vector-search edge cases.

Handled conditions include:

* Empty queries
* Whitespace-only queries
* Invalid query types
* Invalid `top_k` values
* Empty ChromaDB collections
* `top_k` larger than the available collection
* Missing ChromaDB collections
* Duplicate chunk IDs during indexing

The indexing process uses deterministic chunk IDs and `upsert`, allowing the vector store to be rebuilt without unintentionally creating duplicate records.

---

# 8. Retrieval Performance Benchmarking

Retrieval performance is measured using:

```text
scripts/benchmark_retrieval.py
```

The benchmark evaluates multiple representative machine-learning queries rather than relying on a single example.

Current benchmark categories include:

* Deep learning
* Transformer models
* Reinforcement learning
* Computer vision
* Optimization
* Generative models
* Graph neural networks
* Model evaluation
* Attention mechanisms
* Unsupervised learning

For each query, the benchmark records:

```text
Query
Retrieval latency
Number of retrieved results
```

The benchmark summarizes:

* Average latency
* Median latency
* Minimum latency
* Maximum latency
* Number of queries tested

Latency is measured using high-resolution performance timers.

This provides a reproducible baseline for future retrieval optimizations.

---

# 9. Week 3 — LLM Integration & Grounded Generation

Week 3 transforms the retrieval infrastructure into a functional **Retrieval-Augmented Generation pipeline**.

The goal of this stage was to connect the retrieved evidence from ChromaDB to an LLM while ensuring that generated responses remain grounded in the retrieved context.

The resulting pipeline is:

```text
User Query
     │
     ▼
Query Embedding
     │
     ▼
ChromaDB Retrieval
     │
     ▼
Top-K Relevant Chunks
     │
     ▼
Context Formatting
     │
     ▼
Grounded Prompt
     │
     ▼
LLM API
     │
     ▼
Generated Answer
     │
     ▼
Source-Aware Response
```

---

## 9.1 LLM Integration

The generation layer integrates an external LLM API with the existing retrieval pipeline.

The LLM is not given unrestricted access to the complete corpus.

Instead, the system follows the RAG pattern:

```text
Large Corpus
     ↓
Semantic Retrieval
     ↓
Relevant Evidence
     ↓
LLM Context
     ↓
Answer
```

This reduces the amount of irrelevant information passed to the model and provides a grounding mechanism for generated responses.

The generation layer is intentionally separated from the retrieval implementation so that retrieval and generation can be tested independently.

---

## 9.2 Prompt Construction

Retrieved chunks are converted into structured context before being passed to the LLM.

Conceptually:

```text
System Instructions
        +
Retrieved Evidence
        +
User Question
        ↓
Grounded Prompt
        ↓
LLM
```

The prompt instructs the model to prioritize the supplied evidence and avoid presenting unsupported information as fact.

The generation layer therefore acts as a controlled interface between:

```text
Retriever
    ↓
Context
    ↓
LLM
```

rather than directly sending the user query to the model.

---

## 9.3 Grounded Generation

The system is designed to answer questions using the retrieved research context.

For supported queries:

```text
Question
   ↓
Relevant Papers Retrieved
   ↓
Evidence Passed to LLM
   ↓
Answer Generated From Evidence
```

If the retrieved context does not contain sufficient information, the system is designed to avoid confidently fabricating an answer.

This distinction is important because an LLM-only chatbot and a RAG system have different reliability characteristics.

The purpose of the retrieval layer is to provide an evidence boundary around generation.

---

## 9.4 Source-Aware Responses

Retrieved chunks retain their document metadata throughout the pipeline.

The generation stage can therefore associate generated answers with their underlying evidence:

```text
Answer
  │
  ├── Source Document
  ├── Paper Title
  ├── Chunk Information
  └── Retrieved Context
```

Maintaining this metadata makes the system more transparent and provides a foundation for explicit source citations and evidence presentation in later versions.

---

## 9.5 Error Handling

The generation layer includes defensive handling for failures that can occur when communicating with an external LLM service.

The pipeline accounts for conditions such as:

* Missing API credentials
* Invalid API configuration
* Empty queries
* Empty retrieved context
* API request failures
* Timeout conditions
* Rate limiting
* Unexpected LLM responses
* Generation failures

The goal is to prevent external API failures from silently breaking the complete RAG pipeline.

---

## 9.6 Hallucination Mitigation

The project does not treat prompt engineering as a guarantee that hallucinations cannot occur.

Instead, hallucination mitigation is approached through multiple layers:

```text
Semantic Retrieval
        ↓
Relevant Evidence
        ↓
Explicit Grounding Instructions
        ↓
Controlled Context
        ↓
LLM Generation
```

The retrieved context acts as the primary evidence source, while the prompt instructs the model to avoid unsupported claims.

Further quantitative hallucination and groundedness evaluation is planned for Week 5.

---

## 9.7 Off-Topic Query Handling

The system distinguishes between queries that can reasonably be answered from the indexed corpus and questions for which the retrieved evidence is insufficient.

The generation layer is designed to avoid treating arbitrary unrelated questions as if they were supported by the research corpus.

This establishes an important boundary:

```text
Relevant Query
      ↓
Retrieve Evidence
      ↓
Generate Grounded Answer


Unsupported / Off-Topic Query
      ↓
Insufficient Evidence
      ↓
Controlled Response
```

This behavior will be further evaluated and improved during the formal evaluation phase.

---

## 9.8 End-to-End RAG Flow

With Week 3 completed, the system now operates conceptually as:

```text
                 ┌──────────────────┐
                 │    User Query    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Query Embedding  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    ChromaDB      │
                 │ Semantic Search  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Retrieved Chunks │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Context Builder  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  Grounded Prompt │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │       LLM        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Grounded Answer  │
                 │ + Source Context │
                 └──────────────────┘
```

This represents the first complete end-to-end RAG generation path in the project.

---

# 10. Project Structure

The project structure now reflects the transition from a preprocessing pipeline into an end-to-end RAG system.

```text
RAG-Powered-Knowledge-Extraction-System/
│
├── data/
│   ├── raw/
│   │   └── arxiv_5500.jsonl
│   │
│   └── processed/
│       ├── clean_corpus.jsonl
│       └── chunks.jsonl
│
├── chroma_db/
│   └── [local persistent vector database]
│
├── scripts/
│   ├── __init__.py
│   ├── acquire_dataset.py
│   ├── benchmark_retrieval.py
│   ├── build_vector_store.py
│   ├── create_chunks.py
│   ├── dataset_stats.py
│   ├── preprocess_dataset.py
│   └── verify_env.py
│
├── src/
│   ├── __init__.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── preprocessing.py
│   └── retrieval.py
│
├── tests/
│   ├── test_chunking.py
│   ├── test_preprocessing.py
│   └── test_retrieval.py
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

> **Note:** The exact generation-layer filenames may evolve as the project progresses. Generated datasets, vector stores, caches, virtual environments and secrets remain intentionally excluded from version control.

---

# 11. Environment & Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* PyArrow

### NLP / Machine Learning

* Sentence Transformers
* spaCy
* langdetect
* PyTorch

### Data Acquisition

* Hugging Face Datasets

### Text Processing

* BeautifulSoup4
* Unicode normalization
* Regular expressions

### Vector Database

* ChromaDB

### LLM / Generative AI

* DeepSeek
* OpenRouter
* Groq
* Prompt construction
* Grounded generation
* Context-aware generation

### Testing

* pytest

### Development

* Visual Studio Code
* Git
* GitHub
* Python virtual environment

### Current RAG Components

* Recursive document chunking
* Configurable chunk size and overlap
* Deterministic chunk identifiers
* Sentence Transformer embeddings
* `all-MiniLM-L6-v2`
* ChromaDB persistent vector storage
* Batched vector indexing
* Semantic top-K retrieval
* Retrieval edge-case handling
* Retrieval latency benchmarking
* LLM integration
* Retrieved-context prompt construction
* Grounded generation
* Source-aware responses
* External API error handling
* Off-topic query handling

### Planned Components

* Topic modeling
* Sentiment analysis
* Named Entity Recognition
* FastAPI
* Formal retrieval evaluation
* Generation evaluation
* End-to-end benchmarking
* Production-oriented API layer

---

# 12. Environment Verification

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

# 13. Dataset Acquisition

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

# 14. Data Preprocessing

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

# 15. Document Chunking

Generate retrieval-ready chunks using:

```bash
python -m scripts.create_chunks
```

The output is written to:

```text
data/processed/chunks.jsonl
```

The chunking pipeline reports:

* Documents processed
* Number of chunks created
* Average chunk size
* Minimum chunk size
* Maximum chunk size
* Processing time
* Average chunks per document

---

# 16. Vector Store Construction

Build the ChromaDB vector store using:

```bash
python -m scripts.build_vector_store --reset
```

The process:

1. Loads the cleaned chunks
2. Initializes the Sentence Transformer model
3. Generates embeddings in batches
4. Records embedding performance
5. Stores embeddings and metadata in ChromaDB
6. Reports final collection size

The resulting vector database is stored locally under:

```text
chroma_db/
```

---

# 17. Semantic Search

Semantic retrieval can be accessed through:

```text
src/retrieval.py
```

A typical retrieval operation performs:

```text
Query
  ↓
Query embedding
  ↓
ChromaDB similarity search
  ↓
Top-K chunks
  ↓
Text + metadata + distance
```

This retrieval layer is intentionally separated from the LLM generation layer so that retrieval quality can be evaluated independently.

---

# 18. RAG Generation

The Week 3 generation layer extends semantic retrieval into an end-to-end RAG workflow.

The general execution path is:

```text
User Query
      ↓
Semantic Retrieval
      ↓
Relevant Context
      ↓
Prompt Construction
      ↓
LLM API
      ↓
Grounded Response
```

The generation process uses retrieved evidence rather than relying solely on the model's internal knowledge.

This separation allows the system to maintain independent retrieval and generation components while still exposing a unified RAG workflow.

---

# 19. Retrieval Benchmark

Run the retrieval benchmark using:

```bash
python -m scripts.benchmark_retrieval
```

The benchmark evaluates multiple representative queries and reports:

```text
Average latency
Median latency
Minimum latency
Maximum latency
Number of queries
```

Separating retrieval benchmarking from LLM generation allows the project to distinguish:

```text
Retrieval Performance
        +
LLM Generation Performance
        =
End-to-End RAG Performance
```

Future benchmarking will extend this into full end-to-end measurements.

---

# 20. Testing

Unit tests are implemented using `pytest`.

Run the complete test suite:

```bash
pytest -v
```

The test suite currently covers preprocessing and chunking behavior, including:

### Preprocessing

* Unicode normalization
* HTML removal
* Whitespace normalization
* English-language detection
* Non-English filtering
* End-to-end text cleaning

### Chunking

* Empty input
* Whitespace-only input
* Short documents
* Long documents
* Multiple chunk generation
* Maximum chunk size
* Empty chunk prevention
* Context overlap
* Zero-overlap configuration
* Invalid chunk sizes
* Invalid overlap values
* Invalid input types

### Retrieval

Input validation and retrieval edge cases are tested independently where possible.

The testing architecture is designed to expand as the RAG system becomes more complex.

---

# 21. Reproducibility

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

The retrieval pipeline can be reconstructed in sequence:

```bash
python -m scripts.preprocess_dataset

python -m scripts.create_chunks

python -m scripts.build_vector_store --reset

python -m scripts.benchmark_retrieval
```

The RAG generation stage additionally requires a configured LLM API credential.

This allows the complete preprocessing-to-retrieval pipeline to be reproduced from the source dataset and project code while keeping credentials outside version control.

---

# 22. Engineering Principles

The project is being developed with the following engineering principles:

### Modular Architecture

Data acquisition, preprocessing, chunking, embedding, vector storage, retrieval and generation are separated into independent components.

### Reproducibility

Environment dependencies, model configuration and execution procedures are documented.

### Testability

Core transformation and retrieval components are independently tested.

### Data Quality

Raw, processed and chunked representations are separated, allowing each stage of the pipeline to be inspected independently.

### Deterministic Identification

Documents and chunks use stable identifiers, allowing vector records to be updated safely.

### Measurable Performance

Embedding generation and retrieval latency are measured rather than relying only on qualitative demonstrations.

### Grounded Generation

LLM responses are generated using retrieved evidence rather than treating the LLM as an unrestricted knowledge source.

### Separation of Concerns

Retrieval is implemented independently from LLM generation, allowing retrieval quality and generation quality to be evaluated separately.

### Scalability

Batch-based processing and persistent vector storage allow the system to operate on thousands of documents and provide a foundation for larger corpora.

### Version Control

Source code and configuration are version-controlled while large generated datasets and vector stores remain outside Git.

---

# 23. Current RAG Architecture

With Weeks 1–3 completed, the system now follows this architecture:

```text
                 ┌───────────────────┐
                 │   Source Corpus   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   Preprocessing   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │     Chunking      │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    Embeddings     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │     ChromaDB      │
                 └─────────┬─────────┘
                           │
                           │
                 ┌─────────▼─────────┐
                 │ Semantic Retrieval│
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Retrieved Context │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  Prompt Builder   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │       LLM         │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Grounded Response │
                 │  + Source Context │
                 └───────────────────┘
```

The architecture deliberately separates:

```text
Retrieval Layer
      │
      ├── Chunking
      ├── Embeddings
      ├── Vector Storage
      └── Semantic Search
```

from:

```text
Generation Layer
      │
      ├── Context Construction
      ├── Prompt Construction
      ├── LLM Integration
      ├── Grounding
      └── Response Handling
```

This makes the system easier to evaluate, debug and optimize.

---

# 24. Evaluation Strategy

The final system will be evaluated beyond simple response generation.

Evaluation will be divided into separate retrieval, generation and system-level measurements.

### Retrieval

Planned measurements include:

* Precision@K
* Recall@K
* Semantic relevance
* Retrieval latency
* Top-K retrieval consistency
* Chunking configuration comparison

### Generation

Planned measurements include:

* Groundedness
* Hallucination rate
* Answer relevance
* Context utilization
* Source attribution
* Off-topic query handling

### System Performance

Planned measurements include:

* Embedding throughput
* Retrieval latency
* LLM generation latency
* End-to-end latency
* API response time
* Error handling

### NLP Analysis

Planned measurements include:

* Topic modeling quality
* Sentiment classification
* Named Entity Recognition

The Week 3 implementation establishes the generation pipeline required for these measurements. Formal quantitative generation and end-to-end evaluation will be introduced during the evaluation phase.

---

# 25. Six-Week Development Roadmap

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

Implemented components:

* Recursive document chunking
* Configurable chunk size and overlap
* Deterministic chunk identifiers
* Chunk metadata preservation
* Sentence Transformer embeddings
* Embedding performance measurement
* ChromaDB persistent vector storage
* Batched vector indexing
* Semantic top-K retrieval
* Retrieval edge-case handling
* Retrieval latency benchmarking
* Chunking unit tests

**Status: Completed**

---

## Week 3 — RAG Generation

**LLM integration and grounded responses**

Implemented components:

* LLM API integration
* Prompt construction
* Retrieved-context injection
* Grounded generation
* Source-aware response handling
* API error handling
* Timeout and rate-limit handling
* Hallucination mitigation
* Off-topic query handling
* End-to-end RAG flow

**Status: Completed**

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

**Status: Planned**

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
* Groundedness evaluation
* Latency benchmarking
* API unit tests

**Status: Planned**

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

**Status: Planned**

---

# 26. Security & Data Management

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


Large Data / Generated Artifacts
     │
     ├── Local Storage
     │
     ▼
Git-Ignored
```

This includes:

```text
data/raw/
data/processed/
chroma_db/
.venv/
hf_cache/
```

LLM API credentials are supplied through environment variables and are never hard-coded into the application.

---

# 27. Future Extensions

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

These components will only be introduced where they provide measurable improvements to retrieval quality, system performance or user experience.

---

# 28. Portfolio & Engineering Direction

The project is intentionally being developed beyond a minimal internship implementation.

The long-term objective is to demonstrate the complete engineering lifecycle of a modern RAG system:

```text
Data Engineering
       ↓
Text Processing
       ↓
Information Retrieval
       ↓
Embedding Infrastructure
       ↓
Vector Database
       ↓
Semantic Search
       ↓
LLM Generation
       ↓
RAG Grounding
       ↓
Retrieval & Generation Evaluation
       ↓
API Engineering
       ↓
Performance Optimization
       ↓
Production-Ready System
```

The emphasis is on **measurable engineering decisions rather than simply adding technologies**.

Each major component should have:

* A clear purpose
* A modular implementation
* Tests where appropriate
* Measurable behavior
* Documented design decisions
* A reproducible execution path

---

# 29. Author

**AI/ML Engineer — Parallax Labs Internship**

This repository documents the engineering progression of a complete RAG-powered knowledge extraction system from raw document ingestion through semantic retrieval, grounded LLM generation, evaluation and API deployment.

---

## License

This project is developed for educational and internship purposes as part of the Parallax Labs AI/ML Engineer Internship.
