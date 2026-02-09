
## Project Structure

```text
TSD-RAG/
├── run.py                  # Main entry point for the pipeline
├── requirements.txt        # Python dependencies
├── readme.md               # Project documentation
└── src/
    ├── tsdrag.py           # Core TSD-RAG algorithm 
    ├── config.py           # Global configuration management
    ├── embedding_store.py  # Vectorized storage engine
    ├── ner.py              # Entity Recognition module (Spacy wrapper)
    ├── evaluate.py         # Automated evaluation tools
    └── utils.py            # Utilities and LLM interface

```

## Quick Start

### 1. Prerequisites

A Python 3.9+ environment is recommended.

### 2. Installation

Install the required dependencies:

```bash
numpy
pandas
torch
spacy
python-igraph
tqdm
transformers
sentence-transformers
openai
httpx
spacy en_core_web_trf package

```

### 3. Data Preparation

After set your API, ensure your data directory (default: `./dataset`) contains the target dataset folder with the following structure:

* `chunks.json`: A list containing all document text chunks.
* `questions.json`: A list of JSON objects containing `question` and `answer` fields.

### 5. Running the Model

Launch the complete indexing and retrieval pipeline using the following command:

```bash
python run.py \
    --dataset_name 2wikimultihop \
    --use_vectorized_retrieval \
    --max_iterations 3 \
    --retrieval_top_k 5
```

