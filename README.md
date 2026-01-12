# Retrieval-Augmented Reddit Search Engine
**Advanced Information Retrieval – Group 12**

This project is an implementation of a **Retrieval-Augmented Generation (RAG)** system for searching and answering questions based on Reddit posts. The system retrieves relevant Reddit content from a vector database and uses locally hosted Large Language Models (LLMs) to generate real life grounded responses.

Our primary goal is to **evaluate response quality**, source attribution, and robustness across different LLMs when answering real-world questions using real life data gathered on the social media platform Reddit.
---

## System Overview

The pipeline consists of:

1. Reddit data collection
2. Preprocessing and filtering
3. Vector embedding and database insertion
4. Retrieval-augmented response generation
5. Evaluation of response quality

All components required for **full reproducibility** are included in this repository.

---

## Project Structure

```text
.
├── scraper.py          # Reddit data collection
├── preprocess.py       # Data cleaning and preprocessing
├── insert_data.py      # Insert processed data into vector DB
├── eval.py             # Model evaluation logic
├── run.py              # Main system entry point
├── queries.rs          # Optimized database queries (Rust)
├── db/                 # Vector database storage
├── datasets/
│   └── scrapes/        # Raw and processed Reddit data
├── results/            # Evaluation plots and summaries
├── requirements.txt    # Python dependencies
├── pyproject.toml
├── uv.lock
└── README.md
```

## Dataset(s)

We use a publicly available Reddit dataset:

- **Dataset:** https://huggingface.co/datasets/PranavVerma-droid/reddit
- **Source:** Hugging Face
- **Content:** Reddit posts and comments across multiple subreddits


## Models & Tools

### Embeddings

- **mixedbread-ai/mxbai-embed-large-v1**
  https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1

### Large Language Models (Ollama)

- **Llama 3.2 (3B)**
  https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

- **Mistral (7B)**
  https://mistral.ai/news/announcing-mistral-7b

---

## Installation

### Requirements

- Python ≥ 3.10
- Ollama installed and running locally

Install dependencies:

```bash
pip install -r requirements.txt
```

# Reproducibility

To reproduce the complete pipeline from scratch, follow these steps:

1. **Scrape Reddit Data**
   ```bash
   python scraper.py
   ```

2. **Preprocess the Data**
   ```bash
   python preprocess.py
   ```

3. **Insert Data into the Database**
   ```bash
   python insert_data.py
   ```

4. **Run Evaluation**
   ```bash
   python eval.py
   ```

5. **Start the System**
   ```bash
   python run.py
   ```


## Evaluation

### Methodology

Responses were evaluated manually across a fixed set of questions, ordered as defined in `run.py`.

Each answer was rated on a **0–10 scale** based on:

- Explanation quality
- Use of retrieved Reddit content
- Presence of source URLs
- Correct link formatting
- Redundancy handling
- Ability to answer using provided sources

### Results Summary

| Model          | Average Score |
|----------------|---------------
| Llama 3.2 (3B) | ~7.1 / 10     |
| Mistral (7B)   | ~5.7 / 10     |


---

## Limitations

- Manual evaluation introduces subjectivity
- Reddit content varies widely in quality
- Some LLMs refuse to generate Reddit URLs under certain conditions
- Link formatting errors were common and model-dependent

---

## External Resources & Credits

- **Reddit Dataset:**
  https://huggingface.co/datasets/PranavVerma-droid/reddit

- **Embedding Model:**
  https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1

- **Mistral 7B:**
  https://mistral.ai/news/announcing-mistral-7b

- **Llama 3.2 (3B):**
  https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
