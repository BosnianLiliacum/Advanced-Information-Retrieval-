# Retrieval-Augmented Generation Reddit Search Engine
**Advanced Information Retrieval – Group 12**

---

## Overview
This project aims to develop a **Retrieval-Augmented Generation (RAG)** system that enhances large language models (LLMs) using **real Reddit user experiences**.

Often, finding a Reddit post that answers a specific question is time-consuming, while asking an LLM alone may lead to inaccurate or “hallucinated” responses.  
For example:  
> *“What’s the experience like riding the trams in Graz, Austria?”*  

An LLM has never had this experience, but Reddit users often share such firsthand accounts.  
Our system bridges this gap by retrieving relevant Reddit posts and injecting them into an LLM’s context window to improve the realism and accuracy of its responses.

---

## Main Idea
1. **Scrape or load** Reddit data (posts and threads) from public sources such as  
   - [HuggingFace Datasets](https://huggingface.co/)  
   - [Reddit Dataset on GitHub](https://github.com/linanqiu/reddit-dataset/tree/master)
2. **Pre-process and embed** each post using transformer-based embeddings (e.g., BERT).
3. **Store** both text and embeddings in a **vector database** that supports efficient similarity search.
4. **Retrieve** the most relevant posts for a given user query using cosine similarity or other heuristics.
5. **Augment** an LLM (e.g., LLaMA 4, Mistral 3.1) with those retrieved posts to generate a richer, more personal answer.

---

## Technologies & Tools
| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **ML / NLP** | PyTorch, Transformers |
| **Embeddings** | BERT, Sentence-Transformers |
| **Databases** | [HelixDB](https://github.com/HelixDB/helix-db), [Qdrant](https://github.com/qdrant/qdrant) |
| **LLMs** | [Meta’s LLaMA 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), [Mistral 3.1](https://mistral.ai/news/mistral-small-3-1) |
| **Visualization / Analysis** | Matplotlib, Pandas |
| **Documentation** | LaTeX, Overleaf |

---

## Data Processing Pipeline
```text
Dataset / Web Scraping
        ↓
Data Cleaning & Tokenization
        ↓
Embedding Generation (BERT)
        ↓
Vector Database Insertion (HelixDB or Qdrant)
        ↓
Query Embedding & Similarity Search
        ↓
Retrieval-Augmented LLM Response
