# Retrieval-Augmented Generation Reddit Search Engine
**Advanced Information Retrieval – Group 12**

## Getting started
```
uv sync
source .venv/bin/activate
curl -sSL "https://install.helix-db.com" | bash
helix push dev
python insert_data.py
```

Now all the data should be inserted into the database and you can start using the system via `./run.py`!

## Description

## Resources
- Dataset: `https://huggingface.co/datasets/PranavVerma-droid/reddit`
- Embedding model: `https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1`
- LLMs via Ollama
    - Mistral 7B: `https://mistral.ai/news/announcing-mistral-7b`
    - Llama 3.2 3B: `https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/`
