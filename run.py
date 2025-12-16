#!/usr/bin/env python3

from typing import List, Tuple
from insert_data import vectorize_text
import json
import helix
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

db = helix.Client(local=True, verbose=True)

tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
model = AutoModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
model.to(device)
model.eval()

class search_posts_vec(helix.Query):
    def __init__(
        self,
        query_vec: List[float],
        k: int,
    ):
        super().__init__()
        self.query_vec = query_vec
        self.k = k

    def query(self) -> List[helix.Payload]:
        return [{
            "query": self.query_vec,
            "k": self.k,
        }]

    def response(self, response):
        return response

#text = input("prompt: ")
text = "Why is there a significant performance difference between downloading models directly from Ollama and installing GGUF models on Ollama, even when using the same quantization method?"
vec = vectorize_text(text)
res = db.query(search_posts_vec(vec, 10))
print(res)
[print(f"{o}\n\n") for o in res[0]["posts"]]
