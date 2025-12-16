from typing import List, Tuple
import json
import helix
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

db = helix.Client(local=True, verbose=True)

tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
model = AutoModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def vectorize_text(text):
    inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()

    return embedding

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
text = "Hello everyone, I'm having a little trouble understanding how an LLM works locally.  If I'm using Ollama and I'm talking with, let say gemma2, I have a conversation.  I first give my name to the LLM. Get the response, then ask a bunch more questions.  If I then say what's my name, it will tell me my name."
vec = vectorize_text(text)
res = db.query(search_posts_vec(vec, 5))
print(res)
[print(f"{o}\n\n") for o in res[0]["posts"]]
