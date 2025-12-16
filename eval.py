
from typing import List, Tuple
from insert_data import vectorize_text
import helix
import torch
from transformers import AutoTokenizer, AutoModel

# 50 texts/questions as input for the system each with a specific subreddit as a label
#   - then query top 20 from database and
#   - calcualte what % are the correct subreddit
#   - sum and average
# this is our recall score

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
