from preprocess import load_all_posts
from typing import List, Tuple
import json
import helix
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
model = AutoModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
model.to(device)
model.eval()

class upload_a_post(helix.Query):
    def __init__(
        self,
        subreddit: str,
        title: str,
        content: str,
        vector: List[float],
        url: str,
        score: int,
        comments: List[str]
    ):
        super().__init__()
        self.subreddit = subreddit
        self.title = title
        self.content = content
        self.vector = vector
        self.url = url
        self.score = score
        self.comments = comments

    def query(self) -> List[helix.Payload]:
        #comments_payload = []
        #for ct in comments: comments_payload.append(ct)
        return [{
            "subreddit": self.subreddit,
            "title": self.title,
            "content": self.content,
            "vector": self.vector,
            "url": self.url,
            "score": int(self.score),
            "comments": self.comments,
        }]

    def response(self, response): return response

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

if __name__ == "__main__":
    data = load_all_posts("datasets/scrapes")

    VECTORIZE = False # set to true then run then false then run again

    if VECTORIZE:
        vecs = []
        for _, _, content, _, _, _ in tqdm(data):
            vec = vectorize_text(content)
            vecs.append(vec)
        with open("embeded_vectors.json", "w") as f: json.dump(vecs, f, indent=2)
    else:
        db = helix.Client(local=True, verbose=True)
        n_data = []
        vecs = json.load(open("embeded_vectors.json"))
        for (subreddit, title, content, url, score, comments), vec in tqdm(zip(data, vecs)):
            comments = [c for c, _ in comments]
            db.query(upload_a_post(subreddit, title, content, vec, url, score, comments))

        posts = db.query("get_all_posts")
        print(len(posts[0]["posts"]))
