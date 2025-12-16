from preprocess import load_all_posts
from typing import List, Tuple
import helix
import torch
from transformers import AutoTokenizer, AutoModel

db = helix.Client(local=True, verbose=True)

tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
model = AutoModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

class load_a_post(helix.Query):
    def __init__(
        self,
        subreddit: str,
        title: str,
        content: str,
        vector: List[float],
        url: str,
        score: int,
        comments: List[Tuple[str, int]],
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
        comments_payload = []
        for ct, score in self.comments:
            comments_payload.append({ "ic_content": ct, "score": score })
        return [{
                 "subreddit": self.subreddit,
                 "title": self.title,
                 "content": self.content,
                 "vector": self.vector,
                 "url": self.score,
                 "comments": comments_payload,
        }]

    def response(self, response):
        return response

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

data = load_all_posts("datasets/scrapes")
for subreddit, title, content, url, score, comments in data[:10]:
    vec = vectorize_text(content)
    payload = load_a_post(
            subreddit,
            title,
            content,
            vec,
            url,
            score,
            comments
    )
    db.query(payload)

posts = db.query("get_all_posts")
print(posts)
#print(f"num posts: ")
