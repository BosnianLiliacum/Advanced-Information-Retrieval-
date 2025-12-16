from preprocess import load_all_posts
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

class upload_all_posts(helix.Query):
    def __init__(
        self,
        posts: List[Tuple[str, str, str, List[float], str, int, List[Tuple[str, int]]]]
    ):
        super().__init__()
        self.posts = posts

    def query(self) -> List[helix.Payload]:
        posts_payload = []
        for subreddit, title, content, vector, url, score, comments in self.posts:
            comments_payload = []
            for ct, score in comments:
                comments_payload.append({ "ic_content": ct, "ic_score": int(score) })
            posts_payload.append({
                "subreddit": subreddit,
                "title": title,
                "content": content,
                "vector": vector,
                "url": url,
                "score": int(score),
                "comments": comments_payload,
            })
        return [{ "posts": posts_payload }]

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
#vecs = []
#for _, _, content, _, _, _ in tqdm(data):
#    vec = vectorize_text(content)
#    vecs.append(vec)
#with open("embeded_vectors.json", "w") as f: json.dump(vecs, f, indent=2)

n_data = []
vecs = json.load(open("embeded_vectors.json"))
for (subreddit, title, content, url, score, comments), vec in tqdm(zip(data, vecs)):
    #print(f"subreddit: {subreddit}")
    #print(f"title: {title}")
    #print(f"content: {content}")
    #print(f"vec: {vec}")
    #print(f"url: {url}")
    #print(f"score: {score}")
    #print(f"comments: {comments}")
    n_data.append((subreddit, title, content, vec, url, score, comments))

part_size = len(n_data) // 20
n_parts = []
for i in range(20):
    start = i * part_size
    end = (i + 1) * part_size if i < 19 else len(n_data)
    n_parts.append(n_data[start:end])
print(f"split into 10 parts: {[len(part) for part in n_parts]}")

for i, part in enumerate(n_parts):
    print(part)
    print(f"Uploading part {i+1}/10 ({len(part)} posts)...")
    db.query(upload_all_posts(part))

#print(n_data[0])
#db.query(upload_all_posts(n_data))

posts = db.query("get_all_posts")
print(len(posts[0]["posts"]))
