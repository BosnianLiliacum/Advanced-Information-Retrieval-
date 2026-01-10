#!/usr/bin/env python3

from typing import List, Tuple
from insert_data import vectorize_text
import json
import requests
from pprint import pprint
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

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def get_ollama_response(prompt, model_name):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Ollama API request failed with status {response.status_code}")

def create_rephrase(text: str) -> str:
    prompt_template = f"""<instructions>
    Reformat this prompt that is a question into something that seems a bit more like the content ofo
    a reddit post. Here are three examples to go by. Do not change the topic or content of the question.
    Only output the rephrased text. Keep it short as this output is being used to do a vector search in a
    database of reddit posts.

    Example 1: "I currently have tailscale running in a docker stack and I’ve got the other containers in the stack set to depend on the tailscale instance, but what I really want to ensure is that this stack ONLY uses a certain exit node on my tailnet AND if that exit node becomes unavailable, I need the rest of the containers to shutdown until this requirement is satisfied. Anyone know of a solution?\nI’ve tried messing around with healthcheck’s and that sort of thing, but even that is proving a bit cumbersome with the stock image as it doesn’t even have curl builtin. I’ll follow up and try any ideas anyone may have."
    Example 2: "Was there some major improvement in Llama.cpp that was added to Ollama 0.2.8 or higher? I ask because my token generation speed seems like it got a massive boost after about version 0.2.8. \nI know Llamafile had some big breakthrough recently that improved their tokens speed for CPU inference and their stuff usually makes it into Llama.cpp which eventually makes it into Ollama. Maybe that’s what happened?"
    Example 3: "I was tired of making dirty scripts to bump the version of my projects in CI pipelines, so i think i went a bit overkill and made a dedicated tool for it.\nIncreasing version is a task that was surprisingly harder than i thought for what it is, as you need to parse TOML and Python files, parse versions, increase the version, update the files and push it to your repo without triggering an infinite loop of pipelines. Also since it automatically update files in your project i guess it should be somewhat reliable too."

    <prompt>
    {text}
    </prompt>
    </instructions>
    """
    return prompt_template

def create_prompt(out: list, query: str) -> str:
    formatted_context = ""
    subreddits = set()
    urls = []

    for idx, post in enumerate(out, 1):
        title = post.get("title", "N/A")
        subreddit = post.get("subreddit", "N/A")
        content = post.get("content", "N/A")
        comments = post.get("comments", "N/A")
        url = post.get("url", "N/A")

        formatted_context += f"Source {idx}: {title}\n"
        formatted_context += f"Subreddit: r/{subreddit}\n"
        formatted_context += f"Content: {content}\n"
        formatted_context += f"Comments: {comments}i\n"
        formatted_context += "-" * 80 + "\n\n"

        subreddits.add(subreddit)
        urls.append(url)

    subreddit_recs = ", ".join([f"r/{sub}" for sub in list(subreddits)[:3]])

    # Format URLs list
    urls_section = "\n".join([f"- {url}" for url in urls if url != 'N/A'])

    prompt_template = """<instructions>
    Based on the provided sources below, answer the user's question to the best of your ability. Go into depth,
    but don't ramble on. Keep it short and minimal with only the necessary information.

    In your response:
    1. Provide a comprehensive answer drawing from the sources
    2. List the top key points or answers clearly
    3. Recommend relevant subreddits where the user could visit for more detailed discussions on this topic
    4. Include all source URLs at the end for the user to explore further
    </instructions>

    <question>
    {query}
    </question>

    <sources>
    {context}
    </sources>

    <recommended_subreddits>
    For more discussions and answers on this topic, consider visiting: {subreddits}
    </recommended_subreddits>

    <source_urls>
    {urls}
    </source_urls>
    """

    prompt = prompt_template.format(
        query=query,
        context=formatted_context,
        subreddits=subreddit_recs,
        urls=urls_section
    )
    return prompt

if __name__ == "__main__":
    models = ["mistral:7b", "llama3.2:3b"]
    questions = [
        "Why is there a significant performance difference between downloading models directly from Ollama and installing GGUF models on Ollama, even when using the same quantization method?",
    ]

    n = 5

    for model in models:
        for q in questions:
            print(f"q: {q}")
            text_prompt = create_rephrase(q)
            #print(text_prompt)
            res = get_ollama_response(text_prompt, model)
            #print("----------")
            #print(res)

            vec = vectorize_text(res)
            res = db.query(search_posts_vec(vec, n))
            #pprint(res)

            # ['subreddit', 'title', 'content', 'url', 'comments']
            out = [o for o in res[0]["posts"][:n]]
            print(out)

            prompt = create_prompt(out, q)
            print(prompt)

            print("-----------------------")
            res = get_ollama_response(prompt, model)
            print(res)

            input("press enter for next question...")
        exit(1)
