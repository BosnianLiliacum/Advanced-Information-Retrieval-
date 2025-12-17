# Retrieval-Augmented Generation Reddit Search Engine
**Advanced Information Retrieval – Group 12**

## Getting started
```
uv sync
source .venv/bin/activate
git clone git@hf.co:datasets/PranavVerma-droid/reddit data/reddit

curl -sSL "https://install.helix-db.com" | bash
helix push dev
python insert_data.py
```

### Game plan
text input query
    -> make into sentence like data with llm
    -> search based on similarity (in database)
    -> output top 5 posts/comments, subreddit, sort based on upvotes
    -> have llm make output nicer

- [X] choose which data we want
- [ ] web scraper for more data
- [X] pre-process data (title, content [vector], top k comments, url)
- [X] vectorize all data and insert into db with tags to each vector with subreddit, upvotes, downvotes
- [X] user text input, reformulate with an llm to be like the data in the db
- [X] vector similarity search in db, output top 5 closest nodes
- [ ] evaluation criteria
    - [ ] recall, does the subreddit make sense for the query?
    - [ ] some sort of sentiment analysis with an embedding model?
    - [ ] graph of cosine similarity of prompt to text
    - [ ] WE NEED PLOTS FOR THIS

### Resources
- Dataset: `https://huggingface.co/datasets/PranavVerma-droid/reddit`
- Embedding model: `https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1`
- LLM via Ollama: `https://mistral.ai/news/announcing-mistral-7b`
