# Retrieval-Augmented Generation Reddit Search Engine
**Advanced Information Retrieval – Group 12**

## Getting started
```
uv sync
source .venv/bin/activate
git clone git@hf.co:datasets/PranavVerma-droid/reddit data/reddit

curl -sSL "https://install.helix-db.com" | bash
helix compile
helix push dev
```


### Game plan
text input query
    -> make into sentence like data with llm
    -> search based on similarity (in database)
    -> output top 5 posts/comments, subreddit, sort based on upvotes
    -> have llm make output nicer

- possibly write a web scraper if you want?

- [ ] choose which data we want
- [ ] pre-process data (title, content [vectorize], top k comments, url)
- [ ] vectorize all data and insert into db with tags to each vector with subreddit, upvotes, downvotes
- [ ] user text input, reformulate with an llm to be like the data in the db
- [ ] vector similarity search in db, output top 5 closest nodes
- [ ] evaluation criteria
    - does the subreddit make sense for the query?
- [ ]




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
```
