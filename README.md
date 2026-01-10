# Retrieval-Augmented Generation Reddit Search Engine
**Advanced Information Retrieval – Group 12**

## Getting started
```
uv sync
source .venv/bin/activate
curl -sSL "https://install.helix-db.com" | bash
helix push
python insert_data.py
```

Now all the data should be inserted into the database and you can start using the system via `./run.py`!

## Description

## Testing response quality
Ordered by how the questions are ordered in `run.py`.

### Llama 3.2 (3B)
1. 9/10
2. 5/10 didn't explain well
3. 10/10
4. 10/10
5. 0/10 said something like, it can't respond with reddit urls because its unreliable
6. 7/10 no links provided
7. 9/10
8. 7/10 a bit short
9. 8/10 caught repetition in db query response and filtered it out
10. 3/10 couldn't generate answer using provided sources
11. 10/10 code example as well, very explanatory with links
12. 8/10 informative and with links, not to long
13. 10/10 all links, good explanation
14. 9/10 code example and links, very informative
15. 8/10 good, suggested sub-reddits as well in a nice manner
16. 6/10 one of the links was messed up and couldn't just click on it, it just gave the subreddit/post name
17. 6/10 no source urls to reddit posts
18. 9/10 explains well and gives links
19. 7/10 a bit short
20. 1/10 way to short and repeated links

### Mistral (7B)
1. 9/10 well explained and good links
2. 7/10 a bit short
3. 2/10 no links
4. 9/10 well explained and good links
5. 9/10 well explained and links
6. 2/10 no reddit links to posts
7. 6/10 explained well, but incorrect link formatting
8. 4/10 useless image link, but good explanation
9. 3/10 repeating the same link
10. 2/10 just printed the same link 4 times
11. 5/10 gave a fake false link, but good amount of information
12. 10/10 good, with code example
13. 8/10 good, a bit short tho
14. 8/10 good, a bit short tho
15. 5/10 bad link formatting
16. 2/10 no links
17. 5/10 incorrect link formatting
18. 8/10 a bit short, but explained well
19. 2/10 no links
20. 8/10 explained and with links

## Resources
- Dataset: `https://huggingface.co/datasets/PranavVerma-droid/reddit`
- Embedding model: `https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1`
- LLMs via Ollama
    - Mistral 7B: `https://mistral.ai/news/announcing-mistral-7b`
    - Llama 3.2 3B: `https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/`
