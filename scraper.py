import requests
import json
import time
import os
from pathlib import Path

#SUBREDDITS
target_subreddits = ["MachineLearning", "LangChain", "huggingface", "homelab", "dataengineering"] #<-- @TODO:EDIT THIS LIST TO ADD MORE SUBREDITS

#SETTINGS
POSTS_LIMIT = 100
TOP_K_COMMENTS = 5
OUTPUT_DIR = Path("datasets/scrapes")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
}

def get_json(url):
    """
    Helper to fetch JSON with error handling and rate limit pausing.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 429:
            print("  [!] Rate limited. Sleeping for 5 seconds...")
            time.sleep(5)
            response = requests.get(url, headers=HEADERS, timeout=10)

        if response.status_code != 200:
            print(f"  [Error] Status {response.status_code} for {url}")
            return None
            
        return response.json()
    except Exception as e:
        print(f"  [Exception] {e}")
        return None

def format_post_text(post_data, comments_data):
    """
    Formats the raw JSON data according to preprocess.py
    """

    title = post_data.get('title', 'No Title')
    author = post_data.get('author', '[deleted]')
    score = post_data.get('score', 0)
    url = post_data.get('url', '')
    num_comments = post_data.get('num_comments', 0)
    created_utc = post_data.get('created_utc', 0)
    selftext = post_data.get('selftext', '')
    
    lines = []
    lines.append(f"Post Title: {title}")
    lines.append(f"Author: {author}")
    lines.append(f"Score: {score}")
    lines.append(f"URL: {url}")
    lines.append(f"Number of comments: {num_comments}")
    lines.append(f"Created UTC: {created_utc}")
    
    lines.append("\nPost Content:")
    lines.append(selftext if selftext else "[No content/Link only]")
    
    valid_comments = [c['data'] for c in comments_data 
                      if c['kind'] == 't1' and 'body' in c['data']]
    
    valid_comments.sort(key=lambda x: x.get('score', 0), reverse=True)
    top_comments = valid_comments[:TOP_K_COMMENTS]

    lines.append(f"\n\nTop {len(top_comments)} comments:")
    
    for i, comment in enumerate(top_comments, 1):
        lines.append(f"\nComment {i}:")
        c_author = comment.get('author', '[deleted]')
        c_score = comment.get('score', 0)
        c_created = comment.get('created_utc', 0)
        c_body = comment.get('body', '')
        
        lines.append(f"Author: {c_author}")
        lines.append(f"Score: {c_score}")
        lines.append(f"Created UTC: {c_created}")
        lines.append(f"Comment: {c_body}")
        
    return "\n".join(lines)

def run_scraper():
    print(f"Starting to scrape for {len(target_subreddits)} subreddits...")

    for sub_name in target_subreddits:
        print(f"--- Scraping r/{sub_name} ---")
        
        # URL format: https://www.reddit.com/r/NAME/top.json?limit=100&t=all
        list_url = f"https://www.reddit.com/r/{sub_name}/top.json?limit={POSTS_LIMIT}&t=month"
        data = get_json(list_url)
        
        if not data or 'data' not in data or 'children' not in data['data']:
            print(f"  [Skip] No data found for r/{sub_name}")
            continue

        posts = data['data']['children']
        save_dir = OUTPUT_DIR / f"scrape_{sub_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Found {len(posts)} posts. Fetching details...")

        for i, post_wrapper in enumerate(posts):
            post = post_wrapper['data']
            post_id = post['id']
            permalink = post['permalink'] 
            
            # URL format: https://www.reddit.com/r/python/comments/ID.json
            detail_url = f"https://www.reddit.com{permalink[:-1]}.json"
            
            #DONT REMOVE SLEEP HERE!!!
            time.sleep(1.0) 
            
            detail_data = get_json(detail_url)
            
            if detail_data and isinstance(detail_data, list) and len(detail_data) >= 2:

                comment_listing = detail_data[1]['data']['children']                
                file_content = format_post_text(post, comment_listing)
                
                file_path = save_dir / f"post_{post_id}.txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
                    
                if i % 5 == 0:
                    print(f"Saved {i+1}/{len(posts)}: {post['title'][:30]}...")
            else:
                print(f"[Warning] Failed to fetch details for {post_id}")

    print("\nDone! Verify the files in 'datasets/scrapes/'")

if __name__ == "__main__":
    run_scraper()