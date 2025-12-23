import requests
import pandas as pd
from datetime import datetime, timezone

# ---------- CONFIG ---------- #

BASE_URL = "https://api.pullpush.io/reddit/search/submission/"

SUBREDDITS = [
    "wallstreetbets",
    "MemeStockMarket",
    "pennystocks",
    "stocks",
    "CryptoCurrency",
]

BATCH_SIZE = 100          # max 100 per request for this API
TOTAL_PER_SUB = 5000      # target posts per subreddit (change if you like)

# earliest date you care about (UTC). Older posts will not be fetched.
EARLIEST_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()

OUTPUT_CSV = "meme_reddit_historical.csv"

# ---------- FUNCTION ---------- #

def fetch_submissions_paginated(subreddit,
                                total_limit=5000,
                                batch_size=100,
                                earliest_ts=0):
    
    all_posts = []
    before = None  
    print(f"\n[START] r/{subreddit}")

    while len(all_posts) < total_limit:
        params = {
            "subreddit": subreddit,
            "size": batch_size,
            "sort": "desc",
            "sort_type": "created_utc",
        }
        if before is not None:
            params["before"] = int(before)

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"[ERROR] request failed for r/{subreddit}: {e}")
            break

        data = resp.json()
        batch = data.get("data", [])
        if not batch:
            print("[INFO] no more posts returned, stopping.")
            break

        # collect posts
        for p in batch:
            created_utc = p.get("created_utc")
            if created_utc is None:
                continue

            # stop if we reached earlier than earliest_ts
            if created_utc < earliest_ts:
                print("[INFO] reached earliest date, stopping.")
                return all_posts

            all_posts.append({
                "subreddit": p.get("subreddit", subreddit),
                "id": p.get("id", ""),
                "title": p.get("title", ""),
                "selftext": p.get("selftext", ""),
                "created_utc": created_utc,
                "score": p.get("score", 0),
                "num_comments": p.get("num_comments", 0),
                "url": p.get("full_link") or p.get("url", ""),
            })

        print(f"[OK] r/{subreddit}: total collected so far = {len(all_posts)}")

        # move 'before' to the oldest post we just saw
        oldest_ts = min(p["created_utc"] for p in batch if "created_utc" in p)
        # subtract 1 second to avoid overlap
        before = oldest_ts - 1

        # simple safety break in case something weird happens
        if len(batch) < batch_size:
            print("[INFO] last batch smaller than batch_size, likely reached end.")
            break

    return all_posts

# ---------- MAIN COLLECTION ---------- #

all_posts = []

for sub in SUBREDDITS:
    posts_sub = fetch_submissions_paginated(
        subreddit=sub,
        total_limit=TOTAL_PER_SUB,
        batch_size=BATCH_SIZE,
        earliest_ts=EARLIEST_DATE,
    )
    print(f"[DONE] r/{sub}: collected {len(posts_sub)} posts.")
    all_posts.extend(posts_sub)

if not all_posts:
    raise RuntimeError("No posts collected. Check API / params.")

df = pd.DataFrame(all_posts)

# remove duplicates by Reddit id
if "id" in df.columns:
    df = df.drop_duplicates(subset="id")

# convert timestamp to datetime
df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)

# order columns
cols = [
    "subreddit",
    "id",
    "created_utc",
    "created_dt",
    "score",
    "num_comments",
    "title",
    "selftext",
    "url",
]
df = df[[c for c in cols if c in df.columns]]

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n[FINAL] Saved {len(df)} posts to '{OUTPUT_CSV}'.")
print(df.groupby("subreddit").size())
print("\nPreview:")
print(df.head())