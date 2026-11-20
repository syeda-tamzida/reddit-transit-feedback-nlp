import os
import duckdb
from huggingface_hub import hf_hub_download

# ===== SETTINGS: change these =====
MONTHS = ["2015-03", "2015-06", "2015-09", "2015-12", "2016-01"]
SUBREDDITS = ["pittsburgh", "transit", "seattle", "chicago", "boston",
              "nyc", "sanfrancisco", "bayarea", "washingtondc",
              "philadelphia", "portland", "toronto", "losangeles",
              "urbanplanning"]
KEYWORDS = ["bus", "buses", "bus stop", "bus route", "bus driver",
            "transit", "port authority", "busway", "light rail",
            "subway", "metro", "transfer"]
OUTPUT_FOLDER = "reddit_data"
DOWNLOAD_FOLDER = "downloads"
# ==================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
con = duckdb.connect()

sub_list = ", ".join(f"'{s}'" for s in SUBREDDITS)
pattern = "|".join(r"\b" + k + r"\b" for k in KEYWORDS)

for month in MONTHS:
    out_file = os.path.join(OUTPUT_FOLDER, f"reddit_transit_{month}.csv")
    if os.path.exists(out_file):
        print("Already done, skipping:", month)
        continue

    print("Downloading", month, "(about 9.5 GB, please wait)...")
    path = hf_hub_download(
        repo_id="fddemarco/pushshift-reddit-comments",
        repo_type="dataset",
        filename=f"data/RC_{month}.parquet",
        local_dir=DOWNLOAD_FOLDER,
    )

    print("Filtering", month, "...")
    p = path.replace("\\", "/")
    o = out_file.replace("\\", "/")
    con.sql(f"""
        COPY (
            SELECT id, subreddit, created_utc, score, body
            FROM read_parquet('{p}')
            WHERE lower(subreddit) IN ({sub_list})
              AND body NOT IN ('[deleted]', '[removed]')
              AND regexp_matches(lower(body), '{pattern}')
        ) TO '{o}' (HEADER)
    """)

    os.remove(path)
    print("Saved:", out_file, "(downloaded file deleted)")

print("All done.")
