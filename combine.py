import csv
import glob
import re
from collections import Counter
import duckdb

files = sorted(f.replace("\\", "/") for f in glob.glob("reddit_data/reddit_transit_*.csv"))
print("Files found:", len(files))

rows = duckdb.sql(f"""
    SELECT id, subreddit, created_utc,
           strftime(to_timestamp(created_utc), '%Y-%m') AS month,
           score, body
    FROM read_csv({files!r}, union_by_name = true)
    QUALIFY row_number() OVER (PARTITION BY id) = 1
""").fetchall()

def clean(t):
    t = re.sub(r"\s+([.,!?;:)\]%])", r"\1", t)          # "word ," -> "word,"
    t = re.sub(r"([(\[$])\s+", r"\1", t)                 # "( word" -> "(word"
    t = re.sub(r"(\w)\s*'\s*(t|s|re|ve|ll|d|m)\b", r"\1'\2", t)  # "isn ' t" -> "isn't"
    t = re.sub(r'"\s+(.*?)\s+"', r'"\1"', t)             # '" text "' -> '"text"'
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

with open("reddit_transit_all.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["id", "subreddit", "created_utc", "month", "score", "body"])
    for r in rows:
        w.writerow(list(r[:5]) + [clean(r[5])])

print("Total comments:", len(rows))
print()
print("By subreddit:")
for sub, n in Counter(r[1] for r in rows).most_common():
    print(f"  {sub:15} {n}")
print()
print("By month:")
for m, n in sorted(Counter(r[3] for r in rows).items()):
    print(f"  {m}  {n}")
print()
print("Saved: reddit_transit_all.csv")
