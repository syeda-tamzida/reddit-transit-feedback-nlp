import duckdb

f = "reddit_data/reddit_transit_2016-02.csv"

print("Total comments:")
print(duckdb.sql(f"SELECT count(*) AS comments FROM '{f}'"))

print("Comments per subreddit:")
print(duckdb.sql(f"SELECT subreddit, count(*) AS n FROM '{f}' GROUP BY subreddit"))

print("First 10 comments:")
rows = duckdb.sql(f"SELECT body FROM '{f}' LIMIT 10").fetchall()
for i, (body,) in enumerate(rows, 1):
    print(f"{i}. {body[:200]}")
    print()
