"""Split labeled Reddit comments into three groups (works with prompt version v3).

  1. in_taxonomy       - real rider concerns that fit your 10 categories  -> keep for training
  2. outside_taxonomy  - real concerns that fit NO subtype                 -> keep for new-category review
  3. dropped           - NOT_RELEVANT_FEEDBACK                             -> excluded

Items whose evidence quote was not found in the comment are KEPT but marked
evidence_found = False, so you can review them instead of losing them.

Standard library only.   Run:  python filter_feedback.py
"""
import csv
import json
from collections import Counter

INPUT = "reddit_labels_v3.jsonl"
OUT_IN = "reddit_feedback_in_taxonomy.csv"
OUT_OUTSIDE = "reddit_feedback_outside_taxonomy.csv"
OUT_DROPPED = "reddit_dropped.csv"

recs = [json.loads(line) for line in open(INPUT, encoding="utf-8")]
recs = [r for r in recs if r.get("status") == "ok"]

in_rows, out_rows, drop_rows = [], [], []
groups, subtypes = Counter(), Counter()
quotes_missing = 0

for r in recs:
    res = r["result"]
    reqs = [s for s in res["service_requests"] if s["taxonomy_valid"]]
    outside = res["outside_taxonomy_concerns"]

    if reqs:
        groups["in taxonomy"] += 1
        for s in reqs:
            subtypes[f"{s['category']} / {s['subtype']}"] += 1
            quotes_missing += not s["evidence_found"]
            in_rows.append([r["id"], r["city"], res["city_context"], r["body"], s["category"], s["subtype"],
                            s["evidence"], s["evidence_found"], s["location"], s["location_role"],
                            s["route"], s["requested_change"], s["confidence"]])
    if outside:
        groups["outside taxonomy"] += 1
        for o in outside:
            quotes_missing += not o["evidence_found"]
            out_rows.append([r["id"], r["city"], res["city_context"], r["body"], o["description"],
                             o["evidence"], o["evidence_found"], o["location"], o["location_role"]])
    if not reqs and not outside:
        groups["dropped"] += 1
        drop_rows.append([r["id"], r["city"], r["body"], res["feedback_status"], res["explanation"]])


def save(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


save(OUT_IN, ["id", "subreddit_city", "city_context", "comment", "category", "subtype", "evidence",
              "evidence_found", "location", "location_role", "route", "requested_change", "confidence"], in_rows)
save(OUT_OUTSIDE, ["id", "subreddit_city", "city_context", "comment", "need_description", "evidence",
                   "evidence_found", "location", "location_role"], out_rows)
save(OUT_DROPPED, ["id", "subreddit_city", "comment", "feedback_status", "explanation"], drop_rows)

print(f"Comments labeled: {len(recs)}")
print(f"  Fits your taxonomy : {groups['in taxonomy']}  -> {OUT_IN} ({len(in_rows)} request rows)")
print(f"  Outside taxonomy   : {groups['outside taxonomy']}  -> {OUT_OUTSIDE} ({len(out_rows)} rows)")
print(f"  Dropped            : {groups['dropped']}  -> {OUT_DROPPED}")
print("  (a comment can have both in-taxonomy and outside-taxonomy concerns)")
print(f"  Items whose quote was not found in the comment (review these): {quotes_missing}")
print("\nSubtypes found:")
for k, v in subtypes.most_common():
    print(f"  {v:4}  {k}")
