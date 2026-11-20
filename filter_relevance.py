"""STEP 1 - Filter: keep only Reddit comments that contain a transit request or concern.

Reads reddit_transit_all.csv (41,907 comments) and asks GPT ONE short question per comment:
    Does the commenter state a transit request, need, problem, or concern?   KEEP / DROP
Off-topic comments (housing advice, policy debate, car traffic, jokes...) are dropped.
Concerns that do not fit your taxonomy (safety, driver behaviour, fares...) are KEPT.
When GPT is unsure it keeps the comment - it is better to keep a few extra than to lose real concerns.

Cheap: a short prompt, 20 comments per API call, and a tiny answer (just the comment numbers to keep\nand drop - no explanations), so very few tokens are used.
Only the kept comments go on to the detailed labeling (label_reddit.py).

Standard library only.   Run:  python filter_relevance.py
Outputs (resumable - rerun to continue):
    relevance.jsonl                 one line per comment: keep true/false
    reddit_transit_concerns.csv     only the KEPT comments  -> input for the next step
"""
import csv
import getpass
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== SETTINGS =====
INPUT_FILE = "reddit_transit_all.csv"
LIMIT = 200              # test on 200 first; then set to 100000 for all 41,907
BATCH_SIZE = 20          # comments per API call
WORKERS = 4              # calls running at the same time
MAX_CHARS = 1500         # very long comments are cut to this length
MODEL = "gpt-5.6-luna"
OUTPUT_JSONL = "relevance.jsonl"
OUTPUT_KEPT = "reddit_transit_concerns.csv"
# ====================

SYSTEM_PROMPT = """Task: screen Reddit comments for a study of public-transit rider feedback.

KEEP a comment if the writer states a problem, need, request, or complaint about public transit:
- service: routes, frequency, hours, delays, crowding, missing connections
- stops, stations, getting to them (sidewalks, crossings), transfers
- new transit lines or modes
- fares, safety, drivers or staff, vehicles, cleanliness
- support or opposition to a specific route, stop, or proposed change; questions about service
A personal experience counts, even without "please" ("the bus keeps passing me").

DROP a comment if transit is only mentioned:
- housing, neighbourhood, or directions advice
- policy, funding, or history debate with no concrete need
- car traffic or parking only
- jokes or off-topic

If unsure, KEEP.

Output JSON only: {"keep": [comment numbers], "drop": [comment numbers]}
Every comment number must appear exactly once."""

lock = threading.Lock()


def call_gpt(api_key, batch):
    listing = "\n".join(f"[{k}] {r['body'][:MAX_CHARS]}" for k, r in enumerate(batch, 1))
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": f"{len(batch)} comments:\n\n{listing}"}],
        "max_completion_tokens": 2000,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return json.loads(data["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(15 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < 4:
                time.sleep(15)
                continue
            raise RuntimeError(f"error: {e}")


def process(api_key, batch, out):
    ans = call_gpt(api_key, batch)
    def nums(key):
        return {int(x) for x in (ans.get(key) or []) if str(x).strip().isdigit()}
    keep, drop = nums("keep"), nums("drop")
    written = 0
    with lock:
        for k, r in enumerate(batch, 1):
            if k not in keep and k not in drop:   # no answer -> not written, retried on the next run
                continue
            out.write(json.dumps({"id": r["id"], "keep": k in keep}) + "\n")   # in both lists -> keep
            written += 1
        out.flush()
    return written, len(batch) - written


def write_kept(rows_by_id):
    keep = {}
    for line in open(OUTPUT_JSONL, encoding="utf-8"):
        r = json.loads(line)
        keep[r["id"]] = r
    kept_rows = [rows_by_id[i] for i, r in keep.items() if r["keep"] and i in rows_by_id]
    with open(OUTPUT_KEPT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "subreddit", "month", "score", "body"])
        for row in kept_rows:
            w.writerow([row["id"], row["subreddit"], row.get("month", ""), row.get("score", ""), row["body"]])
    return keep, kept_rows


def main():
    with open(INPUT_FILE, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows_by_id = {r["id"]: r for r in rows}

    # Screen the 60 evaluation comments (pilot rows 20-79) first, so the test run can be
    # compared with GPT's detailed v3 labels and with your hand coding.
    if os.path.exists("pilot_300.csv"):
        with open("pilot_300.csv", encoding="utf-8-sig") as f:
            first = [r["id"] for r in list(csv.DictReader(f))[20:80]]
        first_set = set(first)
        rows = [rows_by_id[i] for i in first if i in rows_by_id] + [r for r in rows if r["id"] not in first_set]

    done = set()
    if os.path.exists(OUTPUT_JSONL):
        done = {json.loads(line)["id"] for line in open(OUTPUT_JSONL, encoding="utf-8")}
    todo = [r for r in rows[:LIMIT] if r["id"] not in done]
    print(f"{min(LIMIT, len(rows))} comments selected, {len(done)} already screened, {len(todo)} to screen.")

    if todo:
        api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("Paste your OpenAI API key (hidden): ").strip()
        if not api_key:
            sys.exit("No API key given.")
        batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
        print(f"{len(batches)} API calls, {WORKERS} at a time...")
        done_n = missing = 0
        start = time.time()
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as out, ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(process, api_key, b, out) for b in batches]
            for k, fut in enumerate(as_completed(futures), 1):
                try:
                    w, m = fut.result()
                    done_n += w
                    missing += m
                except Exception as e:
                    print("   batch failed:", e)
                    if "HTTP 401" in str(e) or "HTTP 404" in str(e):
                        pool.shutdown(cancel_futures=True)
                        sys.exit("Stopping: check your API key or the MODEL name.")
                if k % 10 == 0 or k == len(batches):
                    mins = (time.time() - start) / 60
                    print(f"   {k}/{len(batches)} calls done, {done_n} comments screened ({mins:.1f} min)")
        if missing:
            print(f"   {missing} comments got no answer - run the script again to retry them.")

    keep, kept_rows = write_kept(rows_by_id)
    n = len(keep)
    k = sum(r["keep"] for r in keep.values())
    print(f"\nScreened: {n}   KEPT: {k} ({k / n:.0%})   DROPPED: {n - k}")
    print(f"Kept comments saved to {OUTPUT_KEPT}")
    by_sub = Counter(rows_by_id[i]["subreddit"] for i, r in keep.items() if r["keep"] and i in rows_by_id)
    print("Kept by subreddit:", ", ".join(f"{s} {c}" for s, c in by_sub.most_common()))

    # quick agreement check with the detailed v3 labels, if any overlap
    if os.path.exists("reddit_labels_v3.jsonl"):
        agree = total = 0
        for line in open("reddit_labels_v3.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok" and r["id"] in keep:
                v3_rel = r["result"]["feedback_status"] != "NOT_RELEVANT_FEEDBACK"
                agree += v3_rel == keep[r["id"]]["keep"]
                total += 1
        if total:
            print(f"Agreement with the detailed v3 labels on {total} shared comments: {agree}/{total}")


if __name__ == "__main__":
    main()
