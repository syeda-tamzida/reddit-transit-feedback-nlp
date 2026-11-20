"""Score GPT's Reddit labels against your hand coding.

Reads:
    Reddit_Hand_Coding_60.xlsx   (sheet "Coding", only rows marked done = Yes)
    reddit_labels_v3.jsonl       (GPT output from label_reddit.py)
Writes:
    reddit_eval_by_comment.csv   one row per comment: your coding vs GPT, side by side

Metrics:
    Feedback detection precision / recall   (relevant = IN_TAXONOMY or OUTSIDE_TAXONOMY)
    3-way feedback_status accuracy
    Request precision / recall / F1         (match on category + subtype)
    Category-level F1                       (match on category only)
    Request-location pair F1                (subtype AND location must both match)
    Outside-taxonomy recall / precision     (comment level)
    Evidence quotes found in the comment    (GPT side)

Needs openpyxl (pure Python):  pip install openpyxl
Run:  python score_reddit_eval.py
"""
import csv
import json
import os
import re
import sys
from collections import Counter

from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import taxonomy_v2 as tx  # noqa: E402
except ImportError:
    sys.exit("taxonomy_v2.py is required but not included in this repository. "
             "Place your taxonomy file next to this script.")

CODING_FILE = "Reddit_Hand_Coding_60.xlsx"
GPT_FILE = "reddit_labels_v3.jsonl"
OUT_CSV = "reddit_eval_by_comment.csv"

RELEVANT = {"IN_TAXONOMY", "OUTSIDE_TAXONOMY"}
EMPTY_PLACES = {"", "unspecified", "none", "n a", "na", "systemwide"}


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split())


def place_key(text):
    t = norm(text)
    t = re.sub(r"^(the|a|an) ", "", t)
    return "" if t in EMPTY_PLACES else t


def places_match(a, b):
    a, b = place_key(a), place_key(b)
    if not a and not b:
        return True
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) >= 0.5


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------- read hand coding ----------
ws = load_workbook(CODING_FILE, data_only=True)["Coding"]
header = [c.value for c in ws[1]]
human = {}
for row in ws.iter_rows(min_row=2, values_only=True):
    rec = dict(zip(header, row))
    if str(rec.get("done") or "").strip().lower() != "yes":
        continue
    reqs = []
    for k in (1, 2, 3):
        val = str(rec.get(f"req{k}_subtype") or "").strip()
        if not val:
            continue
        cat, _, sub = val.partition(" / ")
        cat, sub, valid, _ = tx.resolve_category_subtype(cat, sub)
        reqs.append({"category": cat, "subtype": sub, "location": rec.get(f"req{k}_location") or ""})
    outside = [rec.get(f"outside{k}_description") for k in (1, 2) if rec.get(f"outside{k}_description")]
    human[str(rec["id"])] = {
        "status": str(rec.get("feedback_status") or "").strip().upper() or "NOT_RELEVANT_FEEDBACK",
        "requests": reqs, "outside": outside, "comment": rec.get("comment") or "", "city": rec.get("city"),
    }

# ---------- read GPT ----------
gpt = {}
for line in open(GPT_FILE, encoding="utf-8"):
    r = json.loads(line)
    if r.get("status") == "ok":
        gpt[str(r["id"])] = r["result"]

ids = [i for i in human if i in gpt]
missing = [i for i in human if i not in gpt]
print(f"Hand-coded comments (done = Yes): {len(human)}   with GPT labels: {len(ids)}")
if missing:
    print(f"  {len(missing)} coded comments have no GPT label yet - run label_reddit.py first.")
if not ids:
    raise SystemExit("Nothing to score yet.")

# ---------- score ----------
c = Counter()
rows_out = []
for i in ids:
    h, g = human[i], gpt[i]
    h_rel, g_rel = h["status"] in RELEVANT, g["feedback_status"] in RELEVANT
    c["fd_tp"] += h_rel and g_rel
    c["fd_fp"] += (not h_rel) and g_rel
    c["fd_fn"] += h_rel and not g_rel
    c["status_ok"] += h["status"] == g["feedback_status"]

    g_reqs = [s for s in g["service_requests"] if s["taxonomy_valid"]]

    # subtype-level (multiset)
    hs = Counter((r["category"], r["subtype"]) for r in h["requests"])
    gs = Counter((s["category"], s["subtype"]) for s in g_reqs)
    tp = sum((hs & gs).values())
    c["rq_tp"] += tp; c["rq_fp"] += sum(gs.values()) - tp; c["rq_fn"] += sum(hs.values()) - tp

    # category-level
    hc = Counter(r["category"] for r in h["requests"])
    gc = Counter(s["category"] for s in g_reqs)
    tp = sum((hc & gc).values())
    c["cat_tp"] += tp; c["cat_fp"] += sum(gc.values()) - tp; c["cat_fn"] += sum(hc.values()) - tp

    # request-location pairs (greedy matching)
    unmatched = list(g_reqs)
    pair_tp = 0
    for r in h["requests"]:
        for s in unmatched:
            if (s["category"], s["subtype"]) == (r["category"], r["subtype"]) and places_match(r["location"], s["location"]):
                unmatched.remove(s)
                pair_tp += 1
                break
    c["pr_tp"] += pair_tp; c["pr_fp"] += len(g_reqs) - pair_tp; c["pr_fn"] += len(h["requests"]) - pair_tp

    # outside taxonomy (comment level)
    h_out, g_out = bool(h["outside"]), bool(g["outside_taxonomy_concerns"])
    c["ot_tp"] += h_out and g_out; c["ot_fp"] += (not h_out) and g_out; c["ot_fn"] += h_out and not g_out

    for item in g["service_requests"] + g["outside_taxonomy_concerns"]:
        c["ev_total"] += 1
        c["ev_found"] += bool(item.get("evidence_found"))

    rows_out.append([
        i, h["city"], h["comment"], h["status"], g["feedback_status"], h["status"] == g["feedback_status"],
        "; ".join(f"{r['subtype']} @ {r['location'] or 'Unspecified'}" for r in h["requests"]),
        "; ".join(f"{s['subtype']} @ {s['location']}" for s in g_reqs),
        "; ".join(h["outside"]),
        "; ".join(o["description"] for o in g["outside_taxonomy_concerns"]),
    ])

with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["id", "city", "comment", "your_status", "gpt_status", "status_match",
                "your_requests", "gpt_requests", "your_outside", "gpt_outside"])
    w.writerows(rows_out)

n = len(ids)
print()
print(f"{'Metric':40} {'Precision':>10} {'Recall':>8} {'F1':>8}")
for name, key in [("Feedback detection (relevant vs not)", "fd"), ("Requests: category + subtype", "rq"),
                  ("Requests: category only", "cat"), ("Request-location pairs", "pr"),
                  ("Outside-taxonomy concerns", "ot")]:
    p, r, f = prf(c[f"{key}_tp"], c[f"{key}_fp"], c[f"{key}_fn"])
    print(f"{name:40} {p:10.2f} {r:8.2f} {f:8.2f}")
print()
print(f"3-way feedback_status accuracy: {c['status_ok']}/{n} = {c['status_ok'] / n:.2f}")
if c["ev_total"]:
    print(f"GPT evidence quotes found in the comment: {c['ev_found']}/{c['ev_total']}")
print(f"\nSide-by-side comparison saved to {OUT_CSV}")
