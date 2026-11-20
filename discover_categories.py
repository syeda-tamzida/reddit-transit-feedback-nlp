"""Discover candidate NEW categories from outside-taxonomy concerns (E3 + E4).

Input : every reddit_labels_v*.jsonl in this folder (the outside_taxonomy_concerns GPT found)
Steps :
  1. Collect the concerns (up to MAX_CONCERNS).
  2. Embed each concern description (OpenAI text-embedding-3-small).
  3. Group similar concerns (average-linkage clustering, cosine similarity >= SIMILARITY).
  4. E3 - plain baseline: GPT only names each group.
  5. E4 - taxonomy-aware review: GPT compares each group with your existing subtypes and decides
          NEW_CATEGORY_CANDIDATE / MAP_TO_EXISTING / NOT_A_TRANSIT_CONCERN / TOO_VAGUE,
          checks whether the group should be split, and checks each concern individually.
  6. Redundancy check: GPT compares all new-category candidates with each other and says
          which ones should be merged, so you do not end up with near-duplicate categories.
Output:
  discovered_categories.csv   one row per group: E3 name vs E4 decision, closest existing subtype...
  concern_assignments.csv     one row per concern: its group and whether it fits an existing subtype
Nothing is added to the taxonomy automatically - you and Vincent decide.

Standard library only.   Run:  python discover_categories.py
"""
import csv
import getpass
import glob
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import taxonomy_v2 as tx  # noqa: E402
except ImportError:
    sys.exit("taxonomy_v2.py is required but not included in this repository. "
             "Place your taxonomy file next to this script.")

# ===== SETTINGS =====
MAX_CONCERNS = 25          # small test first; use a big number (e.g. 5000) for the full run
SIMILARITY = 0.45          # 0-1; higher = stricter groups (more, smaller groups)
MIN_GROUP_SIZE = 3         # groups smaller than this are marked "needs more examples"
CHAT_MODEL = "gpt-5.6-luna"
EMBED_MODEL = "text-embedding-3-small"
OUT_GROUPS = "discovered_categories.csv"
OUT_ASSIGN = "concern_assignments.csv"
# ====================

API = "https://api.openai.com/v1/"


def post(api_key, endpoint, body):
    req = urllib.request.Request(API + endpoint, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 3:
                time.sleep(10)
                continue
            raise RuntimeError(f"network error: {e}")


def embed(api_key, texts):
    vecs = []
    for i in range(0, len(texts), 100):
        data = post(api_key, "embeddings", {"model": EMBED_MODEL, "input": texts[i:i + 100]})
        vecs += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    return vecs


def chat_json(api_key, system, user):
    data = post(api_key, "chat/completions", {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_completion_tokens": 3000,
        "response_format": {"type": "json_object"},
    })
    return json.loads(data["choices"][0]["message"]["content"])


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def cluster(vecs, threshold):
    """average-linkage agglomerative clustering on cosine similarity"""
    n = len(vecs)
    sim = [[cosine(vecs[i], vecs[j]) for j in range(n)] for i in range(n)]
    groups = [[i] for i in range(n)]
    while True:
        best, pair = -1.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                s = sum(sim[i][j] for i in groups[a] for j in groups[b]) / (len(groups[a]) * len(groups[b]))
                if s > best:
                    best, pair = s, (a, b)
        if pair is None or best < threshold:
            break
        a, b = pair
        groups[a] += groups[b]
        del groups[b]
    return sorted(groups, key=len, reverse=True)


def load_concerns():
    items, ids_from_newer = [], set()
    for path in sorted(glob.glob("reddit_labels_v*.jsonl"), reverse=True):   # newest prompt first
        ids_here = set()
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") != "ok" or r["id"] in ids_from_newer:   # use the newest label of each comment
                continue
            ids_here.add(r["id"])
            res = r["result"]
            for o in res.get("outside_taxonomy_concerns") or res.get("needs_outside_taxonomy") or []:
                desc = str(o.get("description") or "").strip()
                if not desc:
                    continue
                items.append({"id": r["id"], "city": r["city"], "comment": r["body"], "description": desc,
                              "evidence": o.get("evidence", ""), "location": o.get("location", o.get("target", "")),
                              "source_file": path})
        ids_from_newer |= ids_here
    return items


NAME_SYSTEM = """You name groups of transit rider concerns. Given a numbered list of concerns,
return JSON: {"name": "<snake_case category name>", "definition": "<one sentence>"}"""

REVIEW_SYSTEM = f"""You are helping decide whether a transit-comment taxonomy needs a NEW category.
Below is the EXISTING taxonomy. You will get a group of rider concerns that a first model could
not fit into it.

{tx.taxonomy_prompt_text()}

For the group, answer these questions honestly:
1. coherent: do all concerns in the group describe the SAME kind of transit problem?
   If not, propose a split (which concern numbers belong together).
2. is_transit_service_concern: is this something a transit agency could act on (service, vehicles,
   stops, staff, fares, safety on the system)? Car traffic, general road design, or urban-planning
   opinions with no transit action are NOT.
3. closest_existing: the closest existing "category / subtype", and overlap = none | partial | duplicate.
4. decision:
   NEW_CATEGORY_CANDIDATE - a distinct, recurring transit concern not covered by any subtype
   MAP_TO_EXISTING        - an existing subtype already covers it (the first model missed it)
   NOT_A_TRANSIT_CONCERN  - not something a transit agency acts on
   TOO_VAGUE              - too unclear to define a category
   Prefer MAP_TO_EXISTING over creating a near-duplicate. Do not create a category just to be complete.
5. member_checks: for each concern number, the existing "category / subtype" it fits, or "none".

Return JSON:
{{"coherent": true, "split": [{{"name": "", "members": [1, 2]}}],
  "is_transit_service_concern": true,
  "proposed_name": "<snake_case>", "definition": "<one sentence>",
  "closest_existing": "<category / subtype or none>", "overlap": "none|partial|duplicate",
  "decision": "", "reason": "<1-2 sentences>",
  "member_checks": [{{"n": 1, "fits_existing": "<category / subtype or none>"}}]}}"""

MERGE_SYSTEM = """You review proposed new categories for a transit-comment taxonomy and remove redundancy.
Given candidates with definitions, return JSON:
{"merge_groups": [{"keep_name": "", "merge": ["name_a", "name_b"], "reason": ""}],
 "final_candidates": [{"name": "", "definition": ""}]}
Merge only candidates that a transit planner would treat as the same problem."""


def numbered(members):
    return "\n".join(f"{k}. {m['description']}  (evidence: \"{m['evidence']}\")" for k, m in enumerate(members, 1))


def main():
    concerns = load_concerns()[:MAX_CONCERNS]
    if not concerns:
        sys.exit("No outside-taxonomy concerns found. Run label_reddit.py first.")
    print(f"Concerns collected: {len(concerns)}")

    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("Paste your OpenAI API key (hidden): ").strip()
    if not api_key:
        sys.exit("No API key given.")

    print("Embedding concerns...")
    vecs = embed(api_key, [c["description"] for c in concerns])
    groups = cluster(vecs, SIMILARITY)
    print(f"Groups formed: {len(groups)} (sizes: {[len(g) for g in groups]})\n")

    group_rows, assign_rows, candidates = [], [], []
    for gi, idx in enumerate(groups, 1):
        members = [concerns[i] for i in idx]
        listing = numbered(members)
        e3 = chat_json(api_key, NAME_SYSTEM, listing)                      # E3 baseline
        e4 = chat_json(api_key, REVIEW_SYSTEM, f"Group of {len(members)} concerns:\n{listing}")   # E4
        decision = str(e4.get("decision") or "").upper()
        size_note = "" if len(members) >= MIN_GROUP_SIZE else "needs more examples"
        print(f"Group {gi} ({len(members)}): E3 = {e3.get('name')} | E4 = {decision} -> "
              f"{e4.get('proposed_name')} (closest: {e4.get('closest_existing')}, overlap {e4.get('overlap')})")
        if decision == "NEW_CATEGORY_CANDIDATE":
            candidates.append({"name": e4.get("proposed_name"), "definition": e4.get("definition"),
                               "group": gi, "size": len(members)})
        split = "; ".join(f"{s.get('name')}: {s.get('members')}" for s in (e4.get("split") or []) if s.get("members"))
        group_rows.append([gi, len(members), size_note, e3.get("name"), e3.get("definition"), decision,
                           e4.get("proposed_name"), e4.get("definition"), e4.get("closest_existing"),
                           e4.get("overlap"), e4.get("coherent"), split, e4.get("is_transit_service_concern"),
                           e4.get("reason"), " || ".join(m["description"] for m in members[:5])])
        checks = {int(c.get("n", 0)): c.get("fits_existing", "") for c in (e4.get("member_checks") or [])
                  if str(c.get("n", "")).isdigit()}
        for k, m in enumerate(members, 1):
            assign_rows.append([gi, decision, m["id"], m["city"], m["description"], m["evidence"],
                                m["location"], checks.get(k, ""), m["comment"]])

    merges = {}
    if len(candidates) > 1:
        print("\nChecking new-category candidates for redundancy...")
        listing = "\n".join(f"- {c['name']} ({c['size']} concerns): {c['definition']}" for c in candidates)
        mres = chat_json(api_key, MERGE_SYSTEM, listing)
        for m in mres.get("merge_groups") or []:
            for name in m.get("merge") or []:
                if name != m.get("keep_name"):
                    merges[name] = m.get("keep_name")
                    print(f"  merge {name} -> {m.get('keep_name')}: {m.get('reason')}")

    with open(OUT_GROUPS, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "size", "size_note", "E3_name", "E3_definition", "E4_decision", "E4_name",
                    "E4_definition", "closest_existing_subtype", "overlap", "coherent", "suggested_split",
                    "is_transit_concern", "E4_reason", "example_concerns", "merge_into"])
        for r in group_rows:
            w.writerow(r + [merges.get(r[6], "")])
    with open(OUT_ASSIGN, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "group_decision", "id", "city", "concern", "evidence", "location",
                    "fits_existing_subtype", "comment"])
        w.writerows(assign_rows)

    dec = [r[5] for r in group_rows]
    print(f"\nSummary over {len(groups)} groups ({len(concerns)} concerns):")
    for d in ("NEW_CATEGORY_CANDIDATE", "MAP_TO_EXISTING", "NOT_A_TRANSIT_CONCERN", "TOO_VAGUE"):
        print(f"  {d:24} {dec.count(d)}")
    print(f"\nSaved {OUT_GROUPS} and {OUT_ASSIGN}")


if __name__ == "__main__":
    main()
