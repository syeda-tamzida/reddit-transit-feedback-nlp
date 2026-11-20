"""Label Reddit transit comments with a hierarchical transit-request taxonomy (prompt version v3).

The taxonomy definitions (taxonomy_v2.py) are not included in this repository.

For every comment GPT returns:
  * feedback_status  - IN_TAXONOMY, OUTSIDE_TAXONOMY or NOT_RELEVANT_FEEDBACK
  * city_context     - which city/agency the comment is about
  * service_requests - one object per DISTINCT concern: category + subtype + evidence quote
                       + the location it applies to and that location's role
  * outside_taxonomy_concerns - real transit concerns that fit no subtype (kept for review)

After GPT answers, every evidence quote is checked against the comment text. Requests whose
quote cannot be found are NOT deleted - they are flagged (evidence_found = False) for review.

v3 changes (after the 20-comment check): three feedback statuses, every request needs a quote
copied from the comment, no inferred solutions or consequences, a location_role per request,
and a prompt_version stored with every result. Each prompt version writes its own files, so
earlier results are kept for comparison.

Standard library only (no pandas / openai package), so Windows Application Control does not block it.

Run:
    python label_reddit.py
Outputs (resumable - rerun to continue where it stopped):
    reddit_labels_v3.jsonl        full answer per comment
    reddit_request_pairs_v3.csv   one row per request or outside concern
"""
import csv
import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import taxonomy_v2 as tx  # noqa: E402
except ImportError:
    sys.exit("taxonomy_v2.py is required but not included in this repository. "
             "Place your taxonomy file next to this script.")

# ===== SETTINGS =====
PROMPT_VERSION = "v3"
INPUT_FILE = "pilot_300.csv"
# Which rows of pilot_300.csv to label (row 0 = first comment).
# Rows 0-19 were used to tune the prompt, so rows 20-79 are a SEPARATE evaluation set.
# These 60 are the comments in the hand-coding sheet. Later, use ROW_START = 0, ROW_END = 300.
ROW_START = 20
ROW_END = 80
MODEL = "gpt-5.6-luna"
OUTPUT_JSONL = f"reddit_labels_{PROMPT_VERSION}.jsonl"
OUTPUT_PAIRS = f"reddit_request_pairs_{PROMPT_VERSION}.csv"
# ====================

STATUSES = ("IN_TAXONOMY", "OUTSIDE_TAXONOMY", "NOT_RELEVANT_FEEDBACK")
LOCATION_ROLES = ("problem_location", "origin", "destination", "requested_location",
                  "route", "systemwide", "unspecified")
NO_REQUEST = "no_actionable_transit_request"

SYSTEM_PROMPT = f"""You analyze comments about public transportation written by people in ANY city.
These comments come from Reddit. The text is all lowercase and punctuation was re-joined
automatically, so place names are not capitalized.

Your job is to identify transportation problems, needs, and requested improvements that are
DIRECTLY SUPPORTED by the comment text - nothing more.

=== RULES ===
RULE 1 - DO NOT INVENT REQUESTS.
  Do not infer a request merely because the comment mentions a city, bus, route, station,
  transit system, a trip, housing near transit, or transit policy.
RULE 2 - REQUIRE TEXTUAL EVIDENCE.
  Every request and every outside-taxonomy concern must have an "evidence" phrase copied
  WORD FOR WORD from the comment (a short continuous span, 3-25 words). If you cannot quote
  supporting text, do not output that item.
RULE 3 - A CONCERN DOES NOT NEED POLITE "PLEASE" WORDING.
  "the bus keeps passing me even though i'm waiting at the stop" is real rider feedback.
  "i moved here from oklahoma three years ago" is not. General debate about transit policy,
  funding, economics, history, technology, or what other cities do is NOT rider feedback unless
  the commenter states a concrete problem they experience or a specific change they want.
RULE 4 - DO NOT OVER-CLASSIFY.
  Each request must be a DISTINCT transportation problem or requested action with its OWN
  supporting phrase. Do not add consequences or possible solutions the commenter did not ask
  for (e.g. from "the buses are always late, making me late for work" extract improve_reliability
  only - not employment access, not express service). "Buses and trains are late" is ONE request.
RULE 5 - SARCASM AND JOKES: extract the underlying concern only if it is clearly stated.

=== STEP 1: feedback_status ===
IN_TAXONOMY           = at least one concern fits a subtype in the taxonomy below.
OUTSIDE_TAXONOMY      = a real transportation concern exists, but NONE of it fits any subtype
                        (e.g. personal safety or crime on vehicles/at stops, fare cost or payment,
                        driver or staff behaviour, cleanliness). Do NOT force it into the closest subtype.
NOT_RELEVANT_FEEDBACK = no identifiable transportation problem, need, or request.
A comment can be IN_TAXONOMY and still list extra outside_taxonomy_concerns.
If NOT_RELEVANT_FEEDBACK: return empty lists for service_requests and outside_taxonomy_concerns.

=== STEP 2: city_context ===
The city or transit agency the comment is about (use the subreddit as a hint), or "Unspecified".

=== STEP 3: service_requests (taxonomy) ===
Use ONLY actionable categories below. Do not use no_actionable_transit_request on Reddit.

{tx.taxonomy_prompt_text()}

=== STEP 4: where each concern applies ===
For every request and outside concern give:
- location: the exact stop, station, street, corridor, neighborhood, municipality, route or line
  the concern applies to, copied AS WRITTEN (lowercase is fine). "Unspecified" if none is stated.
  Generic words ("buses", "trains", "the metro", "bus stops", "drivers") are NOT locations.
  Do not use the city name unless the concern is truly citywide.
- location_role: one of
    problem_location   - where the problem happens ("packed by the time it gets to oakland")
    origin             - where the trip starts
    destination        - where the trip should go
    requested_location - where a new thing should be added ("put a shelter at forbes and murray")
    route              - the location is a route or line itself
    systemwide         - the concern applies to the whole system
    unspecified        - no location stated
- target_kind: stop, poi, transfer_point, street, corridor, neighborhood, municipality, route,
  systemwide, or unspecified.
- route: route or line identifier as written (e.g. "71a", "red line", "the 36"), or "Unspecified".
- related_places: pipe-separated other places mentioned for context, or "Unspecified".

proposal_stance: use not_applicable unless the comment clearly supports or opposes a named plan.

Respond with ONLY this JSON object:
{{
  "feedback_status": "IN_TAXONOMY | OUTSIDE_TAXONOMY | NOT_RELEVANT_FEEDBACK",
  "city_context": "",
  "service_requests": [
    {{"category": "", "subtype": "", "evidence": "", "location": "", "location_role": "",
      "target_kind": "", "route": "", "related_places": "", "requested_change": "", "confidence": ""}}
  ],
  "outside_taxonomy_concerns": [
    {{"description": "", "evidence": "", "location": "", "location_role": ""}}
  ],
  "proposal_stance": "",
  "access_needs": [{{"subtype": "", "evidence": ""}}],
  "label_confidence": "low | medium | high",
  "explanation": "1-3 plain sentences"
}}"""


def norm(text):
    """lowercase, letters/digits only, single spaces - used to compare quotes with the comment"""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split())


def evidence_found(evidence, comment):
    e = norm(evidence)
    return bool(e) and e in norm(comment)


def call_gpt(api_key, subreddit, comment):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Subreddit: r/{subreddit}\nComment: {comment}"},
        ],
        "max_completion_tokens": 3000,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return json.loads(data["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < 3:
                wait = 10 * (attempt + 1)
                print(f"   server busy ({e.code}), waiting {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 3:
                time.sleep(10)
                continue
            raise RuntimeError(f"network error: {e}")


def role(value):
    v = tx._key(value)
    return v if v in LOCATION_ROLES else "unspecified"


def clean(result, comment):
    status = str(result.get("feedback_status") or "").strip().upper().replace(" ", "_")
    if status not in STATUSES:
        status = "NOT_RELEVANT_FEEDBACK"

    requests = []
    for item in result.get("service_requests") or []:
        if not isinstance(item, dict):
            continue
        cat, sub, valid, note = tx.resolve_category_subtype(item.get("category"), item.get("subtype"))
        if cat == NO_REQUEST:
            continue
        ev = tx._txt(item.get("evidence"), "")
        requests.append({
            "category": cat, "subtype": sub, "taxonomy_valid": valid, "taxonomy_note": note,
            "evidence": ev, "evidence_found": evidence_found(ev, comment),
            "location": tx._txt(item.get("location") or item.get("target")),
            "location_role": role(item.get("location_role")),
            "target_kind": tx._txt(item.get("target_kind"), "unspecified").lower(),
            "route": tx._txt(item.get("route")),
            "related_places": tx._txt(item.get("related_places")),
            "requested_change": tx._txt(item.get("requested_change"), ""),
            "confidence": tx._txt(item.get("confidence"), "medium").lower(),
        })

    outside = []
    for o in result.get("outside_taxonomy_concerns") or []:
        if not isinstance(o, dict) or not str(o.get("description") or "").strip():
            continue
        ev = tx._txt(o.get("evidence"), "")
        outside.append({
            "description": tx._txt(o.get("description"), ""),
            "evidence": ev, "evidence_found": evidence_found(ev, comment),
            "location": tx._txt(o.get("location")),
            "location_role": role(o.get("location_role")),
        })

    # consistency check between the status and what was extracted (flagged, not changed)
    if status == "NOT_RELEVANT_FEEDBACK":
        consistent = not requests and not outside
    elif status == "IN_TAXONOMY":
        consistent = any(r["taxonomy_valid"] for r in requests)
    else:
        consistent = bool(outside)

    return {
        "feedback_status": status,
        "status_consistent": consistent,
        "city_context": tx._txt(result.get("city_context")),
        "service_requests": requests,
        "outside_taxonomy_concerns": outside,
        "proposal_stance": tx.normalize_stance(result.get("proposal_stance")),
        "access_needs": tx.normalize_access_needs(result.get("access_needs")),
        "label_confidence": tx._txt(result.get("label_confidence"), "medium").lower(),
        "explanation": tx._txt(result.get("explanation"), ""),
    }


def write_pairs():
    cols = ["id", "subreddit_city", "comment", "feedback_status", "status_consistent", "city_context",
            "kind", "category", "subtype_or_description", "evidence", "evidence_found",
            "location", "location_role", "target_kind", "route", "related_places",
            "requested_change", "confidence", "taxonomy_valid"]
    n = 0
    stats = Counter()
    with open(OUTPUT_PAIRS, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for line in open(OUTPUT_JSONL, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            r = rec["result"]
            stats[r["feedback_status"]] += 1
            if not r["status_consistent"]:
                stats["inconsistent"] += 1
            base = [rec["id"], rec["city"], rec["body"], r["feedback_status"], r["status_consistent"],
                    r["city_context"]]
            items = [("request", s) for s in r["service_requests"]] + \
                    [("outside_taxonomy", o) for o in r["outside_taxonomy_concerns"]]
            if not items:
                w.writerow(base + ["none"] + [""] * (len(cols) - len(base) - 1))
                n += 1
            for kind, s in items:
                stats["items"] += 1
                stats["evidence_found"] += bool(s["evidence_found"])
                w.writerow(base + [kind, s.get("category", "OUTSIDE_TAXONOMY"),
                                   s.get("subtype", s.get("description", "")), s["evidence"],
                                   s["evidence_found"], s["location"], s["location_role"],
                                   s.get("target_kind", ""), s.get("route", ""), s.get("related_places", ""),
                                   s.get("requested_change", ""), s.get("confidence", ""),
                                   s.get("taxonomy_valid", "")])
                n += 1
    return n, stats


def main():
    if "--show-prompt" in sys.argv:
        print(SYSTEM_PROMPT)
        return

    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("Paste your OpenAI API key (hidden): ").strip()
    if not api_key:
        sys.exit("No API key given.")

    with open(INPUT_FILE, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))[ROW_START:ROW_END]

    done = set()
    if os.path.exists(OUTPUT_JSONL):
        for line in open(OUTPUT_JSONL, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("status") == "ok" and rec.get("prompt_version") == PROMPT_VERSION:
                done.add(rec["id"])

    todo = [r for r in rows if r["id"] not in done]
    print(f"Prompt {PROMPT_VERSION}: rows {ROW_START}-{ROW_END - 1} = {len(rows)} comments, "
          f"{len(rows) - len(todo)} already done, {len(todo)} to label.")

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {row['city']}: {row['body'][:70]}...")
            rec = {"id": row["id"], "city": row["city"], "subreddit": row["subreddit"], "body": row["body"],
                   "prompt_version": PROMPT_VERSION, "model": MODEL}
            try:
                rec["result"] = clean(call_gpt(api_key, row["subreddit"], row["body"]), row["body"])
                rec["status"] = "ok"
                r = rec["result"]
                items = [f"{s['subtype']} @ {s['location']}" + ("" if s["evidence_found"] else " [quote not found]")
                         for s in r["service_requests"]]
                items += [f"OUTSIDE: {o['description'][:40]} @ {o['location']}" for o in r["outside_taxonomy_concerns"]]
                print("   ->", r["feedback_status"], "|", "; ".join(items)[:160])
            except Exception as e:
                rec["status"] = "error"
                rec["error"] = str(e)
                print("   ERROR:", e)
                if "HTTP 401" in str(e) or "HTTP 404" in str(e):
                    out.write(json.dumps(rec) + "\n")
                    sys.exit("Stopping: check your API key or the MODEL name in the settings.")
            out.write(json.dumps(rec) + "\n")
            out.flush()

    n, stats = write_pairs()
    print(f"\nDone. {n} rows written to {OUTPUT_PAIRS}")
    print(f"  IN_TAXONOMY: {stats['IN_TAXONOMY']}   OUTSIDE_TAXONOMY: {stats['OUTSIDE_TAXONOMY']}   "
          f"NOT_RELEVANT_FEEDBACK: {stats['NOT_RELEVANT_FEEDBACK']}")
    if stats["items"]:
        print(f"  Evidence quotes found in the comment: {stats['evidence_found']}/{stats['items']}")
    if stats["inconsistent"]:
        print(f"  Comments where status and extracted items disagree: {stats['inconsistent']} (check these)")


if __name__ == "__main__":
    main()
