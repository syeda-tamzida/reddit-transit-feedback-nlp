# Open-World Transit Feedback from Reddit

Collecting, filtering, and labeling Reddit comments about public transit from many cities, so that a
model can learn **what riders ask for, where they need it, and which concerns fall outside a fixed
category list**.

Course project for CS554 Natural Language Processing (Prof. Xiaozhong Liu), Worcester Polytechnic
Institute, Fall 2026. Work in progress.

## Motivation

Transit agencies receive large volumes of rider feedback, and LLM pipelines can now classify it into
request types. Two gaps remain:

1. **Generality.** Models built on one agency's comments learn that city's place names and writing
   style. Reddit discussions from many cities add informal language and different networks.
2. **Open-world concerns.** A fixed category list misses needs riders actually raise (for example
   personal safety, driver behaviour, or fares). These should be captured and reviewed, not forced
   into the nearest category.

## Pipeline

```
Pushshift Reddit comments (Hugging Face)
        |  1. get_reddit.py          select subreddits + transit keywords, month by month
        v
reddit_transit_all.csv  (41,907 comments)
        |  2. filter_relevance.py    GPT screen: transit request or concern?  KEEP / DROP
        v
reddit_transit_concerns.csv
        |  3. label_reddit.py        GPT labels: status, request types, evidence quote,
        |                            location + location role, outside-taxonomy concerns
        v
reddit_labels_v3.jsonl  ->  filter_feedback.py  (in taxonomy / outside taxonomy / dropped)
        |
        +-- 4. score_reddit_eval.py     compare GPT labels with blind human coding
        +-- 5. discover_categories.py   group outside-taxonomy concerns into candidate new categories
```

## Scripts

| Step | Script | What it does |
|---|---|---|
| 0 | `list_months.py` | Lists the monthly files available in the Pushshift dataset |
| 1 | `get_reddit.py` | Downloads one month at a time, keeps selected subreddits and comments matching transit keywords, deletes the raw file |
| 1 | `combine.py` | Merges monthly files, removes duplicates, rejoins split punctuation |
| 1 | `check_results.py` | Quick look at a monthly file (counts, sample comments) |
| 2 | `filter_relevance.py` | Short, low-token GPT screen, 20 comments per call; keeps anything with a transit request or concern, and keeps when unsure |
| 3 | `label_reddit.py` | Detailed labeling: feedback status (IN_TAXONOMY / OUTSIDE_TAXONOMY / NOT_RELEVANT_FEEDBACK), one request per distinct concern, verbatim evidence quote, location and location role |
| 3 | `filter_feedback.py` | Splits labeled comments into in-taxonomy, outside-taxonomy, and dropped |
| 4 | `score_reddit_eval.py` | Precision / recall / F1 for feedback detection, request types, request-location pairs, and outside-taxonomy concerns against blind human coding |
| 5 | `discover_categories.py` | Embeds outside-taxonomy concerns, clusters them, and compares a plain naming baseline with a taxonomy-aware review that decides new category / existing category / not a transit concern, then merges near-duplicates |

### Design choices

- **Evidence-grounded extraction.** Every extracted request must quote the comment. Quotes are checked
  against the text; unsupported items are flagged for review, not silently dropped.
- **Three-way status.** Off-topic text is separated from real concerns that the taxonomy does not cover,
  so new concerns are kept for category discovery.
- **Request-location pairs.** Each request carries its own location as written and a role
  (problem location, origin, destination, requested location, route, systemwide).
- **Versioned prompts.** Each prompt version writes its own output files so results stay comparable.
- **Standard-library HTTP.** Scripts that call the OpenAI API use only the Python standard library.

## Data

- Source: [`fddemarco/pushshift-reddit-comments`](https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments)
  on Hugging Face (monthly Parquet files, January 2012 to February 2016).
- Subreddits: r/pittsburgh, r/transit, r/seattle, r/chicago, r/boston, r/nyc, r/sanfrancisco,
  r/bayarea, r/washingtondc, r/philadelphia, r/portland, r/toronto, r/losangeles, r/urbanplanning.
- Months used so far: 2015-03, 2015-06, 2015-09, 2015-12, 2016-01, 2016-02
  (2016-02 covers only r/pittsburgh and r/transit).
- The text in this dataset is lowercase with punctuation split into tokens; usernames are not kept.

**No data, labels, or human annotations are included in this repository.**

## Results so far

| Stage | Result |
|---|---|
| Collection | 41,907 keyword-matched comments from 14 subreddits |
| Keyword exploration | Keyword rules alone are a poor relevance filter: in a hand-read sample, roughly one third of keyword-flagged comments were real rider feedback |
| Relevance filter (200-comment test) | 92 of 200 kept (46%); 48 of 60 decisions agree with the detailed labels; most disagreements were cases where the detailed labels were too generous |
| Detailed labeling (60-comment evaluation set) | 19 in taxonomy, 12 outside taxonomy, 29 not relevant; 45 of 45 evidence quotes found in the comment text |
| Outside-taxonomy concerns | Recurring themes include driver behaviour, personal safety at stops, fare payment, and system condition |

Human coding of the 60-comment evaluation set and the full relevance run are in progress.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
huggingface-cli login            # or: hf auth login  (needed to avoid rate limits when downloading)
```

Scripts that call GPT read the API key from the `OPENAI_API_KEY` environment variable or ask for it
with a hidden prompt. Never hard-code a key.

`label_reddit.py`, `score_reddit_eval.py`, and `discover_categories.py` require a taxonomy definition
file, `taxonomy_v2.py`, which is not part of this repository.

## Next steps

1. Run the relevance filter on all 41,907 comments.
2. Label a random sample (about 2,000) of the kept comments in detail.
3. Finish blind human coding and report evaluation metrics.
4. Run taxonomy-aware category discovery on the larger set of outside-taxonomy concerns.
5. Fine-tune a small open model (LoRA) on the labeled data and compare it with the base model and GPT.

## Limitations

- The Pushshift snapshot used here ends in February 2016, and the text is lowercased, which removes
  capitalization cues for place names.
- Training and evaluation labels come partly from GPT; human-coded comments are used only for testing.
- Reddit is not a representative sample of transit riders.

## Author

Syeda Tamzida Akter, Department of Civil, Environmental and Architectural Engineering,
Worcester Polytechnic Institute.
