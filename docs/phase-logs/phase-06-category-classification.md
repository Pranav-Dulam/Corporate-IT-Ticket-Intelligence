# Phase 6 — Category Classification

**Status:** Complete
**Outputs:** `models/category_model_{raw,stripped}.joblib`,
`data/processed/category_{metrics,per_class,confusion}_{raw,stripped}.csv`

---

## What we set out to build

A baseline classifier that predicts a ticket's category from its
description text, trained on the `jira_project` category scheme — the only
scheme in `unified_tickets.csv` with both real text and usable labels.

Phase 4's log had already named the constraint this phase inherits:

> Phase 6 (category) has 23,421 usable rows, but ~23,064 of them are
> JOSSE. A model trained on this predicts *which Apache project a bug
> report belongs to* — real machine learning, wrong domain for an IT
> helpdesk system.

That is exactly what was built. The value of this phase is the method and
the honest measurement, not the label vocabulary.

---

## Configuration

```
SCHEME          jira_project
MIN_CLASS_SIZE  10          # classes below this are dropped
MAX_WORDS       512         # description cap before vectorizing
TEST_SIZE       0.2         # stratified
RANDOM_SEED     42
model           TF-IDF (50,000 features) -> LogisticRegression
```

`MAX_WORDS = 512` is a direct answer to Phase 4's closing warning about the
single 57,662-word Jira issue with a stack trace pasted into it. Without the
cap that one row distorts the TF-IDF vocabulary.

---

## Data funnel

```
rows in scheme 'jira_project'          23,068
duplicate descriptions removed             38
rows under 3 words after stripping          9   (stripped variant only)
rows in classes with < 10 examples        688

final training set                     22,333 rows, 138 classes
  train                                17,866
  test                                  4,467
```

The raw variant keeps the 9 short rows, so its test set is 4,469 rather
than 4,467. The two variants are otherwise trained on identical splits.

---

## The leakage question

Ticket descriptions in JOSSE frequently contain the project name and issue
key inline — `CARBONDATA-1234`, `see AMBARI-5678`. A classifier predicting
`project_key` from text containing the project key is not classifying, it
is reading the answer off the page.

`build_label_pattern()` compiles an alternation over every label, longest
first, matching an optional `-\d+` issue suffix:

```python
re.compile(rf"\b(?:{alternation})(?:-\d+)?\b", flags=re.IGNORECASE)
```

`strip_project_names()` applies it identically to train, test, and any
future input. This matters: it removes a shortcut uniformly rather than
removing information from one split only, which would be its own kind of
cheating.

Running both variants turns leakage from a suspicion into a measurement.

---

## Results

| Metric | raw | stripped | Δ |
|---|---|---|---|
| accuracy | 0.6876 | 0.5767 | −0.111 (−16%) |
| **macro F1** | **0.5237** | **0.4082** | **−0.116 (−22%)** |
| weighted F1 | 0.7137 | 0.6104 | −0.103 (−15%) |
| macro precision | 0.5081 | 0.3876 | −0.121 (−24%) |
| macro recall | 0.6042 | 0.4965 | −0.108 (−18%) |

**About a fifth of the raw model's apparent skill was leakage.** The
stripped macro F1 of **0.4082** is the number this phase reports.

The effect is broad, not driven by a few classes:

```
classes worse after stripping     114
classes better                      7
classes unchanged                  17   (all were F1 = 0.0 in both)
mean per-class F1 delta        -0.1155
```

Largest drops among classes with meaningful test support:

```
NETBEANS    support=  63   0.821 -> 0.630   -0.191
ACCUMULO    support= 252   0.742 -> 0.554   -0.188
STORM       support= 185   0.857 -> 0.669   -0.188
GEODE       support= 277   0.732 -> 0.552   -0.180
AMBARI      support= 363   0.851 -> 0.721   -0.130
FLINK       support= 142   0.713 -> 0.587   -0.125
```

No class with support ≥ 50 improved. The shortcut was being used
everywhere it was available.

---

## Why macro F1 is so much lower than accuracy

The class distribution is severely imbalanced, and the two metrics
disagree for a structural reason worth stating plainly.

```
classes                          138
median test support              8.5
min / max test support          2 / 371
top 10 classes                  2,526 test rows (56.5% of the test set)
classes with test F1 = 0.0        17 (12% of classes, 1.1% of test rows)
classes with test support <= 5    48
```

Every one of the 17 classes scoring exactly 0.0 has a test support of 2–4.
That is a consequence of `MIN_CLASS_SIZE = 10` combined with a 20% split: a
class with exactly 10 examples contributes 8 to train and 2 to test, which
is not enough for logistic regression to ever prefer it over a large
neighbour. Those classes were never winnable, and they drag macro F1 down
while barely touching accuracy.

Accuracy of 0.577 is therefore flattered by the ten projects that make up
over half the test set. Macro F1 of 0.408 is the honest summary — and it
is still far above the ~0.007 a random guess over 138 classes would give.

Note also that macro precision (0.388) fell further than macro recall
(0.497): the stripped model over-predicts the large projects.

---

## A bug found and fixed during this phase

The first run of `train_category_model.py` failed immediately:

```
FileNotFoundError: [Errno 2] No such file or directory:
'data/processed/unified_tickets.csv'
```

The file existed. The problem was that every script in `src/` declared its
paths relative to the current working directory:

```python
INPUT_PATH = Path("data/processed/unified_tickets.csv")
```

A relative path resolves against wherever Python was launched, not against
where the script lives, so the pipeline silently depended on always being
run from the repo root. All seven scripts were changed to anchor on the
file's own location:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data/processed/unified_tickets.csv"
```

This matters for outputs as much as inputs: `MODEL_DIR` and `REPORT_DIR`
were relative too, so a run launched from a subdirectory would have written
a 57 MB model into the wrong folder without complaint.

**Lesson carried forward, and it rhymes with Phase 4's:** a script that
fails loudly at the wrong moment is cheaper than one that succeeds in the
wrong place.

---

## Honest assessment

The pipeline is correct and the measurement is trustworthy. The labels are
not the ones this project ultimately needs.

- **What this model actually does** is route an Apache bug report to its
  originating project. It is a genuine text classifier with a defensible
  evaluation, and it demonstrates the full path from unified table to saved
  artifact — but a user of an IT helpdesk system will never type a sentence
  whose correct answer is `CARBONDATA`.
- **The leakage check is the transferable result.** The raw/stripped
  comparison is the part of this phase that should survive into the final
  write-up, because it is the difference between reporting 0.69 and
  reporting 0.41.
- **`MIN_CLASS_SIZE = 10` is too permissive.** 48 of 138 classes have five
  or fewer test examples. Raising the floor to 30 would produce a smaller,
  more defensible class set and a macro F1 that measures something real
  rather than averaging in dozens of foregone conclusions.
- **`models/*.joblib` is gitignored**, correctly — the two models are
  57 MB each. They exist only locally and must be retrained from the
  committed scripts.

---

## Files

**Created**
- `models/category_model_raw.joblib`
- `models/category_model_stripped.joblib`
- `data/processed/category_metrics_raw.csv`
- `data/processed/category_metrics_stripped.csv`
- `data/processed/category_per_class_raw.csv`
- `data/processed/category_per_class_stripped.csv`
- `data/processed/category_confusion_raw.csv`
- `data/processed/category_confusion_stripped.csv`
- `docs/phase-logs/phase-06-category-classification.md`

**Modified**
- `src/ml/train_category_model.py` — anchor paths to `PROJECT_ROOT`
- `src/data/build_unified_dataset.py` — same
- `src/data/clean_josse.py` — same
- `src/data/clean_mendeley.py` — same
- `src/data/clean_servicenow.py` — same
- `src/data/explore_unified.py` — same
- `src/data/profile_mendeley.py` — same
