# Phase 4 — Dataset Integration

**Status:** Complete
**Output:** `data/processed/unified_tickets.csv` (114,795 × 15)

---

## What we set out to build

Three datasets had already been cleaned independently — Mendeley Helpdesk,
ServiceNow (UCI), and JOSSE (Apache Jira). Phase 4's job was to merge them
into a single standardized table that every later phase could consume:

```
ticket_id | description | category | priority | resolution | status | source
```

The instruction was explicit: inspect first, propose a mapping, verify the
mapping, and only then merge. Do not blindly concatenate columns.

That ordering turned out to matter more than expected.

---

## What the inspection found

Before writing any merge code, each cleaned file was inspected for the
fields the pipeline actually needs. The result was not what the project
plan assumed.

| Dataset | Rows | Description text | Resolution text |
|---|---|---|---|
| Mendeley | 66,691 | none | none |
| ServiceNow | 24,918 | none | none |
| JOSSE | 23,186 | `corpus_clean` | none |

**Mendeley** is a process-mining dataset. All 62 columns are identifiers,
timestamps, and `wf_*` workflow durations. There is no subject or body
field anywhere in `issues.csv`, and `issues_snapshot.csv` has the same
60 columns with no text either.

**ServiceNow** is fully anonymized. `category` is the literal string
`"Category 42"`, `subcategory` is `"Subcategory 174"`, and the closure
reason is `"code 6"`. These work as structured labels, but no free-text
ticket can ever be mapped onto them — which makes them unusable for a
system whose entire input is a user typing a problem description.

**JOSSE** has real text, but it is Apache Jira development issues
(ZOOKEEPER, CARBONDATA, ARROW), not IT helpdesk tickets. It also carries
no priority labels and no resolution text — its `reference` column is
just the Jira URL.

Two of the four fields the pipeline depends on did not exist in any of
the three sources.

---

## How the problem was diagnosed

The gap was not visible from the cleaning reports. Each of the three
cleaning scripts had run successfully and produced sensible-looking
summaries — row counts preserved, duplicates checked, timestamps parsed,
negative values flagged.

The reason is a pattern shared by all three scripts:

```python
if column not in df.columns:
    continue
```

This is safe in the sense that it never crashes. It is unsafe in the sense
that a missing column produces a clean-looking output file with a silently
absent field. The absence only surfaces when something downstream tries to
use the field — which is exactly what happened, one phase later.

**Lesson carried forward:** a cleaning script should *assert* that the
columns its consumers depend on are present, and fail loudly when they are
not. The integration script added a `check_unique_ids` assertion for the
same reason.

---

## How it was solved

Rather than abandon the merge, the raw Mendeley folder was re-examined for
anything the cleaning step had skipped. `sample_utterances.csv` turned out
to contain 30,104 rows of genuine helpdesk conversation, keyed by `issueid`
and tagged by `author_role`:

- `reporter` turns describe the problem
- `assignee` turns describe what was actually done about it

That is a description and a resolution sitting side by side — the only
place in all three datasets where both exist. It covers 360 issues, of
which **357** successfully joined back to `mendeley_clean.csv`.

The integration script therefore does two things:

1. Maps each dataset onto the unified schema through its own adapter
   function
2. Enriches the Mendeley rows with reporter/assignee text where the
   utterances file covers them

---

## Design decisions

**Adapters, not a single mapping table.** Each dataset gets an independent
`adapt_*()` function. Adding a fourth dataset later means writing one more
function rather than rewriting the merge logic.

**Categories were deliberately NOT harmonized.** Mendeley's `issue_type`,
ServiceNow's `Category 42`, and JOSSE's `project_key` are three unrelated
label vocabularies. Flattening them into one column would produce a
classifier trained on labels that do not share a meaning. Instead every row
carries a `category_scheme` column naming its vocabulary, so Phase 6 can
train per-scheme.

**`closed_code` was not mapped onto `resolution`.** ServiceNow's
`closed_code` and Mendeley's `issue_resolution` (`Done`, `Duplicate`,
`Cannot Reproduce`, `Won't Do`) are closure *dispositions*, not
troubleshooting procedures. Putting them in the `resolution` field would
have made the coverage statistics look far better than the data warrants,
and would have fed meaningless context to the LLM in Phase 9.

**Usability flags are computed once, here.** `has_description`,
`has_usable_resolution`, `trainable_category`, `trainable_priority`, and
`retrievable` are written into the unified table rather than re-derived in
each later phase. Phase 8's requirement that tickets without a usable
resolution never pollute the retrieval context is enforced by the
`retrievable` flag.

---

## Results

```
total_rows             114795
rows_mendeley           66691
rows_servicenow         24918
rows_josse              23186

with_description        23421
with_usable_resolution    333
trainable_category      23421
trainable_priority        353
retrievable_for_rag       327
```

Priority distribution across the unified table:

```
<NA>    57151
P3      48254
P2       4962
P1       3010
P4       1418
```

Description length, for rows that have one:

```
mean       75.3 words
median     41.0 words
max      57662.0 words
```

---

## Honest assessment

Phase 4 succeeded as an engineering task and produced an accurate picture
of the data. That picture constrains what comes next:

- **Phase 6 (category)** has 23,421 usable rows, but ~23,064 of them are
  JOSSE. A model trained on this predicts *which Apache project a bug
  report belongs to* — real machine learning, wrong domain for an IT
  helpdesk system.
- **Phase 7 (priority)** has 353 rows across four classes. This will
  produce a weak model, and the evaluation should say so rather than
  disguise it.
- **Phase 9 (RAG)** has 327 grounded tickets. Enough to demonstrate the
  architecture end to end; not enough to claim coverage.

The single-row maximum description of 57,662 words is a Jira issue with a
large stack trace pasted in. Phase 6 should cap description length before
vectorizing, or that one row will distort the TF-IDF vocabulary.

---

## Files

**Created**
- `src/data/build_unified_dataset.py`
- `data/processed/unified_tickets.csv`
- `data/processed/unified_tickets_report.csv`
- `docs/phase-logs/phase-04-dataset-integration.md`

**Modified**
- `.gitignore` — exclude `data/interim/` bulk CSVs, keep the reports
