"""
Phase 5 - Exploratory Analysis and Feature Engineering.

Reads data/processed/unified_tickets.csv and answers the questions
Phase 6 needs settled before any model is trained:

  1. Which rows are actually usable for which model?
  2. How imbalanced are the label sets, and where should rare
     classes be cut off?
  3. What description length cap prevents one outlier row from
     dominating the TF-IDF vocabulary?
  4. Where could label information leak into the features?

This script only reads and reports. It writes summary CSVs but does
not modify unified_tickets.csv - feature construction happens in
Phase 6 inside the training pipeline, so that fitting never sees
the test split.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# Anchor all paths to the project root so the script works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data/processed/unified_tickets.csv"

SUMMARY_PATH = PROJECT_ROOT / "data/processed/eda_summary.csv"
CATEGORY_PATH = PROJECT_ROOT / "data/processed/eda_category_counts.csv"


# Class-size thresholds to evaluate. A class with fewer examples
# than this cannot be split into train and test and still be
# meaningfully evaluated.
MIN_CLASS_SIZES = [2, 5, 10, 20, 50]

SEPARATOR = "=" * 70


def section(title):
    print("\n" + SEPARATOR)
    print(title)
    print(SEPARATOR)


# ---------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------

def report_overview(df, summary):

    section("1. OVERVIEW")

    print(f"\nTotal rows: {len(df):,}")
    print(f"Columns:    {len(df.columns)}")

    print("\nRows by source:\n")
    print(df["source"].value_counts().to_string())

    print("\nUsability flags:\n")

    flags = [
        "has_description",
        "has_usable_resolution",
        "trainable_category",
        "trainable_priority",
        "retrievable",
    ]

    for flag in flags:
        count = int(df[flag].sum())
        percent = 100 * count / len(df)
        print(f"  {flag:<24} {count:>7,}  ({percent:5.2f}%)")
        summary.append({"metric": flag, "value": count})

    print("\nUsable rows by source:\n")

    usable_by_source = (
        df[df["has_description"]]["source"]
        .value_counts()
    )

    print(usable_by_source.to_string())

    return summary


# ---------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------

def report_missing(df):

    section("2. MISSING VALUES")

    missing = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "missing_pct": (100 * df.isna().mean()).round(2),
        }
    )

    missing = missing[missing["missing"] > 0]
    missing = missing.sort_values("missing", ascending=False)

    print()
    print(missing.to_string())

    print(
        "\nNote: high missingness here is structural, not a data "
        "quality fault.\nServiceNow has no description because the "
        "source is anonymized;\nJOSSE has no priority because the "
        "source never recorded one."
    )


# ---------------------------------------------------------
# 3. Category distribution and imbalance
# ---------------------------------------------------------

def report_categories(df, summary):

    section("3. CATEGORY DISTRIBUTION AND CLASS IMBALANCE")

    trainable = df[df["trainable_category"]]

    print(f"\nRows available for category training: {len(trainable):,}")

    print("\nBy scheme:\n")
    print(trainable["category_scheme"].value_counts().to_string())

    rows = []

    for scheme, group in trainable.groupby("category_scheme"):

        counts = group["category"].value_counts()

        print("\n" + "-" * 70)
        print(f"SCHEME: {scheme}")
        print("-" * 70)

        print(f"\n  rows:            {len(group):,}")
        print(f"  distinct classes: {len(counts):,}")
        print(f"  largest class:    {counts.iloc[0]:,} "
              f"({counts.index[0]})")
        print(f"  smallest class:   {counts.iloc[-1]:,} "
              f"({counts.index[-1]})")

        # Imbalance ratio: how many times bigger is the majority
        # class than the minority class.
        ratio = counts.iloc[0] / counts.iloc[-1]
        print(f"  imbalance ratio:  {ratio:,.0f} : 1")

        # What share of rows sit in the top 10 classes?
        top10_share = 100 * counts.head(10).sum() / len(group)
        print(f"  top-10 classes hold {top10_share:.1f}% of rows")

        print("\n  Coverage if rare classes are dropped:\n")

        for minimum in MIN_CLASS_SIZES:

            kept_classes = counts[counts >= minimum]
            kept_rows = int(kept_classes.sum())

            print(
                f"    >= {minimum:>3} examples: "
                f"{len(kept_classes):>4} classes, "
                f"{kept_rows:>7,} rows "
                f"({100 * kept_rows / len(group):5.1f}% kept)"
            )

        print("\n  Top 10 classes:\n")
        print(counts.head(10).to_string())

        for category, count in counts.items():
            rows.append(
                {
                    "scheme": scheme,
                    "category": category,
                    "count": count,
                }
            )

        summary.append(
            {
                "metric": f"{scheme}__classes",
                "value": len(counts),
            }
        )

    pd.DataFrame(rows).to_csv(CATEGORY_PATH, index=False)

    return summary


# ---------------------------------------------------------
# 4. Priority distribution
# ---------------------------------------------------------

def report_priority(df, summary):

    section("4. PRIORITY DISTRIBUTION")

    print("\nAcross the whole table:\n")
    print(df["priority"].value_counts(dropna=False).to_string())

    trainable = df[df["trainable_priority"]]

    print(f"\nRows usable for priority training: {len(trainable):,}")

    if len(trainable) == 0:
        print("\n[!] No rows have both a description and a priority.")
        return summary

    counts = trainable["priority"].value_counts()

    print("\nClass distribution in the trainable set:\n")

    for label, count in counts.items():
        percent = 100 * count / len(trainable)
        print(f"  {label}   {count:>5,}  ({percent:5.1f}%)")

    smallest = counts.min()

    print(f"\n  smallest class: {smallest} examples")

    # A stratified 80/20 split needs at least 5 examples per class
    # to place even one in the test fold.
    if smallest < 5:
        print(
            "\n[!] The smallest class cannot support a stratified\n"
            "    train/test split. Expect unstable metrics, or merge\n"
            "    it into an adjacent class."
        )

    print("\nSources contributing trainable priority rows:\n")
    print(trainable["source"].value_counts().to_string())

    summary.append(
        {
            "metric": "priority_smallest_class",
            "value": int(smallest),
        }
    )

    return summary


# ---------------------------------------------------------
# 5. Description length
# ---------------------------------------------------------

def report_description_length(df, summary):

    section("5. DESCRIPTION LENGTH")

    lengths = df.loc[df["has_description"], "description_word_count"]

    print("\nWord count distribution:\n")
    print(lengths.describe().round(1).to_string())

    print("\nPercentiles:\n")

    for q in [0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0]:
        value = lengths.quantile(q)
        print(f"  p{q * 100:6.1f}: {value:>10,.0f} words")

    # How much total text sits above each candidate cap?
    print("\nEffect of capping description length:\n")

    total_words = lengths.sum()

    for cap in [128, 256, 512, 1024]:

        over = lengths[lengths > cap]
        truncated_words = (over - cap).sum()

        print(
            f"  cap {cap:>5} words: "
            f"{len(over):>6,} rows affected "
            f"({100 * len(over) / len(lengths):5.2f}%), "
            f"{100 * truncated_words / total_words:5.2f}% of all text cut"
        )

    longest = df.loc[lengths.idxmax()]

    print(
        f"\nLongest description: {longest['description_word_count']:,} "
        f"words  ({longest['ticket_id']}, {longest['source']})"
    )

    summary.append(
        {
            "metric": "description_p99_words",
            "value": int(lengths.quantile(0.99)),
        }
    )

    return summary


# ---------------------------------------------------------
# 6. Duplicates
# ---------------------------------------------------------

def report_duplicates(df, summary):

    section("6. DUPLICATE DESCRIPTIONS (LEAKAGE RISK)")

    with_text = df[df["has_description"]].copy()

    normalized = (
        with_text["description"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    duplicate_mask = normalized.duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())

    distinct_texts = int(normalized.nunique())

    print(f"\nRows with a description:     {len(with_text):,}")
    print(f"Distinct description texts:  {distinct_texts:,}")
    print(f"Rows sharing a description:  {duplicate_rows:,}")

    if duplicate_rows:

        print(
            "\n[!] These must be removed before splitting. If the same\n"
            "    text lands in both train and test, Phase 6 measures\n"
            "    memorization rather than generalization."
        )

        # Do duplicated texts ever carry conflicting labels?
        conflicts = (
            with_text[duplicate_mask]
            .assign(_key=normalized[duplicate_mask])
            .groupby("_key")["category"]
            .nunique()
        )

        conflicting = int((conflicts > 1).sum())

        print(
            f"\n    Duplicate texts with conflicting categories: "
            f"{conflicting:,}"
        )

        if conflicting:
            print(
                "    These are genuinely ambiguous - identical text,\n"
                "    different label. They cap the accuracy any model\n"
                "    can reach."
            )

    summary.append(
        {"metric": "duplicate_description_rows", "value": duplicate_rows}
    )

    return summary


# ---------------------------------------------------------
# 7. Resolution availability
# ---------------------------------------------------------

def report_resolutions(df, summary):

    section("7. RESOLUTION AVAILABILITY (RAG CORPUS)")

    retrievable = df[df["retrievable"]]

    print(f"\nTickets usable as grounded RAG context: {len(retrievable):,}")

    if len(retrievable) == 0:
        print("\n[!] No grounded context available.")
        return summary

    print("\nBy source:\n")
    print(retrievable["source"].value_counts().to_string())

    print("\nResolution word count:\n")
    print(
        retrievable["resolution_word_count"]
        .describe()
        .round(1)
        .to_string()
    )

    print("\nCategory spread of the RAG corpus:\n")
    print(
        retrievable["category"]
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\nPriority spread of the RAG corpus:\n")
    print(
        retrievable["priority"]
        .value_counts(dropna=False)
        .to_string()
    )

    return summary


# ---------------------------------------------------------
# 8. Leakage checks
# ---------------------------------------------------------

def report_leakage(df, summary):

    section("8. LEAKAGE CHECKS")

    print(
        "\nA. Does the label appear inside its own description text?\n"
    )

    trainable = df[df["trainable_category"]].copy()

    leak_counts = {}

    for scheme, group in trainable.groupby("category_scheme"):

        label_in_text = [
            str(label).lower() in str(text).lower()
            for label, text in zip(
                group["category"],
                group["description"],
            )
        ]

        leaked = int(np.sum(label_in_text))
        percent = 100 * leaked / len(group)

        leak_counts[scheme] = leaked

        print(
            f"  {scheme:<24} {leaked:>7,} / {len(group):,} "
            f"({percent:5.1f}%)"
        )

        summary.append(
            {"metric": f"{scheme}__label_in_text", "value": leaked}
        )

    print(
        "\n  A high percentage means the classifier can succeed by\n"
        "  string-matching rather than understanding the problem.\n"
        "  Such tokens should be stripped before vectorizing."
    )

    print("\nB. Fields that must NOT become features:\n")

    forbidden = [
        ("resolution", "known only after the ticket is solved"),
        ("status", "encodes the outcome"),
        ("has_usable_resolution", "derived from the resolution"),
        ("resolution_word_count", "derived from the resolution"),
        ("retrievable", "derived from the resolution"),
    ]

    for field, reason in forbidden:
        print(f"  {field:<24} {reason}")

    print(
        "\n  Each of these is populated only after a ticket is closed.\n"
        "  A live ticket arriving at the CLI has none of them, so a\n"
        "  model trained on them would score well offline and fail in\n"
        "  production."
    )

    return summary


# ---------------------------------------------------------
# 9. Recommendation
# ---------------------------------------------------------

def report_recommendation(df):

    section("9. WHAT PHASE 6 SHOULD USE")

    trainable = df[df["trainable_category"]]

    josse = trainable[trainable["category_scheme"] == "jira_project"]

    print(
        f"""
  Category model
    features : description (text only)
    target   : category, filtered to one category_scheme
    rows     : {len(josse):,} for jira_project
    caution  : dominated by one scheme - state this in the README

  Priority model
    features : description (text only)
    target   : priority
    rows     : {int(df['trainable_priority'].sum()):,}
    caution  : too few rows for a strong model - report the weak
               metrics honestly rather than tuning until they look good

  Retrieval corpus
    text     : description
    payload  : resolution, category, priority, source
    rows     : {int(df['retrievable'].sum()):,}

  Excluded from every model
    resolution, status, and all flags derived from them
        """
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print(SEPARATOR)
    print("PHASE 5 - EXPLORATORY ANALYSIS")
    print(SEPARATOR)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run "
            f"src/data/build_unified_dataset.py first."
        )

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    summary = []

    summary = report_overview(df, summary)
    report_missing(df)
    summary = report_categories(df, summary)
    summary = report_priority(df, summary)
    summary = report_description_length(df, summary)
    summary = report_duplicates(df, summary)
    summary = report_resolutions(df, summary)
    summary = report_leakage(df, summary)

    report_recommendation(df)

    pd.DataFrame(summary).to_csv(SUMMARY_PATH, index=False)

    section("OUTPUTS")
    print(f"\n  {SUMMARY_PATH}")
    print(f"  {CATEGORY_PATH}")


if __name__ == "__main__":
    main()
