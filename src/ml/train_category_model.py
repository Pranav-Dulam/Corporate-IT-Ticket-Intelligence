"""
Phase 6 - Category Classification.

Baseline model: TF-IDF + Logistic Regression.

Trains on the jira_project scheme only, applying the five decisions
made in Phase 5:

  1. jira_project scheme only (mendeley_issue_type is 94.9% one class)
  2. classes with at least MIN_CLASS_SIZE examples
  3. descriptions capped at MAX_WORDS
  4. duplicate descriptions removed before splitting
  5. optional removal of project-name tokens, to measure how much
     of the score is string-matching rather than understanding

Run both ways and compare:

    python src/ml/train_category_model.py
    python src/ml/train_category_model.py --strip-project-names
"""

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Anchor all paths to the project root so the script works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data/processed/unified_tickets.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "data/processed"

SCHEME = "jira_project"

MIN_CLASS_SIZE = 10      # Phase 5 decision
MAX_WORDS = 512          # Phase 5 decision
TEST_SIZE = 0.2
RANDOM_SEED = 42         # reproducibility

SEPARATOR = "=" * 70


def section(title):
    print("\n" + SEPARATOR)
    print(title)
    print(SEPARATOR)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def cap_words(text, limit=MAX_WORDS):
    """
    Truncate to the first `limit` words.

    One JOSSE ticket is 57,662 words of pasted stack trace. Without
    a cap it contributes more tokens than a thousand normal tickets
    combined and distorts the whole vocabulary.
    """

    words = str(text).split()

    if len(words) <= limit:
        return str(text)

    return " ".join(words[:limit])


def build_project_name_pattern(labels):
    """
    One regex matching any known project name as a whole word,
    plus Jira-style issue keys such as ZOOKEEPER-3063.

    Longest names first so that, where one name is a prefix of
    another, the longer match wins.
    """

    names = sorted(
        {str(label) for label in labels if pd.notna(label)},
        key=len,
        reverse=True,
    )

    escaped = [re.escape(name) for name in names]

    alternation = "|".join(escaped)

    return re.compile(
        rf"\b(?:{alternation})(?:-\d+)?\b",
        flags=re.IGNORECASE,
    )


def strip_project_names(series, pattern):
    """
    Remove project-name tokens from every description.

    Applied identically to train, test, and any future input, so it
    needs no knowledge of the label. It is preprocessing, not
    cheating: it removes a shortcut uniformly rather than removing
    information from one split only.
    """

    return (
        series
        .astype(str)
        .str.replace(pattern, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ---------------------------------------------------------
# Data preparation
# ---------------------------------------------------------

def load_dataset(strip_names):

    section("1. LOADING DATA")

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    df = df[
        df["trainable_category"]
        & (df["category_scheme"] == SCHEME)
    ].copy()

    print(f"\nRows in scheme '{SCHEME}': {len(df):,}")

    # --- deduplicate BEFORE splitting -------------------
    normalized = (
        df["description"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    before = len(df)
    df = df[~normalized.duplicated(keep="first")].copy()

    print(f"Duplicate descriptions removed: {before - len(df):,}")

    # --- cap length --------------------------------------
    df["text"] = df["description"].apply(cap_words)

    # --- optionally strip project names ------------------
    if strip_names:
        pattern = build_project_name_pattern(df["category"].unique())
        df["text"] = strip_project_names(df["text"], pattern)

        empty = int((df["text"].str.split().str.len() < 3).sum())
        print(f"Rows left with under 3 words after stripping: {empty:,}")

        df = df[df["text"].str.split().str.len() >= 3].copy()

    # --- drop rare classes -------------------------------
    counts = df["category"].value_counts()
    keep = counts[counts >= MIN_CLASS_SIZE].index

    before = len(df)
    df = df[df["category"].isin(keep)].copy()

    print(
        f"Classes with >= {MIN_CLASS_SIZE} examples: {len(keep):,} "
        f"(dropped {before - len(df):,} rows)"
    )

    print(f"\nFinal training set: {len(df):,} rows, "
          f"{df['category'].nunique():,} classes")

    return df


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_pipeline():
    """
    TF-IDF and the classifier live in one Pipeline.

    This is the leakage guard: pipeline.fit() fits the vectorizer on
    the training fold only. Vectorizing the full dataset beforehand
    would let test-set vocabulary and document frequencies influence
    the features, inflating every score.
    """

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate(y_true, y_pred, labels):

    section("4. EVALUATION")

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    macro_precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"""
  accuracy          {accuracy:.4f}
  macro F1          {macro_f1:.4f}     <- the honest number
  weighted F1       {weighted_f1:.4f}
  macro precision   {macro_precision:.4f}
  macro recall      {macro_recall:.4f}
""")

    print(
        "  Accuracy is flattered by the large classes. Macro F1 gives\n"
        "  every class equal weight, so a model that only handles the\n"
        "  top 10 projects cannot hide behind it."
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strip-project-names",
        action="store_true",
        help="Remove project-name tokens before vectorizing.",
    )

    args = parser.parse_args()

    variant = "stripped" if args.strip_project_names else "raw"

    print(SEPARATOR)
    print(f"PHASE 6 - CATEGORY CLASSIFICATION  [{variant}]")
    print(SEPARATOR)

    df = load_dataset(args.strip_project_names)

    # -----------------------------------------------------
    section("2. TRAIN / TEST SPLIT")

    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"\n  train: {len(X_train):,} rows")
    print(f"  test:  {len(X_test):,} rows")
    print(f"  stratified, seed={RANDOM_SEED}")

    # -----------------------------------------------------
    section("3. TRAINING")

    pipeline = build_pipeline()

    print("\nFitting TF-IDF + Logistic Regression...")
    print("(a few minutes with this many classes)")

    pipeline.fit(X_train, y_train)

    vocab_size = len(
        pipeline.named_steps["tfidf"].vocabulary_
    )

    print(f"\n  vocabulary: {vocab_size:,} features")

    # -----------------------------------------------------
    y_pred = pipeline.predict(X_test)

    labels = sorted(y.unique())

    metrics = evaluate(y_test, y_pred, labels)

    # -----------------------------------------------------
    section("5. PER-CLASS REPORT (worst 15 by F1)")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    per_class = (
        pd.DataFrame(report)
        .transpose()
        .drop(
            index=["accuracy", "macro avg", "weighted avg"],
            errors="ignore",
        )
        .sort_values("f1-score")
    )

    print()
    print(per_class.head(15).round(3).to_string())

    # -----------------------------------------------------
    section("6. SAVING")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"category_model_{variant}.joblib"

    joblib.dump(pipeline, model_path)

    # The whole Pipeline is saved, vectorizer included, so inference
    # never needs to rebuild or refit anything.

    metrics_path = REPORT_DIR / f"category_metrics_{variant}.csv"

    pd.DataFrame([{"variant": variant, **metrics}]).to_csv(
        metrics_path, index=False
    )

    per_class_path = REPORT_DIR / f"category_per_class_{variant}.csv"
    per_class.to_csv(per_class_path)

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_path = REPORT_DIR / f"category_confusion_{variant}.csv"

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path)

    print(f"\n  {model_path}")
    print(f"  {metrics_path}")
    print(f"  {per_class_path}")
    print(f"  {cm_path}")


if __name__ == "__main__":
    main()