"""
Phase 7 - Priority Prediction.

Predicts ticket urgency from description text.

The constraint that shapes everything here is size: only 353 rows in
unified_tickets.csv have both a description and a priority label, all
of them Mendeley tickets enriched from sample_utterances.csv.

At that size a single train/test split is not a measurement - 71 test
rows move several points on the seed alone. So evaluation uses:

  * RepeatedStratifiedKFold (5 folds x 5 repeats) -> mean and std
  * a DummyClassifier baseline, so "better than guessing" is proven
    rather than assumed
  * a permutation test, which re-scores with shuffled labels to give
    a p-value for the result being real

Phase 5 found the four-class target unusable - P4 has 8 examples, so a
stratified test fold receives one or two. The default mode collapses
the labels to High (P1+P2, 193 rows) and Low (P3+P4, 160 rows), which
is nearly balanced. Four-class is still available for comparison, and
reporting how badly it does is part of the finding.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data/processed/unified_tickets.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "data/processed"

MAX_WORDS = 512          # Phase 5 decision, kept consistent
N_SPLITS = 5
N_REPEATS = 5
N_PERMUTATIONS = 100
RANDOM_SEED = 42

BINARY_MAP = {
    "P1": "High",
    "P2": "High",
    "P3": "Low",
    "P4": "Low",
}

SEPARATOR = "=" * 70


def section(title):
    print("\n" + SEPARATOR)
    print(title)
    print(SEPARATOR)


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

def cap_words(text, limit=MAX_WORDS):
    words = str(text).split()
    return str(text) if len(words) <= limit else " ".join(words[:limit])


def load_dataset(mode):

    section("1. LOADING DATA")

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    df = df[df["trainable_priority"]].copy()

    print(f"\nRows with description and priority: {len(df):,}")
    print("\nSources:\n")
    print(df["source"].value_counts().to_string())

    # Deduplicate before anything else, same as Phase 6.
    normalized = (
        df["description"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    before = len(df)
    df = df[~normalized.duplicated(keep="first")].copy()
    print(f"\nDuplicate descriptions removed: {before - len(df):,}")

    df["text"] = df["description"].apply(cap_words)

    if mode == "binary":
        df["target"] = df["priority"].map(BINARY_MAP)
    else:
        df["target"] = df["priority"]

    df = df[df["target"].notna()].copy()

    print(f"\nFinal dataset: {len(df):,} rows\n")

    counts = df["target"].value_counts()

    for label, count in counts.items():
        print(f"  {label:<6} {count:>4}  ({100 * count / len(df):5.1f}%)")

    smallest = counts.min()
    print(f"\n  smallest class: {smallest} examples")

    if smallest < N_SPLITS:
        print(
            f"\n[!] Smallest class has fewer examples than the "
            f"{N_SPLITS} CV folds.\n"
            f"    Some folds will contain none of it. Treat every "
            f"per-class\n    number for that label as unreliable."
        )

    return df, int(smallest)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_pipeline(model_name, smallest_class):
    """
    TF-IDF and classifier in one Pipeline, so the vectorizer is only
    ever fitted on the training fold.

    Settings differ from Phase 6 because the dataset is 65x smaller:
    min_df=1 (dropping terms seen once would discard most of the
    vocabulary) and no max_features cap (there is nothing to cap).

    LinearSVC is wrapped for probabilities, but calibration needs
    enough examples per class in each inner fold. With a tiny
    minority class that is not available, so the code falls back to
    logistic regression rather than producing a silently broken
    calibrator.
    """

    if model_name == "linearsvc" and smallest_class >= 15:
        classifier = CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=5000,
            ),
            cv=3,
            method="sigmoid",
        )
    else:
        if model_name == "linearsvc":
            print(
                f"\n[!] Smallest class ({smallest_class}) is too small "
                f"for reliable\n    SVC calibration - using logistic "
                f"regression instead."
            )

        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            ("clf", classifier),
        ]
    )


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def cross_validate_model(pipeline, X, y, label):

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )

    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

    print(f"\n  {label}  ({N_SPLITS} folds x {N_REPEATS} repeats)\n")

    summary = {}

    for metric in scoring:
        scores = results[f"test_{metric}"]
        mean, std = scores.mean(), scores.std()

        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std

        print(f"    {metric:<18} {mean:.4f}  +/- {std:.4f}")

    return summary


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["binary", "four-class"],
        default="binary",
        help="Target granularity (default: binary High/Low).",
    )

    parser.add_argument(
        "--model",
        choices=["linearsvc", "logreg"],
        default="linearsvc",
    )

    args = parser.parse_args()

    tag = f"{args.model}_{args.mode}"

    print(SEPARATOR)
    print(f"PHASE 7 - PRIORITY PREDICTION  [{tag}]")
    print(SEPARATOR)

    df, smallest = load_dataset(args.mode)

    X = df["text"]
    y = df["target"]

    # -----------------------------------------------------
    section("2. BASELINE")

    print(
        "\nA model on 353 rows must be shown to beat guessing.\n"
        "The dummy classifier always predicts the majority class."
    )

    dummy = DummyClassifier(
        strategy="most_frequent",
        random_state=RANDOM_SEED,
    )

    baseline = cross_validate_model(dummy, X, y, "DummyClassifier")

    # -----------------------------------------------------
    section("3. MODEL")

    pipeline = build_pipeline(args.model, smallest)

    metrics = cross_validate_model(pipeline, X, y, tag)

    gain = metrics["f1_macro_mean"] - baseline["f1_macro_mean"]

    print(f"\n  macro F1 over baseline: {gain:+.4f}")

    if gain < 2 * metrics["f1_macro_std"]:
        print(
            "\n[!] The gain is smaller than two standard deviations of\n"
            "    the model's own fold-to-fold variation. That is not a\n"
            "    convincing improvement over guessing."
        )

    # -----------------------------------------------------
    section("4. PERMUTATION TEST")

    print(
        f"\nRefitting {N_PERMUTATIONS} times with the labels shuffled.\n"
        "If the real score sits inside the shuffled distribution, the\n"
        "model learned nothing and the result is chance."
    )

    score, permutation_scores, pvalue = permutation_test_score(
        pipeline,
        X,
        y,
        scoring="f1_macro",
        cv=StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        ),
        n_permutations=N_PERMUTATIONS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print(f"""
    real macro F1       {score:.4f}
    shuffled mean       {permutation_scores.mean():.4f}
    shuffled max        {permutation_scores.max():.4f}
    p-value             {pvalue:.4f}
""")

    if pvalue > 0.05:
        print(
            "[!] p > 0.05 - this result is not distinguishable from\n"
            "    chance. Report it as a negative finding."
        )
    else:
        print("    p <= 0.05 - the model is learning real signal.")

    # -----------------------------------------------------
    section("5. PER-CLASS DETAIL")

    y_pred = cross_val_predict(
        pipeline,
        X,
        y,
        cv=StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        ),
    )

    print()
    print(classification_report(y, y_pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)

    print("Confusion matrix (rows = true, columns = predicted):\n")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    # -----------------------------------------------------
    section("6. SAVING")

    # Fitted on all rows. With 353 examples, holding some back
    # permanently costs more than the independent estimate is worth -
    # and the cross-validated numbers above are that estimate.
    pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"priority_model_{tag}.joblib"
    joblib.dump(pipeline, model_path)

    metrics_path = REPORT_DIR / f"priority_metrics_{tag}.csv"

    pd.DataFrame(
        [
            {
                "variant": tag,
                "rows": len(df),
                "classes": y.nunique(),
                **metrics,
                "baseline_f1_macro": baseline["f1_macro_mean"],
                "permutation_pvalue": pvalue,
            }
        ]
    ).to_csv(metrics_path, index=False)

    cm_path = REPORT_DIR / f"priority_confusion_{tag}.csv"
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path)

    print(f"\n  {model_path}")
    print(f"  {metrics_path}")
    print(f"  {cm_path}")


if __name__ == "__main__":
    main()