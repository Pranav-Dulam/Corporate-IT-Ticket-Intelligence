from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/raw/jira_josse/JOSSE_18092020.sqlite3"
)

OUTPUT_PATH = Path(
    "data/interim/josse_clean.csv"
)

REPORT_PATH = Path(
    "data/interim/josse_cleaning_report.csv"
)


SEMANTIC_MISSING = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "-",
}


def load_data():
    """
    Load the JOSSE Jira dataset from SQLite.
    """

    conn = sqlite3.connect(INPUT_PATH)

    try:
        df = pd.read_sql_query(
            'SELECT * FROM "case";',
            conn
        )
    finally:
        conn.close()

    return df


def normalize_text_columns(df):
    """
    Trim whitespace from text fields.
    """

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    return df


def replace_semantic_missing(df):
    """
    Convert placeholder strings to real missing values.
    """

    replacements = 0

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:

        normalized = (
            df[column]
            .str.strip()
            .str.lower()
        )

        mask = normalized.isin(
            SEMANTIC_MISSING
        )

        replacements += int(
            mask.sum()
        )

        df.loc[
            mask,
            column
        ] = pd.NA

    return df, replacements


def clean_effort_columns(df):
    """
    Negative effort values such as -1 represent
    unavailable/invalid effort measurements.

    Preserve an audit flag and replace the value with NaN.
    """

    effort_columns = [
        "expert_estimated_effort",
        "actual_effort",
    ]

    negative_counts = {}

    for column in effort_columns:

        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        negative_mask = (
            df[column] < 0
        )

        negative_counts[column] = int(
            negative_mask.sum()
        )

        df[
            f"{column}_was_negative"
        ] = negative_mask

        df.loc[
            negative_mask,
            column
        ] = np.nan

    return df, negative_counts


def clean_count_columns(df):
    """
    Validate comment/activity counts.
    """

    count_columns = [
        "num_comment",
        "num_activities",
    ]

    negative_counts = {}

    for column in count_columns:

        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        negative_mask = (
            df[column] < 0
        )

        negative_counts[column] = int(
            negative_mask.sum()
        )

        df[
            f"{column}_was_negative"
        ] = negative_mask

        df.loc[
            negative_mask,
            column
        ] = np.nan

    return df, negative_counts


def clean_corpus(text):
    """
    Clean Jira markup while preserving technical content.
    """

    if pd.isna(text):
        return pd.NA

    text = str(text)

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    # Remove common Jira formatting markers
    text = re.sub(
        r"\{code(?::[^}]*)?\}",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\{noformat\}",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove Jira user mention syntax
    text = re.sub(
        r"\[~[^\]]+\]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return pd.NA

    return text


def add_text_features(df):
    """
    Create cleaned NLP text and basic text statistics.
    """

    if "corpus" not in df.columns:
        return df

    df["corpus_clean"] = (
        df["corpus"]
        .apply(clean_corpus)
        .astype("string")
    )

    df["corpus_char_count"] = (
        df["corpus_clean"]
        .fillna("")
        .str.len()
        .astype("int64")
    )

    df["corpus_word_count"] = (
        df["corpus_clean"]
        .fillna("")
        .str.split()
        .str.len()
        .astype("int64")
    )

    df["empty_corpus"] = (
        df["corpus_clean"].isna()
        |
        (df["corpus_word_count"] == 0)
    )

    return df


def add_project_key(df):
    """
    Extract Jira project key.

    Example:
    ZOOKEEPER-3063 -> ZOOKEEPER
    """

    if "id" not in df.columns:
        return df

    df["project_key"] = (
        df["id"]
        .astype("string")
        .str.extract(
            r"^([A-Za-z0-9_]+)-",
            expand=False
        )
        .str.upper()
    )

    return df


def main():

    print("=" * 65)
    print("JOSSE JIRA DATA CLEANING")
    print("=" * 65)

    print("\nLoading SQLite dataset...")

    df = load_data()

    rows_before = len(df)
    columns_before = len(df.columns)

    print(
        f"Raw rows:    {rows_before:,}"
    )

    print(
        f"Raw columns: {columns_before}"
    )

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Normalize text
    df = normalize_text_columns(df)

    # Exact duplicates
    duplicates_removed = int(
        df.duplicated().sum()
    )

    if duplicates_removed > 0:

        df = (
            df
            .drop_duplicates()
            .copy()
        )

    # Semantic missing
    (
        df,
        semantic_missing_replaced
    ) = replace_semantic_missing(df)

    # Effort validation
    (
        df,
        effort_negative_counts
    ) = clean_effort_columns(df)

    # Comment/activity validation
    (
        df,
        count_negative_counts
    ) = clean_count_columns(df)

    # Clean Jira issue text
    df = add_text_features(df)

    # Extract project key
    df = add_project_key(df)

    # Dataset statistics
    empty_corpus_rows = int(
        df["empty_corpus"].sum()
    )

    missing_reference_rows = int(
        df["reference"].isna().sum()
    )

    unique_projects = int(
        df["project_key"].nunique(
            dropna=True
        )
    )

    # Save clean data
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Cleaning report
    report_rows = [
        {
            "metric": "rows_before",
            "value": rows_before
        },
        {
            "metric": "rows_after",
            "value": len(df)
        },
        {
            "metric": "columns_before",
            "value": columns_before
        },
        {
            "metric": "columns_after",
            "value": len(df.columns)
        },
        {
            "metric": "duplicates_removed",
            "value": duplicates_removed
        },
        {
            "metric": "semantic_missing_replaced",
            "value": semantic_missing_replaced
        },
        {
            "metric": "empty_corpus_rows",
            "value": empty_corpus_rows
        },
        {
            "metric": "missing_reference_rows",
            "value": missing_reference_rows
        },
        {
            "metric": "unique_projects",
            "value": unique_projects
        },
    ]

    for column, count in (
        effort_negative_counts.items()
    ):

        report_rows.append(
            {
                "metric":
                    f"negative_{column}_replaced",
                "value": count
            }
        )

    for column, count in (
        count_negative_counts.items()
    ):

        report_rows.append(
            {
                "metric":
                    f"negative_{column}_replaced",
                "value": count
            }
        )

    report = pd.DataFrame(
        report_rows
    )

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    print("\n" + "=" * 65)
    print("CLEANING COMPLETE")
    print("=" * 65)

    print(
        f"\nClean shape: {df.shape}"
    )

    print(
        f"\nClean dataset:\n{OUTPUT_PATH}"
    )

    print(
        f"\nCleaning report:\n{REPORT_PATH}"
    )

    print("\nCleaning summary:\n")

    print(
        report.to_string(
            index=False
        )
    )

    print("\nTop Jira projects:\n")

    print(
        df["project_key"]
        .value_counts(
            dropna=False
        )
        .head(15)
        .to_string()
    )

    print("\nCorpus word-count summary:\n")

    print(
        df["corpus_word_count"]
        .describe()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()