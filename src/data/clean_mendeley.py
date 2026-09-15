from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# Anchor all paths to the project root so the script works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data/raw/mendeley_helpdesk/issues.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/interim/mendeley_clean.csv"
REPORT_PATH = PROJECT_ROOT / "data/interim/mendeley_cleaning_report.csv"


# ---------------------------------------------------------
# Cleaning configuration
# ---------------------------------------------------------

SEMANTIC_MISSING = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "-",
}

TIMESTAMP_COLUMNS = [
    "started",
    "ended",
    "issue_created",
    "issue_resolution_date",
    "last_change_date",
]

PRIORITY_MAP = {
    "highest": "P1",
    "blocker": "P1",
    "high": "P2",
    "medium": "P3",
    "low": "P4",
    "lowest": "P4",
}


# ---------------------------------------------------------
# String cleaning
# ---------------------------------------------------------

def normalize_object_columns(df):
    """
    Remove leading/trailing whitespace from text columns.
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


# ---------------------------------------------------------
# Semantic missing values
# ---------------------------------------------------------

def replace_semantic_missing(df):
    """
    Convert placeholder strings such as 'unknown' and 'N/A'
    into proper missing values.

    Returns:
        cleaned dataframe
        number of replacements
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

        mask = normalized.isin(SEMANTIC_MISSING)

        replacements += int(mask.sum())

        df.loc[mask, column] = pd.NA

    return df, replacements


# ---------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------

def parse_timestamps(df):
    """
    Parse timestamps that occur both with and without
    fractional seconds.

    Examples:
        2016-04-02 12:20:21+00:00
        2018-08-27 09:53:57.546000+00:00
    """

    invalid_timestamp_count = 0

    for column in TIMESTAMP_COLUMNS:

        if column not in df.columns:
            continue

        raw = df[column]

        # Pass 1: timestamps without fractional seconds
        parsed = pd.to_datetime(
            raw,
            format="%Y-%m-%d %H:%M:%S%z",
            errors="coerce",
            utc=True,
        )

        # Pass 2: retry failures with fractional seconds
        failed_mask = raw.notna() & parsed.isna()

        if failed_mask.any():

            parsed_fractional = pd.to_datetime(
                raw[failed_mask],
                format="%Y-%m-%d %H:%M:%S.%f%z",
                errors="coerce",
                utc=True,
            )

            parsed.loc[failed_mask] = parsed_fractional

        # Count values that still failed both formats
        final_failed = raw.notna() & parsed.isna()

        invalid_timestamp_count += int(
            final_failed.sum()
        )

        df[column] = parsed

    return df, invalid_timestamp_count


# ---------------------------------------------------------
# Priority normalization
# ---------------------------------------------------------

def add_priority_normalization(df):
    """
    Preserve original priority and create standardized
    enterprise priority values P1-P4.
    """

    if "issue_priority" not in df.columns:
        return df

    normalized = (
        df["issue_priority"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["priority_normalized"] = (
        normalized
        .map(PRIORITY_MAP)
        .astype("string")
    )

    return df


# ---------------------------------------------------------
# Resolution-time feature
# ---------------------------------------------------------

def add_resolution_features(df):
    """
    Calculate ticket resolution time.

    Negative resolution times are invalid.
    They are flagged and replaced with NaN.
    """

    required_columns = {
        "issue_created",
        "issue_resolution_date",
    }

    if not required_columns.issubset(df.columns):
        return df

    resolution_delta = (
        df["issue_resolution_date"]
        - df["issue_created"]
    )

    df["resolution_time_hours"] = (
        resolution_delta.dt.total_seconds()
        / 3600
    )

    df["invalid_resolution_chronology"] = (
        df["resolution_time_hours"] < 0
    )

    df.loc[
        df["invalid_resolution_chronology"],
        "resolution_time_hours",
    ] = np.nan

    return df


# ---------------------------------------------------------
# Workflow anomaly cleaning
# ---------------------------------------------------------

def clean_negative_workflow_values(df):
    """
    Detect invalid negative workflow durations.

    The affected ticket is preserved and flagged.
    Only the invalid negative duration is replaced with NaN.

    Returns:
        cleaned dataframe
        number of affected rows
        number of negative values replaced
    """

    workflow_columns = [
        column
        for column in df.columns
        if (
            column.startswith("wf_")
            or column.startswith("wfe_")
        )
    ]

    affected_rows = pd.Series(
        False,
        index=df.index,
    )

    negative_values_replaced = 0

    for column in workflow_columns:

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        negative_mask = df[column] < 0

        negative_values_replaced += int(
            negative_mask.sum()
        )

        affected_rows |= negative_mask

        df.loc[
            negative_mask,
            column,
        ] = np.nan

    df["has_negative_workflow_value"] = affected_rows

    return (
        df,
        int(affected_rows.sum()),
        negative_values_replaced,
    )


# ---------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("MENDELEY HELPDESK DATA CLEANING")
    print("=" * 60)

    print("\nLoading raw dataset...")

    df = pd.read_csv(INPUT_PATH)

    rows_before = len(df)
    columns_before = len(df.columns)

    print(f"Raw rows:    {rows_before:,}")
    print(f"Raw columns: {columns_before}")

    # -----------------------------------------------------
    # Normalize column names
    # -----------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # -----------------------------------------------------
    # Clean text
    # -----------------------------------------------------

    df = normalize_object_columns(df)

    # -----------------------------------------------------
    # Remove exact duplicates
    # -----------------------------------------------------

    duplicates_removed = int(
        df.duplicated().sum()
    )

    if duplicates_removed > 0:
        df = (
            df
            .drop_duplicates()
            .copy()
        )

    # -----------------------------------------------------
    # Semantic missing values
    # -----------------------------------------------------

    df, semantic_missing_replaced = (
        replace_semantic_missing(df)
    )

    # -----------------------------------------------------
    # Parse timestamps
    # -----------------------------------------------------

    df, invalid_timestamps = (
        parse_timestamps(df)
    )

    # -----------------------------------------------------
    # Standardize priority
    # -----------------------------------------------------

    df = add_priority_normalization(df)

    # -----------------------------------------------------
    # Resolution metrics
    # -----------------------------------------------------

    df = add_resolution_features(df)

    # -----------------------------------------------------
    # Workflow anomaly cleaning
    # -----------------------------------------------------

    (
        df,
        negative_workflow_rows,
        negative_workflow_values_replaced,
    ) = clean_negative_workflow_values(df)

    # -----------------------------------------------------
    # Count invalid chronology
    # -----------------------------------------------------

    if "invalid_resolution_chronology" in df.columns:

        invalid_resolution_rows = int(
            df[
                "invalid_resolution_chronology"
            ].sum()
        )

    else:

        invalid_resolution_rows = 0

    # -----------------------------------------------------
    # Save cleaned dataset
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Cleaning report
    # -----------------------------------------------------

    report = pd.DataFrame(
        [
            {
                "metric": "rows_before",
                "value": rows_before,
            },
            {
                "metric": "rows_after",
                "value": len(df),
            },
            {
                "metric": "columns_before",
                "value": columns_before,
            },
            {
                "metric": "columns_after",
                "value": len(df.columns),
            },
            {
                "metric": "duplicates_removed",
                "value": duplicates_removed,
            },
            {
                "metric": "semantic_missing_replaced",
                "value": semantic_missing_replaced,
            },
            {
                "metric": "invalid_timestamps",
                "value": invalid_timestamps,
            },
            {
                "metric": "invalid_resolution_chronology_rows",
                "value": invalid_resolution_rows,
            },
            {
                "metric": "negative_workflow_rows",
                "value": negative_workflow_rows,
            },
            {
                "metric": "negative_workflow_values_replaced",
                "value": negative_workflow_values_replaced,
            },
        ]
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(f"\nClean shape: {df.shape}")

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

    print("\nPriority distribution:\n")

    if "priority_normalized" in df.columns:

        print(
            df["priority_normalized"]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


if __name__ == "__main__":
    main()