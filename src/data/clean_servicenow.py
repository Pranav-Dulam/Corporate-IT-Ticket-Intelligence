from pathlib import Path

import numpy as np
import pandas as pd


# Anchor all paths to the project root so the script works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data/raw/servicenow_uci/incident_event_log.csv"
EVENT_OUTPUT_PATH = PROJECT_ROOT / "data/interim/servicenow_events_clean.csv"
INCIDENT_OUTPUT_PATH = PROJECT_ROOT / "data/interim/servicenow_incidents_clean.csv"
REPORT_PATH = PROJECT_ROOT / "data/interim/servicenow_cleaning_report.csv"


SEMANTIC_MISSING = {"?", "", "unknown", "none", "null", "n/a", "na", "-"}

TIMESTAMP_COLUMNS = [
    "opened_at",
    "sys_created_at",
    "sys_updated_at",
    "resolved_at",
    "closed_at",
]

PRIORITY_MAP = {
    "1 - critical": "P1",
    "2 - high": "P2",
    "3 - moderate": "P3",
    "4 - low": "P4",
}

VALID_INCIDENT_STATES = {
    "New",
    "Active",
    "Awaiting User Info",
    "Awaiting Problem",
    "Awaiting Vendor",
    "Awaiting Evidence",
    "Resolved",
    "Closed",
}


def normalize_text_columns(df):
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


def parse_timestamps(df):
    invalid_timestamp_count = 0

    for column in TIMESTAMP_COLUMNS:

        if column not in df.columns:
            continue

        raw = df[column]

        parsed = pd.to_datetime(
            raw,
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )

        invalid_mask = raw.notna() & parsed.isna()

        invalid_timestamp_count += int(
            invalid_mask.sum()
        )

        df[column] = parsed

    return df, invalid_timestamp_count


def normalize_priority(df):
    if "priority" not in df.columns:
        return df

    normalized = (
        df["priority"]
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


def clean_incident_state(df):
    """
    Preserve valid ServiceNow states.
    Invalid values such as -100 are flagged and replaced
    with missing values.
    """

    invalid_mask = (
        df["incident_state"].notna()
        & ~df["incident_state"].isin(VALID_INCIDENT_STATES)
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    df["invalid_incident_state"] = invalid_mask

    df.loc[
        invalid_mask,
        "incident_state",
    ] = pd.NA

    return df, invalid_count


def add_event_features(df):
    """
    Add event-level ordering and lifecycle features.
    """

    df = df.sort_values(
        ["number", "sys_updated_at"],
        kind="stable",
    ).copy()

    df["event_sequence"] = (
        df.groupby("number")
        .cumcount()
        + 1
    )

    df["incident_event_count"] = (
        df.groupby("number")["number"]
        .transform("size")
    )

    if {
        "opened_at",
        "sys_updated_at",
    }.issubset(df.columns):

        delta = (
            df["sys_updated_at"]
            - df["opened_at"]
        )

        df["event_age_hours"] = (
            delta.dt.total_seconds()
            / 3600
        )

        df["invalid_event_chronology"] = (
            df["event_age_hours"] < 0
        )

        df.loc[
            df["invalid_event_chronology"],
            "event_age_hours",
        ] = np.nan

    return df


def add_resolution_features(df):
    """
    Calculate overall incident resolution and closure time.
    """

    if {
        "opened_at",
        "resolved_at",
    }.issubset(df.columns):

        resolution_delta = (
            df["resolved_at"]
            - df["opened_at"]
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

    if {
        "opened_at",
        "closed_at",
    }.issubset(df.columns):

        closure_delta = (
            df["closed_at"]
            - df["opened_at"]
        )

        df["closure_time_hours"] = (
            closure_delta.dt.total_seconds()
            / 3600
        )

        df["invalid_closure_chronology"] = (
            df["closure_time_hours"] < 0
        )

        df.loc[
            df["invalid_closure_chronology"],
            "closure_time_hours",
        ] = np.nan

    return df


def build_incident_snapshot(df):
    """
    Create one analytical row per incident.

    The last known event is used as the final incident state.
    """

    snapshot = (
        df.sort_values(
            ["number", "sys_updated_at"],
            kind="stable",
        )
        .groupby(
            "number",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    snapshot = snapshot.reset_index(
        drop=True
    )

    return snapshot


def main():

    print("=" * 65)
    print("SERVICENOW UCI DATA CLEANING")
    print("=" * 65)

    print("\nLoading raw event log...")

    df = pd.read_csv(INPUT_PATH)

    rows_before = len(df)
    columns_before = len(df.columns)
    incidents_before = df["number"].nunique()

    print(f"Raw event rows:       {rows_before:,}")
    print(f"Raw columns:          {columns_before}")
    print(f"Unique incidents:     {incidents_before:,}")

    # Normalize names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Normalize text
    df = normalize_text_columns(df)

    # Exact duplicate detection
    duplicates_removed = int(
        df.duplicated().sum()
    )

    if duplicates_removed > 0:
        df = (
            df
            .drop_duplicates()
            .copy()
        )

    # Replace placeholders
    (
        df,
        semantic_missing_replaced,
    ) = replace_semantic_missing(df)

    # Timestamps
    (
        df,
        invalid_timestamps,
    ) = parse_timestamps(df)

    # Priority
    df = normalize_priority(df)

    # Incident state anomalies
    (
        df,
        invalid_incident_states,
    ) = clean_incident_state(df)

    # Event features
    df = add_event_features(df)

    # Resolution features
    df = add_resolution_features(df)

    invalid_event_chronology = int(
        df.get(
            "invalid_event_chronology",
            pd.Series(False, index=df.index),
        ).sum()
    )

    invalid_resolution_chronology = int(
        df.get(
            "invalid_resolution_chronology",
            pd.Series(False, index=df.index),
        ).sum()
    )

    invalid_closure_chronology = int(
        df.get(
            "invalid_closure_chronology",
            pd.Series(False, index=df.index),
        ).sum()
    )

    # Create incident-level table
    incident_df = build_incident_snapshot(
        df
    )

    # Output
    EVENT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        EVENT_OUTPUT_PATH,
        index=False,
    )

    incident_df.to_csv(
        INCIDENT_OUTPUT_PATH,
        index=False,
    )

    report = pd.DataFrame(
        [
            {
                "metric": "event_rows_before",
                "value": rows_before,
            },
            {
                "metric": "event_rows_after",
                "value": len(df),
            },
            {
                "metric": "columns_before",
                "value": columns_before,
            },
            {
                "metric": "event_columns_after",
                "value": len(df.columns),
            },
            {
                "metric": "unique_incidents_before",
                "value": incidents_before,
            },
            {
                "metric": "incident_snapshot_rows",
                "value": len(incident_df),
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
                "metric": "invalid_incident_states",
                "value": invalid_incident_states,
            },
            {
                "metric": "invalid_event_chronology_rows",
                "value": invalid_event_chronology,
            },
            {
                "metric": "invalid_resolution_chronology_rows",
                "value": invalid_resolution_chronology,
            },
            {
                "metric": "invalid_closure_chronology_rows",
                "value": invalid_closure_chronology,
            },
        ]
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("\n" + "=" * 65)
    print("CLEANING COMPLETE")
    print("=" * 65)

    print(
        f"\nClean event log: {df.shape}"
    )

    print(
        f"Incident snapshot: {incident_df.shape}"
    )

    print(
        f"\nEvent output:\n{EVENT_OUTPUT_PATH}"
    )

    print(
        f"\nIncident output:\n{INCIDENT_OUTPUT_PATH}"
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

    print("\nPriority distribution — incident level:\n")

    print(
        incident_df[
            "priority_normalized"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print("\nFinal incident states:\n")

    print(
        incident_df[
            "incident_state"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


if __name__ == "__main__":
    main()