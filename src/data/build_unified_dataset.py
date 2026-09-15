"""
Phase 4 - Dataset Integration.

Merge the three cleaned datasets into one standardized table.

Design note:
Each dataset gets its own adapter function that maps its columns
onto the unified schema. Adapters are independent, so adding a
fourth dataset later means writing one more function, not
rewriting this script.

Unified schema:
    ticket_id       stable, source-prefixed identifier
    description     free-text problem statement (may be missing)
    category        raw label, NOT harmonized across sources
    category_scheme which vocabulary `category` came from
    priority        P1-P4 where the source supports it
    resolution      free-text fix (may be missing)
    status          lifecycle state where available
    source          which dataset the row came from
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# Anchor all paths to the project root so the script works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MENDELEY_PATH = PROJECT_ROOT / "data/interim/mendeley_clean.csv"
SERVICENOW_PATH = PROJECT_ROOT / "data/interim/servicenow_incidents_clean.csv"
JOSSE_PATH = PROJECT_ROOT / "data/interim/josse_clean.csv"

UTTERANCES_PATH = PROJECT_ROOT / "data/raw/mendeley_helpdesk/sample_utterances.csv"

OUTPUT_PATH = PROJECT_ROOT / "data/processed/unified_tickets.csv"
REPORT_PATH = PROJECT_ROOT / "data/processed/unified_tickets_report.csv"


UNIFIED_COLUMNS = [
    "ticket_id",
    "description",
    "category",
    "category_scheme",
    "priority",
    "resolution",
    "status",
    "source",
]

# Below this, text is too short to classify or retrieve on.
MIN_DESCRIPTION_WORDS = 3
MIN_RESOLUTION_WORDS = 5


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def collapse_whitespace(series):
    """
    Flatten newlines and repeated spaces into single spaces.
    """

    return (
        series
        .fillna("")
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )


def word_count(series):
    return (
        series
        .fillna("")
        .astype("string")
        .str.split()
        .str.len()
        .astype("int64")
    )


def empty_frame(index):
    """
    A column of missing values, for fields a source cannot supply.
    """

    return pd.Series(pd.NA, index=index, dtype="string")


# ---------------------------------------------------------
# Adapter: Mendeley
# ---------------------------------------------------------

def adapt_mendeley():
    """
    Mendeley is a process-mining dataset: rich workflow timings,
    no ticket text. Descriptions are filled in afterwards for the
    360 issues covered by sample_utterances.csv.
    """

    df = pd.read_csv(MENDELEY_PATH, low_memory=False)

    out = pd.DataFrame(index=df.index)

    out["ticket_id"] = (
        "MND-"
        + df["id"].astype("Int64").astype("string")
    )

    out["description"] = empty_frame(df.index)

    out["category"] = df["issue_type"].astype("string")
    out["category_scheme"] = "mendeley_issue_type"

    out["priority"] = df["priority_normalized"].astype("string")

    out["resolution"] = empty_frame(df.index)

    out["status"] = df["issue_status"].astype("string")
    out["source"] = "mendeley"

    # Kept for the enrichment join below, dropped afterwards.
    out["_join_id"] = df["id"].astype("Int64")

    return out


def enrich_mendeley_with_utterances(mendeley):
    """
    sample_utterances.csv holds the conversation for 360 issues.

    Reporter turns describe the problem.
    Assignee turns describe what was actually done about it.

    This is the only place in the three datasets where a
    description and a real resolution sit side by side.
    """

    if not UTTERANCES_PATH.exists():
        print(f"[!] {UTTERANCES_PATH} not found - skipping enrichment.")
        return mendeley, 0

    utterances = pd.read_csv(UTTERANCES_PATH, low_memory=False)

    utterances = utterances.sort_values(
        ["issueid", "comment_seq", "utr_seq"],
        kind="stable",
    )

    utterances["issueid"] = utterances["issueid"].astype("Int64")

    def join_role(role):
        subset = utterances[utterances["author_role"] == role]

        joined = (
            subset
            .groupby("issueid")["actionbody"]
            .apply(lambda s: " ".join(s.dropna().astype(str)))
        )

        return collapse_whitespace(joined)

    descriptions = join_role("reporter")
    resolutions = join_role("assignee")

    mapped_description = mendeley["_join_id"].map(descriptions)
    mapped_resolution = mendeley["_join_id"].map(resolutions)

    mendeley["description"] = mendeley["description"].fillna(
        mapped_description
    )

    mendeley["resolution"] = mendeley["resolution"].fillna(
        mapped_resolution
    )

    enriched = int(mapped_description.notna().sum())

    return mendeley, enriched


# ---------------------------------------------------------
# Adapter: ServiceNow
# ---------------------------------------------------------

def adapt_servicenow():
    """
    ServiceNow is fully anonymized: category is the literal string
    "Category 42", closure reason is "code 6". Useful as structured
    labels, unusable as text. No description exists to supply.
    """

    df = pd.read_csv(SERVICENOW_PATH, low_memory=False)

    out = pd.DataFrame(index=df.index)

    out["ticket_id"] = "SNW-" + df["number"].astype("string")

    out["description"] = empty_frame(df.index)

    out["category"] = df["category"].astype("string")
    out["category_scheme"] = "servicenow_anonymized"

    out["priority"] = df["priority_normalized"].astype("string")

    # closed_code is a disposition code, not a procedure,
    # so it is deliberately NOT mapped onto `resolution`.
    out["resolution"] = empty_frame(df.index)

    out["status"] = df["incident_state"].astype("string")
    out["source"] = "servicenow"

    return out


# ---------------------------------------------------------
# Adapter: JOSSE
# ---------------------------------------------------------

def adapt_josse():
    """
    JOSSE supplies the bulk of the text - Apache Jira issues.
    No priority labels and no resolution text exist in the source.
    """

    df = pd.read_csv(JOSSE_PATH, low_memory=False)

    out = pd.DataFrame(index=df.index)

    out["ticket_id"] = "JOS-" + df["id"].astype("string")

    out["description"] = collapse_whitespace(df["corpus_clean"])

    out["category"] = df["project_key"].astype("string")
    out["category_scheme"] = "jira_project"

    out["priority"] = empty_frame(df.index)
    out["resolution"] = empty_frame(df.index)
    out["status"] = empty_frame(df.index)

    out["source"] = "josse"

    return out


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def add_usability_flags(df):
    """
    Mark which rows can actually feed which pipeline stage.

    Phase 8 requires that tickets without a usable resolution never
    enter the retrieval context. These flags are how that rule is
    enforced downstream instead of being re-derived each time.
    """

    df["description_word_count"] = word_count(df["description"])
    df["resolution_word_count"] = word_count(df["resolution"])

    df["has_description"] = (
        df["description"].notna()
        & (df["description_word_count"] >= MIN_DESCRIPTION_WORDS)
    )

    df["has_usable_resolution"] = (
        df["resolution"].notna()
        & (df["resolution_word_count"] >= MIN_RESOLUTION_WORDS)
    )

    df["trainable_category"] = (
        df["has_description"] & df["category"].notna()
    )

    df["trainable_priority"] = (
        df["has_description"] & df["priority"].notna()
    )

    df["retrievable"] = (
        df["has_description"] & df["has_usable_resolution"]
    )

    return df


def check_unique_ids(df):
    duplicates = int(df["ticket_id"].duplicated().sum())

    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate ticket_id values - "
            f"adapters are colliding."
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("PHASE 4 - DATASET INTEGRATION")
    print("=" * 65)

    print("\nLoading Mendeley...")
    mendeley = adapt_mendeley()

    print("Enriching Mendeley with helpdesk utterances...")
    mendeley, enriched_count = enrich_mendeley_with_utterances(mendeley)
    print(f"  enriched {enriched_count} tickets with real text")

    mendeley = mendeley.drop(columns=["_join_id"])

    print("Loading ServiceNow...")
    servicenow = adapt_servicenow()

    print("Loading JOSSE...")
    josse = adapt_josse()

    unified = pd.concat(
        [
            mendeley[UNIFIED_COLUMNS],
            servicenow[UNIFIED_COLUMNS],
            josse[UNIFIED_COLUMNS],
        ],
        ignore_index=True,
    )

    check_unique_ids(unified)

    unified = add_usability_flags(unified)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUTPUT_PATH, index=False)

    # -----------------------------------------------------
    # Validation report
    # -----------------------------------------------------

    report = pd.DataFrame(
        [
            {"metric": "total_rows", "value": len(unified)},
            {
                "metric": "rows_mendeley",
                "value": int((unified["source"] == "mendeley").sum()),
            },
            {
                "metric": "rows_servicenow",
                "value": int((unified["source"] == "servicenow").sum()),
            },
            {
                "metric": "rows_josse",
                "value": int((unified["source"] == "josse").sum()),
            },
            {
                "metric": "with_description",
                "value": int(unified["has_description"].sum()),
            },
            {
                "metric": "with_usable_resolution",
                "value": int(unified["has_usable_resolution"].sum()),
            },
            {
                "metric": "trainable_category",
                "value": int(unified["trainable_category"].sum()),
            },
            {
                "metric": "trainable_priority",
                "value": int(unified["trainable_priority"].sum()),
            },
            {
                "metric": "retrievable_for_rag",
                "value": int(unified["retrievable"].sum()),
            },
        ]
    )

    report.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 65)
    print("INTEGRATION COMPLETE")
    print("=" * 65)

    print(f"\nUnified shape: {unified.shape}")
    print(f"\nOutput:\n{OUTPUT_PATH}")
    print(f"\nReport:\n{REPORT_PATH}")

    print("\nValidation summary:\n")
    print(report.to_string(index=False))

    print("\nRows by source:\n")
    print(unified["source"].value_counts().to_string())

    print("\nCategory schemes present:\n")
    print(unified["category_scheme"].value_counts().to_string())

    print("\nPriority distribution:\n")
    print(
        unified["priority"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDescription word count (rows that have one):\n")
    print(
        unified.loc[
            unified["has_description"],
            "description_word_count",
        ]
        .describe()
        .round(1)
        .to_string()
    )


if __name__ == "__main__":
    main()
