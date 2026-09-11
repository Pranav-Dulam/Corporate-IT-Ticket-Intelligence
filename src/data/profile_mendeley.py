import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/raw/mendeley_helpdesk/issues.csv")
OUTPUT_PATH = Path("data/processed/mendeley_data_quality.csv")

SEMANTIC_MISSING = {
    "unknown",
    "none",
    "n/a",
    "-",
    "",
}

def count_semantic_missing(series: pd.Series) -> int:
    if series.dtype != "object":
        return 0

    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin(SEMANTIC_MISSING).sum()

def sample_values(series: pd.Series, limit: int= 5) -> str:
    values = (
        series.dropna()
        .astype(str)
        .drop_duplicates()
        .head(limit)
        .to_list()
    )
    return " | ".join(values)

def main():

    print("Loading Mendeley help-desk dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows: {len(df):,}")

    print(f"Columns: {len(df.columns)}")

    profile = []

    for column in df.columns:

        series = df[column]

        missing_count = int(series.isna().sum())

        missing_percent = round(series.isna().mean() * 100, 2)

        semantic_missing_count = int(count_semantic_missing(series))

        semantic_missing_percent = round(

            semantic_missing_count / len(df) * 100,

            2

        )

        unique_values = int(series.nunique(dropna=True))

        zero_count = 0

        if pd.api.types.is_numeric_dtype(series):

            zero_count = int((series == 0).sum())

        profile.append(

            {

                "column": column,

                "dtype": str(series.dtype),

                "rows": len(df),

                "missing_count": missing_count,

                "missing_percent": missing_percent,

                "semantic_missing_count": semantic_missing_count,

                "semantic_missing_percent": semantic_missing_percent,

                "unique_values": unique_values,

                "zero_count": zero_count,

                "sample_values": sample_values(series),

            }

        )

    profile_df = pd.DataFrame(profile)

    profile_df["effective_missing_count"] = (

        profile_df["missing_count"]

        + profile_df["semantic_missing_count"]

    )

    profile_df["effective_missing_percent"] = (

        profile_df["effective_missing_count"] / len(df) * 100

    ).round(2)

    profile_df = profile_df.sort_values(

        by="effective_missing_percent",

        ascending=False,

    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    profile_df.to_csv(OUTPUT_PATH, index=False)

    print("\nData-quality profile created.")

    print(f"Saved to: {OUTPUT_PATH}")

    print("\nTop 15 columns by effective missingness:\n")

    print(

        profile_df[

            [

                "column",

                "missing_percent",

                "semantic_missing_percent",

                "effective_missing_percent",

                "unique_values",

            ]

        ]

        .head(15)

        .to_string(index=False)

    )

if __name__ == "__main__":
    main()