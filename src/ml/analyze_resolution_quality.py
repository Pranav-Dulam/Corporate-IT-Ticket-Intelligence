"""
Phase 8b - Resolution quality analysis.

The retrieval index is only as good as the resolutions it returns.
Phase 4 flagged `has_usable_resolution` on a 5-word minimum, which
boilerplate clears easily.

Boilerplate is found empirically rather than hand-listed: a sentence
appearing in a large share of ALL resolutions cannot be specific to
any one ticket. Filtering happens at sentence level, not n-gram level
- removing an n-gram strands the words around it, which is why an
earlier version left "your ticket has been" in every output.
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data/processed/unified_tickets.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/resolution_quality.csv"

# A whole sentence in this share of resolutions is boilerplate.
# A real fix appears in one ticket; greetings appear in hundreds.
SENTENCE_DF_THRESHOLD = 0.02

# Minimum surviving words. Deliberately low - "clear the browser cache
# and try again" is a complete, useful resolution.
MIN_CONTENT_WORDS = 5

# Greeting and sign-off fragments that ride along inside otherwise
# useful sentences, so document frequency never isolates them.
EDGE_PATTERNS = [
    r"^\s*(greetings|dears?|hi|hello)\b[,:]?\s*",
    r"\b(best\s+)?regards\s*[,.]?\s*(ph_\w+\s*)*$",
    r"\bthanks?\s+(and\s+)?(regards)?\s*[,.]?\s*$",
]

SEPARATOR = "=" * 70


def section(title):
    print("\n" + SEPARATOR)
    print(title)
    print(SEPARATOR)


# ---------------------------------------------------------
# Sentence handling
# ---------------------------------------------------------

def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    return [p.strip() for p in parts if p.strip()]


def sentence_key(sentence):
    """Normalized form used only for matching, never for output."""
    key = re.sub(r"[^a-z0-9_\s]", " ", str(sentence).lower())
    return re.sub(r"\s+", " ", key).strip()


def dedupe_sentences(text):
    """
    One resolution repeats "we are following up ... will update once
    get a feedback" three times. Repetition is not extra evidence.
    """

    seen = set()
    kept = []

    for sentence in split_sentences(text):
        key = sentence_key(sentence)
        if key and key not in seen:
            seen.add(key)
            kept.append(sentence)

    return " ".join(kept)


def find_boilerplate_sentences(resolutions):
    """
    Document frequency over whole sentences.

    A sentence is removed whole or not at all, so nothing is stranded.
    """

    doc_frequency = Counter()

    for text in resolutions:
        keys = {
            sentence_key(trim_edges(s))
            for s in split_sentences(text)
        }
        for key in keys:
            if key:
                doc_frequency[key] += 1

    total = len(resolutions)

    return {
        key: count / total
        for key, count in doc_frequency.items()
        if count / total >= SENTENCE_DF_THRESHOLD
    }


def trim_edges(sentence):
    cleaned = sentence
    for pattern in EDGE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,.;:")


def extract_content(text, boilerplate):
    """
    Keep non-boilerplate sentences in original casing and punctuation -
    this text becomes the LLM's grounding context in Phase 9, so it has
    to stay readable.
    """

    kept = []

    for sentence in split_sentences(dedupe_sentences(text)):

        if sentence_key(sentence) in boilerplate:
            continue

        trimmed = trim_edges(sentence)

        if len(trimmed.split()) >= 3:
            kept.append(trimmed)

    return " ".join(kept)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print(SEPARATOR)
    print("RESOLUTION QUALITY ANALYSIS")
    print(SEPARATOR)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = df[df["retrievable"]].copy().reset_index(drop=True)

    print(f"\nTickets currently flagged retrievable: {len(df):,}")

    resolutions = df["resolution"].fillna("").tolist()

    # -----------------------------------------------------
    section(f"1. BOILERPLATE SENTENCES "
            f"(>= {SENTENCE_DF_THRESHOLD:.0%} of resolutions)")

    boilerplate = find_boilerplate_sentences(resolutions)

    print(f"\n  {len(boilerplate)} boilerplate sentences found.\n")

    for key, share in sorted(
        boilerplate.items(), key=lambda kv: -kv[1]
    )[:20]:
        print(f"  {share:5.1%}  {key[:90]}")

    # -----------------------------------------------------
    section("2. CONTENT REMAINING AFTER FILTERING")

    df["resolution_content"] = df["resolution"].apply(
        lambda t: extract_content(t, boilerplate)
    )

    df["content_words"] = (
        df["resolution_content"].str.split().str.len().fillna(0).astype(int)
    )

    original = df["resolution"].fillna("").str.split().str.len()

    print("\nWord count before filtering:\n")
    print(original.describe().round(1).to_string())

    print("\nWord count after filtering:\n")
    print(df["content_words"].describe().round(1).to_string())

    removed = 100 * (1 - df["content_words"].sum() / original.sum())
    print(f"\n  {removed:.1f}% of all resolution text was boilerplate.")

    # -----------------------------------------------------
    section("3. HOW MANY TICKETS SURVIVE A CONTENT FLOOR")

    print()
    for floor in [5, 10, 20, 30, 50]:
        kept = int((df["content_words"] >= floor).sum())
        print(f"  >= {floor:>3} content words: {kept:>4} tickets "
              f"({100 * kept / len(df):5.1f}%)")

    # -----------------------------------------------------
    section(f"4. SAMPLES AT THE {MIN_CONTENT_WORDS}-WORD FLOOR")

    good = df[df["content_words"] >= MIN_CONTENT_WORDS]
    bad = df[df["content_words"] < MIN_CONTENT_WORDS]

    print(f"\n--- KEPT ({len(good)}) ---\n")
    for _, row in good.head(4).iterrows():
        print(f"  [{row['content_words']} words] "
              f"{row['resolution_content'][:300]}\n")

    print(f"\n--- DROPPED ({len(bad)}) ---\n")
    for _, row in bad.head(5).iterrows():
        print(f"  [{row['content_words']} words] "
              f"{str(row['resolution'])[:200]}\n")

    # -----------------------------------------------------
    df[
        [
            "ticket_id",
            "category",
            "priority",
            "content_words",
            "resolution_content",
        ]
    ].to_csv(OUTPUT_PATH, index=False)

    section("OUTPUT")
    print(f"\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()