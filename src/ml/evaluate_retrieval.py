"""
Phase 8 - Retrieval evaluation.

There are no human relevance judgments for this corpus, so retrieval
is scored with a proxy task that needs no labels:

    split each ticket's description in half
    index the SECOND half
    query with the FIRST half
    ask whether the ticket finds itself

The halves share no text, so a hit means the backend connected two
different passages about the same incident - which is exactly what
retrieval has to do for a new ticket. It is a proxy, not a ground
truth, and the write-up should say so.

Also measures the score distribution for deliberately out-of-domain
queries, which is how SIMILARITY_FLOOR gets set from data rather than
guessed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend/python"))

from retriever import TicketRetriever, load_corpus  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "data/processed/retrieval_evaluation.csv"

MIN_WORDS_TO_SPLIT = 20
TOP_K = 10
K_VALUES = [1, 3, 5, 10]

# Queries about things this corpus does not contain. Their top scores
# are the noise floor: whatever a backend returns for these is what it
# returns when nothing relevant exists.
OUT_OF_DOMAIN = [
    "my vpn keeps disconnecting after the latest windows update",
    "the office printer is jammed and showing a paper error",
    "how do i reset my kubernetes cluster node pool",
    "laptop battery drains overnight while powered off",
    "coffee machine in the break room is leaking water",
]

SEPARATOR = "=" * 70


def section(title):
    print("\n" + SEPARATOR)
    print(title)
    print(SEPARATOR)


def build_split_corpus():
    """
    First half becomes the query, second half becomes the document.
    No shared text, so lexical overlap cannot carry the match.
    """

    df = load_corpus()

    words = df["text"].str.split()
    df = df[words.str.len() >= MIN_WORDS_TO_SPLIT].copy()

    def first_half(text):
        parts = str(text).split()
        return " ".join(parts[: len(parts) // 2])

    def second_half(text):
        parts = str(text).split()
        return " ".join(parts[len(parts) // 2:])

    df["query_half"] = df["text"].apply(first_half)
    df["text"] = df["text"].apply(second_half)

    return df.reset_index(drop=True)


def evaluate(backend, corpus):

    retriever = TicketRetriever(backend=backend).build(corpus=corpus)

    ranks = []
    correct_scores = []
    top1_scores = []

    for index, row in corpus.iterrows():

        hits = retriever.search(
            row["query_half"], k=TOP_K, min_similarity=0.0
        )

        if not hits:
            ranks.append(None)
            continue

        top1_scores.append(hits[0]["similarity"])

        rank = None

        for hit in hits:
            if hit["corpus_index"] == index:
                rank = hit["rank"]
                correct_scores.append(hit["similarity"])
                break

        ranks.append(rank)

    found = [r for r in ranks if r is not None]

    metrics = {"backend": backend, "queries": len(ranks)}

    for k in K_VALUES:
        hits_at_k = sum(1 for r in found if r <= k)
        metrics[f"recall@{k}"] = hits_at_k / len(ranks)

    # Mean reciprocal rank: 1.0 if always first, 0.5 if always second.
    metrics["mrr"] = float(
        np.mean([1 / r if r else 0.0 for r in ranks])
    )

    metrics["median_correct_score"] = (
        float(np.median(correct_scores)) if correct_scores else 0.0
    )
    metrics["p10_correct_score"] = (
        float(np.percentile(correct_scores, 10)) if correct_scores else 0.0
    )

    # Out-of-domain noise floor
    ood = [
        retriever.search(q, k=1, min_similarity=0.0)[0]["similarity"]
        for q in OUT_OF_DOMAIN
    ]

    metrics["ood_max_score"] = float(max(ood))
    metrics["ood_mean_score"] = float(np.mean(ood))

    return metrics, ood


def main():

    print(SEPARATOR)
    print("PHASE 8 - RETRIEVAL EVALUATION")
    print(SEPARATOR)

    corpus = build_split_corpus()

    print(f"\nTickets long enough to split: {len(corpus):,}")
    print(f"Queries: first half, Documents: second half\n")

    rows = []

    for backend in ["tfidf", "embedding"]:

        section(f"BACKEND: {backend}")

        try:
            metrics, ood = evaluate(backend, corpus)
        except ImportError as exc:
            print(f"\n  skipped - {exc}")
            continue

        rows.append(metrics)

        print()
        for k in K_VALUES:
            print(f"  recall@{k:<3} {metrics[f'recall@{k}']:.4f}")

        print(f"  MRR       {metrics['mrr']:.4f}")

        print(f"\n  correct-hit similarity: "
              f"median {metrics['median_correct_score']:.3f}, "
              f"p10 {metrics['p10_correct_score']:.3f}")

        print(f"\n  out-of-domain top-1 scores:")
        for query, score in zip(OUT_OF_DOMAIN, ood):
            print(f"    {score:.3f}  {query[:52]}")

        gap = metrics["p10_correct_score"] - metrics["ood_max_score"]

        print(f"\n  separation (p10 correct - max noise): {gap:+.3f}")

        if gap > 0:
            floor = (
                metrics["p10_correct_score"]
                + metrics["ood_max_score"]
            ) / 2
            print(f"  suggested SIMILARITY_FLOOR: {floor:.3f}")
        else:
            print("  [!] noise overlaps real matches - no clean floor")

    if rows:
        pd.DataFrame(rows).to_csv(REPORT_PATH, index=False)
        section("COMPARISON")
        print()
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        print(f"\n  {REPORT_PATH}")


if __name__ == "__main__":
    main()