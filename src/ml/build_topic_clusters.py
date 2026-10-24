"""
Unsupervised topic discovery.

The Phase 6 category model predicts Apache Jira project names, which
are meaningless for helpdesk text - it reports ~22% confidence on a
browser ticket and the CLI hides it. That is a labelling problem, not
a modelling one: the only labelled data came from the wrong domain.

So the taxonomy is discovered instead of imposed. The Phase 8
embeddings are already computed and already capture topical
similarity; clustering them yields categories that exist in this
corpus rather than in Apache's issue tracker.

k is chosen by silhouette score rather than by hand. Cluster labels
come from distinctive TF-IDF terms - the terms a cluster uses more
than the corpus average.
"""

import json
import sys
from pathlib import Path

import re

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend/python"))

from retriever import TicketRetriever  # noqa: E402


TOPIC_DIR = PROJECT_ROOT / "models/topics"
REPORT_PATH = PROJECT_ROOT / "data/processed/topic_clusters.csv"

K_RANGE = range(3, 16)
TOP_TERMS = 6
RANDOM_SEED = 42

PLACEHOLDER_PATTERN = re.compile(r"\bph_\w+\b", re.IGNORECASE)

def strip_placeholders(texts):
    """
    ph_user / ph_technical / ph_item are anonymisation masks, not
    content. They are among the most distinctive terms in every
    cluster, which suggests the embeddings may be grouping tickets by
    which masks they contain rather than by what went wrong.
    """
    cleaned = []
    for text in texts:
        text = PLACEHOLDER_PATTERN.sub(" ", str(text))
        cleaned.append(re.sub(r"\s+", " ", text).strip())
    return cleaned

SEPARATOR = "=" * 70


def section(title):
    print("\n" + SEPARATOR)
    print(title)
    print(SEPARATOR)


def choose_k(matrix):
    """
    Silhouette over a range. Higher is better; it measures how much
    tighter a point sits with its own cluster than the next nearest.
    """

    scores = []

    for k in K_RANGE:

        model = KMeans(
            n_clusters=k, random_state=RANDOM_SEED, n_init=10
        ).fit(matrix)

        score = silhouette_score(matrix, model.labels_, metric="cosine")

        scores.append((k, score, model))

        print(f"  k={k:<3} silhouette {score:.4f}")

    best = max(scores, key=lambda row: row[1])

    return best[0], best[1], best[2]


def distinctive_terms(texts, labels, k):
    """
    Terms a cluster uses more than the corpus average.

    Raw top-TF-IDF would return 'please' and 'ph_user' for every
    cluster; subtracting the global mean leaves what is specific.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=3,
        max_features=5000,
    )

    matrix = vectorizer.fit_transform(texts)
    vocabulary = np.array(vectorizer.get_feature_names_out())

    global_mean = np.asarray(matrix.mean(axis=0)).ravel()

    terms = {}

    for cluster in range(k):

        rows = matrix[labels == cluster]

        if rows.shape[0] == 0:
            terms[cluster] = []
            continue

        cluster_mean = np.asarray(rows.mean(axis=0)).ravel()

        lift = cluster_mean - global_mean

        top = np.argsort(lift)[::-1][:TOP_TERMS]

        terms[cluster] = vocabulary[top].tolist()

    return terms


def main():

    print(SEPARATOR)
    print("UNSUPERVISED TOPIC DISCOVERY")
    print(SEPARATOR)

    try:
        retriever = TicketRetriever.load(backend="embedding")
    except FileNotFoundError:
        print("\nNo saved index. Run:\n"
              "  python3 main.py --rebuild-index 'test'\n")
        return 1

    corpus = retriever.corpus
    matrix = np.asarray(retriever.matrix)

    if "--strip-placeholders" in sys.argv:
        from sentence_transformers import SentenceTransformer

        print("\nRe-encoding with placeholders removed...")

        model = SentenceTransformer("all-MiniLM-L6-v2")

        matrix = model.encode(
            strip_placeholders(corpus["text"].tolist()),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    print(f"\nTickets: {len(corpus)}   embedding dim: {matrix.shape[1]}")

    # -----------------------------------------------------
    section("1. CHOOSING k")

    print()
    k, score, model = choose_k(matrix)

    print(f"\n  best k = {k}  (silhouette {score:.4f})")

    if score < 0.05:
        print("\n  [!] Silhouette is very low. The clusters barely\n"
              "      separate - treat these topics as weak structure,\n"
              "      not clean categories.")

    labels = model.labels_

    # Centroids are means of normalised vectors, so renormalise to
    # keep cosine assignment meaningful at inference time.
    centroids = model.cluster_centers_
    centroids = centroids / np.linalg.norm(
        centroids, axis=1, keepdims=True
    )

    # -----------------------------------------------------
    section("2. CLUSTERS")

    terms = distinctive_terms(corpus["text"].tolist(), labels, k)

    similarities = (matrix @ centroids.T)
    assigned_similarity = similarities[np.arange(len(labels)), labels]

    rows = []

    for cluster in range(k):

        mask = labels == cluster
        size = int(mask.sum())

        label_text = ", ".join(terms[cluster])

        print(f"\n  CLUSTER {cluster}  ({size} tickets)")
        print(f"    terms: {label_text}")

        for text in corpus.loc[mask, "text"].head(2):
            snippet = " ".join(str(text).split())[:110]
            print(f"    - {snippet}...")

        rows.append(
            {
                "cluster": cluster,
                "size": size,
                "label": label_text,
                "mean_similarity": float(
                    assigned_similarity[mask].mean()
                ),
            }
        )

    # -----------------------------------------------------
    section("3. ASSIGNMENT THRESHOLD")

    # A new ticket further from every centroid than the weakest
    # in-corpus member is outside the discovered taxonomy.
    floor = float(np.percentile(assigned_similarity, 10))

    print(f"\n  in-corpus similarity: "
          f"median {np.median(assigned_similarity):.3f}, "
          f"p10 {floor:.3f}")
    print(f"  assignment floor: {floor:.3f}")

    # -----------------------------------------------------
    TOPIC_DIR.mkdir(parents=True, exist_ok=True)

    np.save(TOPIC_DIR / "centroids.npy", centroids)

    (TOPIC_DIR / "meta.json").write_text(
        json.dumps(
            {
                "k": int(k),
                "silhouette": float(score),
                "assignment_floor": floor,
                "labels": {str(c): terms[c] for c in range(k)},
                "n_tickets": int(len(corpus)),
            },
            indent=2,
        )
    )

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)

    section("OUTPUT")
    print(f"\n  {TOPIC_DIR}/centroids.npy")
    print(f"  {TOPIC_DIR}/meta.json")
    print(f"  {REPORT_PATH}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())