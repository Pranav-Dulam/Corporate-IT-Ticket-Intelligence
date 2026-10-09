"""
Phase 8 - Semantic Ticket Retrieval.

Finds historical tickets similar to a new description, and returns
their resolutions as grounding material for Phase 9.

Two backends:

  tfidf      - sparse lexical matching. Always available. Strong when
               the query reuses the corpus vocabulary, blind to
               paraphrase.
  embedding  - dense sentence embeddings. Handles paraphrase, needs
               sentence-transformers installed.

No vector database. The corpus is 327 tickets, so similarity is one
dot product against a matrix that fits comfortably in memory. A vector
index would add a dependency, a build step and a failure mode in
exchange for nothing measurable at this scale.

Only rows flagged `retrievable` in unified_tickets.csv are indexed -
that flag means the ticket has both a usable description and a real
resolution. Phase 8's requirement is that tickets without useful
resolutions never pollute the recommendation context, and this is
where that is enforced.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNIFIED_PATH = PROJECT_ROOT / "data/processed/unified_tickets.csv"
QUALITY_PATH = PROJECT_ROOT / "data/processed/resolution_quality.csv"
INDEX_DIR = PROJECT_ROOT / "models/embeddings"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MAX_WORDS = 512
DEFAULT_TOP_K = 5

# Phase 8b: 10 of 327 tickets are pure acknowledgment.
MIN_CONTENT_WORDS = 5

# Below this cosine score, nothing meaningful matched.
# Set empirically in Phase 8 - see evaluate_retrieval.py.
SIMILARITY_FLOOR = 0.45

PAYLOAD_COLUMNS = [
    "ticket_id",
    "description",
    "category",
    "priority",
    "resolution",
    "source",
]


# ---------------------------------------------------------
# Corpus
# ---------------------------------------------------------

def load_corpus():
    """
    The retrievable subset of the unified table.
    """

    df = pd.read_csv(UNIFIED_PATH, low_memory=False)

    df = df[df["retrievable"]].copy()

    # Swap raw resolutions for the boilerplate-filtered text.
    # 22.6% of the original was greetings and acknowledgments,
    # which would otherwise become the LLM's evidence in Phase 9.
    quality = pd.read_csv(QUALITY_PATH)

    df = df.merge(
        quality[["ticket_id", "resolution_content", "content_words"]],
        on="ticket_id",
        how="inner",
    )

    df = df[df["content_words"] >= MIN_CONTENT_WORDS].copy()
    df["resolution"] = df["resolution_content"]

    normalized = (
        df["description"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df[~normalized.duplicated(keep="first")].copy()

    df["text"] = df["description"].apply(
        lambda t: " ".join(str(t).split()[:MAX_WORDS])
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

class TicketRetriever:

    def __init__(self, backend="tfidf"):

        if backend not in {"tfidf", "embedding"}:
            raise ValueError(f"Unknown backend: {backend}")

        self.backend = backend
        self.corpus = None
        self.matrix = None
        self.vectorizer = None
        self._encoder = None

    # -----------------------------------------------------

    def _encoder_model(self):
        """
        Imported lazily so the tfidf backend never requires torch.
        """

        if self._encoder is None:

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. Use "
                    "backend='tfidf', or install it."
                ) from exc

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)

        return self._encoder

    # -----------------------------------------------------

    def build(self, corpus=None):

        self.corpus = load_corpus() if corpus is None else corpus

        texts = self.corpus["text"].tolist()

        if self.backend == "tfidf":

            # TfidfVectorizer L2-normalizes rows by default, so a dot
            # product between two rows is already cosine similarity.
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )

            self.matrix = self.vectorizer.fit_transform(texts)

        else:

            model = self._encoder_model()

            # normalize_embeddings makes dot product == cosine here too.
            self.matrix = model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

        return self

    # -----------------------------------------------------

    def _encode_query(self, query):

        if self.backend == "tfidf":
            return self.vectorizer.transform([query])

        return self._encoder_model().encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # -----------------------------------------------------

    def search(self, query, k=DEFAULT_TOP_K, exclude_index=None,
               min_similarity=None):
        """
        Return the k most similar tickets, most similar first.

        exclude_index drops one corpus row - used by the evaluation
        script for leave-one-out, where a ticket must not retrieve
        itself.
        """

        if self.matrix is None:
            raise RuntimeError("Index not built. Call build() or load().")

        if not str(query).strip():
            return []

        query_vector = self._encode_query(query)

        if self.backend == "tfidf":
            scores = (self.matrix @ query_vector.T).toarray().ravel()
        else:
            scores = self.matrix @ query_vector.ravel()

        if exclude_index is not None:
            scores[exclude_index] = -np.inf

        # Below the floor nothing meaningful matched. Returning
        # an empty list is the weak-evidence signal Phase 9 needs:
        # a 0.38 cosine must not be presented as a 38% match.
        floor = (
            SIMILARITY_FLOOR
            if min_similarity is None
            else min_similarity
        )

        top = np.argsort(scores)[::-1][:k]

        results = []

        for rank, index in enumerate(top, start=1):

            if not np.isfinite(scores[index]):
                continue

            if scores[index] < floor:
                continue

            row = self.corpus.iloc[index]

            results.append(
                {
                    "rank": rank,
                    "corpus_index": int(index),
                    "similarity": float(scores[index]),
                    **{
                        column: row[column]
                        for column in PAYLOAD_COLUMNS
                    },
                }
            )

        return results

    # -----------------------------------------------------

    def save(self):

        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        payload = {
            "backend": self.backend,
            "corpus": self.corpus,
            "vectorizer": self.vectorizer,
        }

        joblib.dump(payload, INDEX_DIR / f"retriever_{self.backend}.joblib")

        if self.backend == "embedding":
            np.save(INDEX_DIR / "embeddings.npy", self.matrix)
        else:
            joblib.dump(self.matrix, INDEX_DIR / "tfidf_matrix.joblib")

    @classmethod
    def load(cls, backend="tfidf"):

        payload = joblib.load(INDEX_DIR / f"retriever_{backend}.joblib")

        retriever = cls(backend=backend)
        retriever.corpus = payload["corpus"]
        retriever.vectorizer = payload["vectorizer"]

        if backend == "embedding":
            retriever.matrix = np.load(INDEX_DIR / "embeddings.npy")
        else:
            retriever.matrix = joblib.load(INDEX_DIR / "tfidf_matrix.joblib")

        return retriever


# ---------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="tfidf",
                        choices=["tfidf", "embedding"])
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    retriever = TicketRetriever(backend=args.backend).build()

    print(f"\nIndexed {len(retriever.corpus)} tickets "
          f"[{args.backend}]\n")

    if args.save:
        retriever.save()
        print(f"Index saved to {INDEX_DIR}\n")

    hits = retriever.search(args.query, k=args.k)

    if not hits:
        print(f"No similar incidents found - nothing scored "
              f"above the {SIMILARITY_FLOOR} similarity floor.\n")
        return

    for hit in hits:

        print("-" * 68)
        print(f"#{hit['rank']}  similarity {hit['similarity']:.3f}  "
              f"| {hit['category']} | {hit['priority']} | {hit['source']}")
        print(f"\n  {str(hit['description'])[:260]}")
        print(f"\n  RESOLUTION: {str(hit['resolution'])[:260]}")
        print()


if __name__ == "__main__":
    main()