"""
Topic assignment for new tickets.

Nearest discovered centroid, cosine similarity as confidence. A ticket
further from every centroid than the assignment floor is reported as
unassigned rather than forced into the closest bucket.
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPIC_DIR = PROJECT_ROOT / "models/topics"


class TopicClassifier:

    def __init__(self):
        self.centroids = None
        self.meta = None
        self.error = None

    def load(self):

        if self.centroids is not None or self.error:
            return self

        try:
            self.centroids = np.load(TOPIC_DIR / "centroids.npy")
            self.meta = json.loads(
                (TOPIC_DIR / "meta.json").read_text()
            )
        except Exception as exc:
            self.error = f"topic model unavailable: {exc}"

        return self

    def assign(self, embedding):
        """
        `embedding` must be the same normalised vector the retriever
        produces, so cosine is a plain dot product.
        """

        self.load()

        if self.error:
            return {
                "label": None, "cluster": None, "confidence": None,
                "uncertain": True, "error": self.error,
            }

        vector = np.asarray(embedding).ravel()

        similarities = self.centroids @ vector

        cluster = int(np.argmax(similarities))
        confidence = float(similarities[cluster])

        floor = self.meta.get("assignment_floor", 0.0)
        terms = self.meta["labels"][str(cluster)]

        return {
            "label": ", ".join(terms[:3]),
            "cluster": cluster,
            "confidence": confidence,
            "uncertain": confidence < floor,
            "error": None,
        }