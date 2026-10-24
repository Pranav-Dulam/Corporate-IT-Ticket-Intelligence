"""
Phase 9 - Model loading and prediction.

Wraps the Phase 6 and Phase 7 models behind one interface so the
recommender, the CLI and the Node backend never touch joblib directly.

Models are loaded once, lazily, and cached. Nothing retrains at
request time.

A missing or unreadable model is reported, not raised: the system is
required to stay useful when one component is unavailable, so a
prediction failure degrades to `None` with a reason attached.
"""

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

# First match wins. Ordered best-first so the system picks the
# strongest available model without the caller naming a file.
CATEGORY_MODELS = [
    "category_model_linearsvc_stripped_min10.joblib",
    "category_model_stripped.joblib",
    "category_model_raw.joblib",
]

PRIORITY_MODELS = [
    "priority_model_linearsvc_binary.joblib",
]

# Below this, a prediction is reported as uncertain rather than
# stated. The floor is PER MODEL because chance level differs: with
# 138 categories chance is ~0.007, with 2 priorities it is 0.50. A
# single shared floor of 0.35 could never fire for the binary model,
# so a coin-flip 54% was being presented as a confident answer.
CONFIDENCE_FLOORS = {
    "category": 0.35,
    "priority": 0.65,
}

DEFAULT_CONFIDENCE_FLOOR = 0.35


def _first_existing(filenames):
    for name in filenames:
        path = MODEL_DIR / name
        if path.exists():
            return path
    return None


class Predictor:

    def __init__(self):
        self._category = None
        self._priority = None
        self._loaded = False
        self.errors = []

    # -----------------------------------------------------

    def load(self):
        """
        Load both models. Each failure is recorded, not raised.
        """

        if self._loaded:
            return self

        for attr, candidates, label in [
            ("_category", CATEGORY_MODELS, "category"),
            ("_priority", PRIORITY_MODELS, "priority"),
        ]:
            path = _first_existing(candidates)

            if path is None:
                self.errors.append(
                    f"{label} model not found in {MODEL_DIR}"
                )
                continue

            try:
                setattr(self, attr, joblib.load(path))
            except Exception as exc:
                self.errors.append(
                    f"{label} model at {path.name} failed to load: {exc}"
                )

        self._loaded = True
        return self

    # -----------------------------------------------------

    def _predict_one(self, model, text, label):

        if model is None:
            return {
                "label": None,
                "confidence": None,
                "uncertain": True,
                "error": f"{label} model unavailable",
            }

        try:
            prediction = model.predict([text])[0]

            confidence = None

            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba([text])[0].max())

            return {
                "label": str(prediction),
                "confidence": confidence,
                "uncertain": (
                    confidence is not None
                    and confidence < CONFIDENCE_FLOORS.get(
                        label, DEFAULT_CONFIDENCE_FLOOR
                    )
                ),
                "error": None,
            }

        except Exception as exc:
            return {
                "label": None,
                "confidence": None,
                "uncertain": True,
                "error": f"{label} prediction failed: {exc}",
            }

    # -----------------------------------------------------

    def predict(self, text):

        self.load()

        return {
            "category": self._predict_one(
                self._category, text, "category"
            ),
            "priority": self._predict_one(
                self._priority, text, "priority"
            ),
            "model_errors": list(self.errors),
        }


if __name__ == "__main__":
    import json
    import sys

    query = " ".join(sys.argv[1:]) or "user cannot open link in internet explorer"

    print(json.dumps(Predictor().predict(query), indent=2))