"""
Phase 9 - Resolution recommendation (retrieval mode).

Builds a recommendation from retrieved historical resolutions only.
No generative model is involved, which makes this the fallback path
when the LLM API is unavailable - a requirement of the project spec,
and the reason it is built first.

Steps are extracted from what human engineers actually wrote on
similar tickets. Nothing is invented; if the evidence is thin, the
output says so rather than filling the gap.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predictor import Predictor               # noqa: E402
from retriever import TicketRetriever, SIMILARITY_FLOOR  # noqa: E402


DEFAULT_TOP_K = 5
MAX_STEPS = 6
MIN_STEP_WORDS = 4

# Sentences that describe process rather than action. Phase 8b removed
# the worst boilerplate; these survive because they vary per ticket.
NON_ACTION_PATTERNS = [
    r"\bwe will\b",
    r"\bwe are (following|checking|waiting|communicat)",
    r"\bwaiting (for )?your\b",
    r"\bkind(ly)? reminder\b",
    r"\bplease (advise|confirm|suggest a time|join the below)\b",
    r"\bticket (is |has been )?(assigned|closed|resolved)\b",
    r"\b(close|closing) (this|the) ticket\b",
    r"\bconsider (it|this ticket) (as )?(resolved|closed)\b",
    r"\bopen a new ticket\b",
    r"\bany update\b",
    r"\bwhat is the error\b",
]

# A sentence is a step only if it names an action. This is a positive
# test rather than "is not a question": the most useful line in the IE
# ticket - "can you check ... trusted sites and compatibility mode?" -
# is phrased as a polite question but IS the diagnostic step.
ACTION_VERBS = [
    "check", "verify", "clear", "restart", "reboot", "reset",
    "add", "remove", "delete", "update", "upgrade", "install",
    "reinstall", "configure", "enable", "disable", "try",
    "run", "set", "change", "replace", "apply", "deploy",
    "connect", "login", "log in", "select", "click", "navigate",
    "ensure", "make sure", "send us", "provide", "use",
]

ACTION_PATTERN = re.compile(
    r"\b(" + "|".join(ACTION_VERBS) + r")\b", re.IGNORECASE
)


def is_actionable(sentence):
    """
    Keep sentences that tell someone to do something.

    "clear the browser cache and try to login" is a step.
    "we will check and get back to you" is not.
    """

    if len(sentence.split()) < MIN_STEP_WORDS:
        return False

    lowered = sentence.lower()

    for pattern in NON_ACTION_PATTERNS:
        if re.search(pattern, lowered):
            return False

    return bool(ACTION_PATTERN.search(lowered))


def extract_steps(hits, limit=MAX_STEPS):
    """
    Pull actionable sentences out of retrieved resolutions, best match
    first, de-duplicated.
    """

    steps = []
    seen = set()

    for hit in hits:

        text = str(hit.get("resolution") or "")

        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):

            sentence = sentence.strip(" .,;:")

            if not sentence or not is_actionable(sentence):
                continue

            key = re.sub(r"[^a-z0-9 ]", "", sentence.lower()).strip()

            if key in seen:
                continue

            seen.add(key)

            steps.append(
                {
                    "text": sentence,
                    "from_ticket": hit["ticket_id"],
                    "similarity": hit["similarity"],
                }
            )

            if len(steps) >= limit:
                return steps

    return steps


def build_warnings(hits, prediction, steps):
    """
    Everything the user should distrust about this answer.
    """

    warnings = []

    if not hits:
        warnings.append(
            "No historical incident scored above the "
            f"{SIMILARITY_FLOOR} similarity threshold. No grounded "
            "recommendation is available."
        )
    elif len(hits) < 3:
        warnings.append(
            f"Only {len(hits)} similar incident(s) found. The "
            "recommendation rests on thin evidence."
        )

    if hits and hits[0]["similarity"] < SIMILARITY_FLOOR + 0.10:
        warnings.append(
            "The closest match is only marginally above the "
            "similarity threshold."
        )

    if hits and not steps:
        warnings.append(
            "Similar incidents were found, but none of their "
            "resolutions contained actionable steps."
        )

    for field in ("category", "priority"):
        if prediction[field]["uncertain"]:
            warnings.append(
                f"The {field} prediction is low-confidence and may "
                "be unreliable."
            )

    for error in prediction["model_errors"]:
        warnings.append(error)

    return warnings


class Recommender:

    def __init__(self, backend="embedding", retriever=None):
        self.predictor = Predictor()

        # An already-built retriever can be injected so the CLI loads
        # a saved index instead of re-encoding 317 tickets per run.
        self.retriever = (
            retriever
            if retriever is not None
            else TicketRetriever(backend=backend).build()
        )

    def recommend(self, description, k=DEFAULT_TOP_K):

        description = str(description or "").strip()

        if len(description.split()) < 3:
            return {
                "error": "Description too short to analyse.",
                "warnings": [],
                "similar": [],
            }

        prediction = self.predictor.predict(description)
        hits = self.retriever.search(description, k=k)
        steps = extract_steps(hits)

        return {
            "description": description,
            "category": prediction["category"],
            "priority": prediction["priority"],
            "similar": hits,
            "steps": steps,
            "explanation": (
                f"These steps were taken from {len({s['from_ticket'] for s in steps})} "
                f"historical incident(s) with similar descriptions. They are "
                f"what engineers did before, not a verified fix for this case."
                if steps else
                "No actionable steps could be extracted from the retrieved "
                "incidents."
            ),
            "source": "retrieval",
            "warnings": build_warnings(hits, prediction, steps),
        }


def main():

    query = " ".join(sys.argv[1:])

    if not query:
        print("usage: python3 recommender.py <ticket description>")
        return

    result = Recommender().recommend(query)

    print("\n" + "=" * 68)
    print(" RESOLUTION RECOMMENDATION")
    print("=" * 68)

    if result.get("error"):
        print(f"\n  {result['error']}\n")
        return

    category = result["category"]
    priority = result["priority"]

    def fmt(p):
        if p["label"] is None:
            return "unavailable"
        conf = f" ({p['confidence']:.0%})" if p["confidence"] else ""
        flag = "  [low confidence]" if p["uncertain"] else ""
        return f"{p['label']}{conf}{flag}"

    print(f"\n  Category: {fmt(category)}")
    print(f"  Priority: {fmt(priority)}")

    print(f"\n  Similar incidents: {len(result['similar'])}")

    for hit in result["similar"][:3]:
        print(f"    {hit['similarity']:.3f}  {hit['ticket_id']}  "
              f"{str(hit['description'])[:60]}...")

    if result["steps"]:
        print("\n" + "-" * 68)
        print(" RECOMMENDED STEPS  (from historical resolutions)")
        print("-" * 68 + "\n")

        for i, step in enumerate(result["steps"], 1):
            print(f"  {i}. {step['text']}")
            print(f"     [{step['from_ticket']}, "
                  f"similarity {step['similarity']:.3f}]\n")

    print(f"  {result['explanation']}\n")

    if result["warnings"]:
        print("-" * 68)
        print(" WARNINGS")
        print("-" * 68 + "\n")
        for warning in result["warnings"]:
            print(f"  ! {warning}")
        print()


if __name__ == "__main__":
    main()