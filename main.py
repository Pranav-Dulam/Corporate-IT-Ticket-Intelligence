"""
Phase 10 - CLI application.

    python3 main.py                      interactive
    python3 main.py "ticket text"        one-shot
    python3 main.py --no-llm "..."       retrieval only
    python3 main.py --verbose "..."      show every warning

The embedding index is cached to disk and reloaded, so startup cost is
paid once rather than on every ticket. The cache is invalidated
automatically when the underlying data changes.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend/python"))

from advisor import Advisor                                    # noqa: E402
from retriever import (                                        # noqa: E402
    INDEX_DIR,
    QUALITY_PATH,
    UNIFIED_PATH,
    TicketRetriever,
)

FEEDBACK_PATH = PROJECT_ROOT / "data/processed/feedback.csv"

MIN_WORDS = 3
MAX_WORDS = 2000

WIDTH = 60
RULE = "=" * WIDTH
THIN = "-" * WIDTH

# Fires on every helpdesk ticket by construction - the category model
# is permanently out of domain. A warning that always fires trains
# people to ignore all warnings, so it is verbose-only.
ALWAYS_ON_WARNINGS = [
    "category prediction is low-confidence",
]


# ---------------------------------------------------------
# Index cache
# ---------------------------------------------------------

def index_is_stale():
    """
    The cache is stale if either source file is newer than it.
    """

    marker = INDEX_DIR / "retriever_embedding.joblib"

    if not marker.exists():
        return True

    built_at = marker.stat().st_mtime

    for source in (UNIFIED_PATH, QUALITY_PATH):
        if source.exists() and source.stat().st_mtime > built_at:
            return True

    return False


def load_retriever(rebuild=False):

    if rebuild or index_is_stale():
        print("Building retrieval index (first run or data changed)...")
        retriever = TicketRetriever(backend="embedding").build()
        retriever.save()
        print(f"Indexed {len(retriever.corpus)} tickets.\n")
        return retriever

    retriever = TicketRetriever.load(backend="embedding")
    print(f"Loaded index: {len(retriever.corpus)} tickets.")

    # Load the sentence encoder now rather than on the first query.
    # On-call, startup latency is expected and query latency is not.
    started = time.perf_counter()
    retriever.search("warmup", k=1)
    print(f"Encoder ready in {time.perf_counter() - started:.1f}s.\n")

    return retriever


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

def validate(text):
    """
    Returns (cleaned_text, error_message).
    """

    text = str(text or "").strip()

    if not text:
        return None, "Please describe the issue."

    words = text.split()

    if len(words) < MIN_WORDS:
        return None, (
            f"Too short to analyse - {len(words)} word(s). "
            f"Describe the problem in at least {MIN_WORDS}."
        )

    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
        print(f"  [truncated to {MAX_WORDS} words]\n")

    return text, None


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

def visible_warnings(warnings, verbose):

    if verbose:
        return warnings

    return [
        w for w in warnings
        if not any(skip in w for skip in ALWAYS_ON_WARNINGS)
    ]


def display(result, verbose=False):

    if result.get("error"):
        print(f"\n  {result['error']}\n")
        return

    priority = result["priority"]
    category = result["category"]

    print(f"\n{THIN}")

    if priority["label"] and not priority["uncertain"]:
        confidence = (
            f"   Confidence: {priority['confidence']:.0%}"
            if priority["confidence"] else ""
        )
        print(f"  Priority: {priority['label']}{confidence}")
    else:
        print("  Priority: uncertain")

    # Shown only when the model is actually confident. On helpdesk
    # text it never is, which is the honest outcome.
    if category["label"] and not category["uncertain"]:
        print(f"  Category: {category['label']}   "
              f"Confidence: {category['confidence']:.0%}")

    print(THIN)

    hits = result["similar"]

    strong = [h for h in hits if not h.get("weak")]

    print(f"\n  {len(hits)} incident(s) retrieved "
          f"({len(strong)} strong):\n")

    for hit in hits[:5]:
        marker = "  weak" if hit.get("weak") else "strong"
        snippet = " ".join(str(hit["description"]).split())[:96]
        print(f"    [{marker}] {hit['similarity']:.0%}  {snippet}...")

    if not strong and len(result["description"].split()) < 10:
        print("\n  Your description is brief. Retrieval works better "
              "with\n  specifics - the error message, the application, "
              "what you\n  already tried.")

    if result["steps"]:
        print(f"\n{THIN}")
        print("  RECOMMENDED RESOLUTION")
        print(THIN + "\n")

        for i, step in enumerate(result["steps"], 1):
            print(f"  {i}. {step['text']}")

        quality = result.get("evidence_quality")

        print(f"\n  Source: {result.get('source')}"
              + (f"   Evidence: {quality}" if quality else ""))

    if result.get("explanation"):
        print(f"\n  {result['explanation']}")

    shown = visible_warnings(result["warnings"], verbose)

    if shown:
        print(f"\n{THIN}")
        print("  WARNINGS")
        print(THIN + "\n")
        for warning in shown:
            print(f"  ! {warning}")

    hidden = len(result["warnings"]) - len(shown)

    if hidden and not verbose:
        print(f"\n  ({hidden} further note(s) - run with --verbose)")

    print()


# ---------------------------------------------------------
# Feedback
# ---------------------------------------------------------

def record_feedback(description, result, useful):

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    is_new = not FEEDBACK_PATH.exists()

    with open(FEEDBACK_PATH, "a", newline="") as handle:

        writer = csv.writer(handle)

        if is_new:
            writer.writerow(
                ["timestamp", "description", "source",
                 "num_similar", "num_steps", "useful"]
            )

        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                description[:300],
                result.get("source"),
                len(result.get("similar", [])),
                len(result.get("steps", [])),
                useful,
            ]
        )


def ask_feedback(description, result):

    if not result.get("steps"):
        return

    try:
        answer = input("  Was this useful? [y/n/skip]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if answer in {"y", "yes"}:
        record_feedback(description, result, "yes")
        print("  Recorded. Thanks.\n")
    elif answer in {"n", "no"}:
        record_feedback(description, result, "no")
        print("  Recorded. Thanks.\n")
    else:
        print()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def analyse(advisor, text, verbose, use_llm, feedback=True):

    cleaned, error = validate(text)

    if error:
        print(f"\n  {error}\n")
        return

    print("\n  Analysing ticket...")

    started = time.perf_counter()
    result = advisor.advise(cleaned, use_llm=use_llm)
    elapsed = time.perf_counter() - started

    display(result, verbose=verbose)

    print(f"  [answered in {elapsed:.2f}s]\n")

    if feedback:
        ask_feedback(cleaned, result)


def main():

    parser = argparse.ArgumentParser(
        description="AI IT ticket analysis and resolution system."
    )
    parser.add_argument("description", nargs="*",
                        help="Ticket text. Omit for interactive mode.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Retrieval only, no generation.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all warnings.")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force a rebuild of the retrieval index.")

    args = parser.parse_args()

    print(f"\n{RULE}")
    print("  AI IT TICKET ANALYSIS & RESOLUTION SYSTEM")
    print(RULE + "\n")

    try:
        retriever = load_retriever(rebuild=args.rebuild_index)
    except Exception as exc:
        print(f"  Could not prepare the retrieval index: {exc}")
        print("  Run the Phase 4 and Phase 8 scripts first.\n")
        return 1

    advisor = Advisor(retriever=retriever)

    use_llm = not args.no_llm

    # One-shot mode: no feedback prompt, so it stays scriptable.
    if args.description:
        analyse(
            advisor,
            " ".join(args.description),
            args.verbose,
            use_llm,
            feedback=False,
        )
        return 0

    print("  Describe an issue, or type 'quit' to exit.\n")

    while True:

        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.\n")
            return 0

        if text.lower() in {"quit", "exit", "q"}:
            print("\n  Goodbye.\n")
            return 0

        analyse(advisor, text, args.verbose, use_llm)


if __name__ == "__main__":
    sys.exit(main())