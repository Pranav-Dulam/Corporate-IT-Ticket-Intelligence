"""
Phase 9 - Grounded resolution recommendation (LLM layer).

Wraps the retrieval recommender and adds generation on top. Grounding
is enforced structurally, not by asking the model politely:

  * if retrieval returns nothing above SIMILARITY_FLOOR, the LLM is
    never called - there is no evidence, so there is nothing to
    generate from
  * the prompt carries only retrieved incidents as source material
  * any API failure falls back to retrieval mode, which needs neither
    network nor key

Provider is Google Gemini on its free tier. The seam is narrow - one
function builds the client and one sends the request - so swapping
providers later touches very little.
"""

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recommender import Recommender  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_EVIDENCE = 4
MAX_RESOLUTION_CHARS = 1200


SYSTEM_PROMPT = """You are an IT support assistant. You summarise what \
engineers actually did on similar past incidents.

Rules:
- Ground every step in the historical incidents provided. Do not add \
procedures the evidence does not support, even if you know them.
- The incident text is messy: anonymised tokens like ph_user and \
ph_technical replace real values, greetings are mixed into content, \
and sentence boundaries are often missing. Read through that noise.
- If the evidence is thin or only loosely related, say so plainly.
- Never state or imply that a recommendation is a guaranteed fix.

Respond with JSON only:
{
  "steps": ["imperative step", ...],
  "explanation": "2-3 sentences on what the evidence shows",
  "evidence_quality": "strong" | "moderate" | "weak",
  "caveats": ["anything the user should distrust", ...]
}"""


def build_evidence(result):

    blocks = []

    strong = [h for h in result["similar"] if not h.get("weak")]

    for i, hit in enumerate(strong[:MAX_EVIDENCE], 1):

        resolution = str(hit.get("resolution") or "")[:MAX_RESOLUTION_CHARS]
        description = str(hit.get("description") or "")[:600]

        blocks.append(
            f"--- INCIDENT {i} "
            f"(similarity {hit['similarity']:.3f}, "
            f"priority {hit.get('priority')}) ---\n"
            f"REPORTED:\n{description}\n\n"
            f"WHAT ENGINEERS DID:\n{resolution}"
        )

    return "\n\n".join(blocks)


def build_prompt(result):

    priority = result["priority"]

    priority_line = (
        f"{priority['label']} ({priority['confidence']:.0%} confidence)"
        if priority["label"] else "unavailable"
    )

    # The category model is out of domain for helpdesk text, so its
    # output is withheld rather than anchoring the model on a
    # meaningless Apache project name.

    return f"""NEW TICKET:
{result['description']}

PREDICTED PRIORITY: {priority_line}

{len(result['similar'])} similar historical incidents were retrieved:

{build_evidence(result)}

Produce the JSON described in your instructions."""


def parse_response(text):
    """
    response_mime_type should give clean JSON, but models occasionally
    wrap it. Extract the outermost object rather than failing.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("no JSON object in response")
        return json.loads(match.group(0))


def call_gemini(prompt, api_key):

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1200,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    return response.text


class Advisor:

    def __init__(self, backend="embedding", retriever=None):
        self.recommender = Recommender(
            backend=backend, retriever=retriever
        )

    def advise(self, description, k=5, use_llm=True):

        result = self.recommender.recommend(description, k=k)

        if result.get("error") or not use_llm:
            return result

        # Structural grounding guard: no evidence, no generation.
        strong = [h for h in result["similar"] if not h.get("weak")]

        if not strong:
            result["warnings"].append(
                "LLM generation skipped: no match strong enough to "
                "ground a recommendation."
            )
            return result

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            result["warnings"].append(
                "LLM unavailable (GEMINI_API_KEY not set). "
                "Showing retrieval-only recommendation."
            )
            return result

        try:
            parsed = parse_response(call_gemini(build_prompt(result), api_key))

            result["steps"] = [
                {"text": step, "from_ticket": None, "similarity": None}
                for step in parsed.get("steps", [])
            ]
            result["explanation"] = parsed.get("explanation", "")
            result["source"] = "llm"
            result["evidence_quality"] = parsed.get(
                "evidence_quality", "unknown"
            )

            for caveat in parsed.get("caveats", []):
                result["warnings"].append(caveat)

            result["warnings"].append(
                "AI-generated from historical incidents. Not a "
                "verified fix - validate before applying."
            )

        except Exception as exc:
            result["warnings"].append(
                f"LLM call failed ({type(exc).__name__}: {exc}). "
                "Falling back to retrieval-only recommendation."
            )

        return result


def main():

    use_llm = "--no-llm" not in sys.argv
    query = " ".join(a for a in sys.argv[1:] if a != "--no-llm")

    if not query:
        print("usage: python3 advisor.py [--no-llm] <description>")
        return

    result = Advisor().advise(query, use_llm=use_llm)

    print("\n" + "=" * 68)
    print(f" RESOLUTION RECOMMENDATION  [source: {result.get('source')}]")
    print("=" * 68)

    if result.get("error"):
        print(f"\n  {result['error']}\n")
        return

    priority = result["priority"]
    category = result["category"]

    if priority["label"]:
        print(f"\n  Priority: {priority['label']} "
              f"({priority['confidence']:.0%})")
    else:
        print("\n  Priority: unavailable")

    if category["uncertain"]:
        print("  Category: uncertain "
              "(model is out of domain for helpdesk text)")
    else:
        print(f"  Category: {category['label']} "
              f"({category['confidence']:.0%})")

    print(f"\n  Similar incidents: {len(result['similar'])}")

    if result.get("evidence_quality"):
        print(f"  Evidence quality: {result['evidence_quality']}")

    if result["steps"]:
        print("\n" + "-" * 68)
        print(" RECOMMENDED STEPS")
        print("-" * 68 + "\n")
        for i, step in enumerate(result["steps"], 1):
            print(f"  {i}. {step['text']}")
        print()

    if result.get("explanation"):
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