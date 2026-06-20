"""
Problem Transfer Engine — Step 2: Analogy Finder

Takes a structured problem and finds analogous domains with known solutions.

Run:  python3 analogies.py
Or:   python3 analogies.py "your problem here"
"""

import json
import sys
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

STRUCTURE_PROMPT = """You are a structural problem analyst. Extract the underlying structure of a problem.

Return a JSON object with exactly these fields:
{
  "mechanisms": [...],
  "constraints": [...],
  "desired_outcomes": [...],
  "anti_goals": [...]
}

Rules:
- Be precise and abstract. "Queue management" is better than "waiting room problems".
- Constraints are the most important field — be specific about what legally, physically, or ethically cannot change.
- List 3-6 items per field.
- Return ONLY valid JSON. No explanation, no markdown."""

ANALOGY_PROMPT = """You are a cross-domain reasoning engine. You find structural analogies between problem domains.

Given a structured problem, find 4 domains that have successfully solved a structurally similar problem.

IMPORTANT RULES:
- Do NOT pick domains that are obvious or already well-known to practitioners in the user's field.
- Do NOT pick the same industry (if input is healthcare, don't suggest "other hospitals").
- Prioritize domains where the solution has documented evidence of success.
- Be honest about where the analogy breaks down — this is more valuable than overselling it.
- Reject superficial analogies. "Both involve people waiting" is NOT structural similarity.

Return a JSON array of exactly 4 objects, each with:
{
  "domain": "name of the domain",
  "why_similar": "specific structural mechanisms they share with the input problem",
  "solution_method": "the concrete technique or approach used in this domain",
  "concrete_example": "a real, specific implementation of this solution",
  "transferability_score": 0-100,
  "where_it_holds": "what makes this analogy genuinely valid",
  "where_it_breaks": "the most important reason this analogy could fail",
  "action_suggestion": "one concrete thing to try, inspired by this domain"
}

Return ONLY valid JSON array. No explanation, no markdown."""


def structure_problem(problem: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": STRUCTURE_PROMPT},
            {"role": "user", "content": problem},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def find_analogies(structure: dict) -> list:
    structure_text = json.dumps(structure, indent=2)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": ANALOGY_PROMPT},
            {"role": "user", "content": f"Find analogies for this problem structure:\n\n{structure_text}"},
        ],
        temperature=0.4,
    )
    content = response.choices[0].message.content.strip()
    # Strip markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def print_structure(structure: dict):
    print("\nPROBLEM STRUCTURE:")
    print("  Mechanisms:       ", ", ".join(structure.get("mechanisms", [])))
    print("  Constraints:      ", ", ".join(structure.get("constraints", [])))
    print("  Desired outcomes: ", ", ".join(structure.get("desired_outcomes", [])))
    print("  Anti-goals:       ", ", ".join(structure.get("anti_goals", [])))


def score_bar(score: int) -> str:
    filled = round(score / 10)
    return "█" * filled + "░" * (10 - filled) + f"  {score}/100"


def print_analogies(analogies: list):
    print(f"\n{'='*60}")
    print("ANALOGIES FROM OTHER DOMAINS")
    print(f"{'='*60}")
    for i, a in enumerate(analogies, 1):
        print(f"\n[{i}] {a['domain'].upper()}")
        print(f"    Transferability: {score_bar(a['transferability_score'])}")
        print(f"\n    Why structurally similar:")
        print(f"    {a['why_similar']}")
        print(f"\n    Solution method:")
        print(f"    {a['solution_method']}")
        print(f"\n    Real example:")
        print(f"    {a['concrete_example']}")
        print(f"\n    Where the analogy holds:")
        print(f"    {a['where_it_holds']}")
        print(f"\n    Where it breaks down:")
        print(f"    {a['where_it_breaks']}")
        print(f"\n    What to try:")
        print(f"    {a['action_suggestion']}")
        print(f"\n    {'-'*56}")


def run(problem: str):
    print(f"\nPROBLEM:\n{problem}")
    print("\n[1/2] Extracting problem structure...", end=" ", flush=True)
    structure = structure_problem(problem)
    print("done")
    print_structure(structure)

    print("\n[2/2] Finding analogies from other domains...", end=" ", flush=True)
    analogies = find_analogies(structure)
    print("done")
    print_analogies(analogies)

    result = {"problem": problem, "structure": structure, "analogies": analogies}
    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nFull output saved to output.json")
    return result


TEST_PROBLEM = "Our hospital struggles with long waiting times in the emergency department. Patients wait 4-6 hours on average. We have enough doctors but the flow between departments is chaotic."


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Set GROQ_API_KEY in a .env file.")
        sys.exit(1)

    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
    else:
        problem = TEST_PROBLEM

    run(problem)


if __name__ == "__main__":
    main()
