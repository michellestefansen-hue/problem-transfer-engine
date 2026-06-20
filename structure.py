"""
Problem Transfer Engine — Step 1: Problem Structuring

Takes a natural language problem description and extracts:
- mechanisms (the underlying processes at play)
- constraints (what cannot be changed)
- desired outcomes (what success looks like)
- anti_goals (what must not happen)

Run: python3 structure.py
Or:  python3 structure.py "your problem here"
"""

import json
import sys
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a structural problem analyst. Your job is to extract the underlying structure of a problem — not its surface description.

Given a problem description, return a JSON object with exactly these fields:

{
  "mechanisms": [...],     // The core processes, dynamics, or patterns at play (e.g. "queue buildup", "resource contention")
  "constraints": [...],    // What CANNOT be changed — rules, regulations, ethics, physics, budget (e.g. "patient rights protected by law")
  "desired_outcomes": [...], // What success looks like in measurable or observable terms
  "anti_goals": [...]      // What must NOT happen, even if it would improve the primary metric
}

Rules:
- Be precise and abstract. "Queue management" is better than "waiting room problems".
- Constraints are the most important field. Be specific. Vague constraints like "budget" are useless.
- List 3-6 items per field.
- Return ONLY valid JSON. No explanation, no markdown.
"""

TEST_PROBLEMS = [
    "Our hospital struggles with long waiting times in the emergency department. Patients wait 4-6 hours on average. We have enough doctors but the flow between departments is chaotic.",
    "Our software team misses deadlines constantly. We have skilled developers but estimates are always wrong and scope keeps growing mid-sprint.",
    "Our city's traffic congestion is getting worse every year. Building more roads hasn't helped. People don't use public transport even though it exists.",
    "Our non-profit struggles to retain volunteers. People sign up enthusiastically but drop out after a few months. We can't pay them.",
    "Our e-commerce checkout has a 70% abandonment rate. Users add items to cart but leave before paying.",
]


def structure_problem(problem: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def print_structure(problem: str, structure: dict):
    print(f"\n{'='*60}")
    print(f"PROBLEM:\n{problem}")
    print(f"\nMECHANISMS:")
    for item in structure.get("mechanisms", []):
        print(f"  • {item}")
    print(f"\nCONSTRAINTS (what cannot change):")
    for item in structure.get("constraints", []):
        print(f"  • {item}")
    print(f"\nDESIRED OUTCOMES:")
    for item in structure.get("desired_outcomes", []):
        print(f"  • {item}")
    print(f"\nANTI-GOALS (must not happen):")
    for item in structure.get("anti_goals", []):
        print(f"  • {item}")
    print(f"{'='*60}")


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in a .env file or environment variable.")
        sys.exit(1)

    if len(sys.argv) > 1:
        # Single problem from command line
        problem = " ".join(sys.argv[1:])
        structure = structure_problem(problem)
        print_structure(problem, structure)
        print("\nRAW JSON:")
        print(json.dumps(structure, indent=2))
    else:
        # Run all test problems
        print("Running test suite — 5 problems\n")
        results = []
        for i, problem in enumerate(TEST_PROBLEMS, 1):
            print(f"[{i}/5] Processing...", end=" ", flush=True)
            structure = structure_problem(problem)
            results.append({"problem": problem, "structure": structure})
            print_structure(problem, structure)

        # Save results for review
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to results.json")
        print("\nNow review the output manually:")
        print("  - Are the mechanisms abstract enough to match across domains?")
        print("  - Are the constraints specific enough to filter bad analogies?")
        print("  - Would two problems with similar mechanisms look structurally similar?")


if __name__ == "__main__":
    main()
