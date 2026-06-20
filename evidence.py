"""
Problem Transfer Engine — Step 3: Evidence Layer

Strategy: ask the LLM to name specific papers it knows exist about cross-domain
transfer, then verify each one in OpenAlex by title search. This is more reliable
than open-ended keyword search because the LLM has been trained on these papers
and knows which cross-domain studies actually exist.
"""

import json
import os
import sys
import urllib.request
import urllib.parse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── LLM: name specific papers ─────────────────────────────────────────────────

PAPER_PROMPT = """You are an academic researcher who knows the cross-domain transfer literature well.

Given an analogy between two domains, name 2-3 REAL, SPECIFIC papers that document
the transfer of solutions from the analogy domain to the target domain (or vice versa).

Only name papers you are highly confident actually exist. It is better to name 1 real
paper than 3 invented ones. If you are not confident any specific papers exist, say so.

Return a JSON array of objects with these fields:
{
  "title": "exact or near-exact paper title",
  "authors": "first author last name et al.",
  "year": 2020,
  "why_relevant": "one sentence explaining what this paper shows about the cross-domain transfer"
}

Return ONLY valid JSON array. No explanation, no markdown."""


def get_paper_suggestions(analogy: dict, problem_domain: str) -> list:
    prompt = (
        f"Analogy source domain: {analogy['domain']}\n"
        f"Solution method: {analogy['solution_method']}\n"
        f"Target domain: {problem_domain}\n"
        f"Why structurally similar: {analogy['why_similar']}"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": PAPER_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except Exception:
        return []


# ── OpenAlex: verify by title ─────────────────────────────────────────────────

def verify_paper_in_openalex(title: str):
    """Search OpenAlex for a paper by title. Return metadata if found."""
    encoded = urllib.parse.quote(f'"{title}"')
    url = (
        f"https://api.openalex.org/works"
        f"?search={encoded}"
        f"&per-page=1"
        f"&select=title,doi,publication_year,cited_by_count,primary_location"
        f"&mailto=problem-transfer-engine@example.com"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read())
        results = data.get("results", [])
        if not results:
            return None
        work = results[0]
        found_title = work.get("title", "")
        # Simple check: at least 60% of the words in the query appear in the result
        query_words = set(title.lower().split())
        result_words = set(found_title.lower().split())
        overlap = len(query_words & result_words) / max(len(query_words), 1)
        if overlap < 0.5:
            return None
        source = work.get("primary_location") or {}
        source_name = (source.get("source") or {}).get("display_name", "")
        return {
            "title": found_title,
            "doi": work.get("doi", ""),
            "year": work.get("publication_year"),
            "citations": work.get("cited_by_count", 0),
            "source": source_name,
            "verified": True,
        }
    except Exception:
        return None


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def get_wikipedia_summary(topic: str) -> dict:
    encoded = urllib.parse.quote(topic)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ProblemTransferEngine/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
        return {
            "title": data.get("title", ""),
            "extract": data.get("extract", "")[:400],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Evidence gathering ────────────────────────────────────────────────────────

def gather_evidence(structure: dict, analogies: list, problem: str) -> list:
    desired = structure.get("desired_outcomes", [])
    problem_domain = desired[0] if desired else problem[:60]

    enriched = []
    for a in analogies:
        domain = a["domain"]
        print(f"  [{domain}]")

        print(f"    Asking LLM for specific papers...", end=" ", flush=True)
        suggestions = get_paper_suggestions(a, problem_domain)
        print(f"got {len(suggestions)} suggestions")

        verified_papers = []
        unverified_papers = []
        for s in suggestions:
            print(f"    Verifying: '{s['title'][:60]}...'", end=" ", flush=True)
            result = verify_paper_in_openalex(s["title"])
            if result:
                result["why_relevant"] = s.get("why_relevant", "")
                verified_papers.append(result)
                print(f"✓ found ({result['citations']} citations)")
            else:
                unverified_papers.append(s)
                print(f"✗ not found in OpenAlex")

        print(f"    Fetching Wikipedia context...", end=" ", flush=True)
        wiki = get_wikipedia_summary(domain)
        print(f"done")

        enriched.append({
            **a,
            "evidence": {
                "verified_papers": verified_papers,
                "unverified_papers": unverified_papers,
                "domain_context": wiki,
            }
        })

    return enriched


# ── Printing ──────────────────────────────────────────────────────────────────

def print_report(problem: str, structure: dict, analogies: list):
    mechanisms = structure.get("mechanisms", [])

    print(f"\n{'='*60}")
    print("PROBLEM TRANSFER REPORT")
    print(f"{'='*60}")
    print(f"\nProblem: {problem}")
    print(f"Core mechanisms: {', '.join(mechanisms[:3])}")

    for i, a in enumerate(analogies, 1):
        ev = a.get("evidence", {})
        verified = ev.get("verified_papers", [])
        unverified = ev.get("unverified_papers", [])
        wiki = ev.get("domain_context", {})
        score = a["transferability_score"]
        bar = "█" * round(score / 10) + "░" * (10 - round(score / 10))

        print(f"\n{'─'*60}")
        print(f"[{i}] {a['domain'].upper()}   {bar} {score}/100")
        print(f"\n  Why structurally similar:")
        print(f"  {a['why_similar']}")
        print(f"\n  Solution: {a['solution_method']}")
        print(f"  Example:  {a['concrete_example']}")
        print(f"\n  Breaks down when: {a['where_it_breaks']}")
        print(f"  Try this:         {a['action_suggestion']}")

        if wiki and "extract" in wiki and not wiki.get("error"):
            print(f"\n  Domain context:")
            print(f"  {wiki['extract'][:250]}...")
            if wiki.get("url"):
                print(f"  {wiki['url']}")

        if verified:
            print(f"\n  Verified research ({len(verified)} papers confirmed in OpenAlex):")
            for p in verified:
                print(f"  ✓ [{p['year']}] {p['title'][:75]}")
                print(f"    {p['citations']} citations | {p['source']}")
                if p.get("why_relevant"):
                    print(f"    → {p['why_relevant']}")
                if p.get("doi"):
                    print(f"    {p['doi']}")

        if unverified:
            print(f"\n  Suggested research (not verified in OpenAlex — check manually):")
            for p in unverified:
                print(f"  ? [{p.get('year', '?')}] {p['title'][:75]}")
                if p.get("why_relevant"):
                    print(f"    → {p['why_relevant']}")

        if not verified and not unverified:
            print(f"\n  No specific papers identified for this analogy.")

    print(f"\n{'='*60}")


def main():
    output_file = "output.json"
    if not os.path.exists(output_file):
        print("ERROR: Run analogies.py first to generate output.json")
        sys.exit(1)

    with open(output_file) as f:
        data = json.load(f)

    problem = data["problem"]
    structure = data["structure"]
    analogies = data["analogies"]

    print(f"Gathering evidence for {len(analogies)} analogies...\n")
    enriched = gather_evidence(structure, analogies, problem)

    print_report(problem, structure, enriched)

    data["analogies"] = enriched
    with open("output_with_evidence.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\nFull report saved to output_with_evidence.json")


if __name__ == "__main__":
    main()
