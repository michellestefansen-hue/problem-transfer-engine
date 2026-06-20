"""
Routing only. No business logic here.
All LLM logic lives in llm.py.
All external data sources live in sources/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from llm import structure_problem, find_analogies, synthesise, deep_dive, validate_analogy_with_wikipedia, generate_questions
from sources.wikipedia import get_summary
from sources.openalex import find_papers

app = Flask(__name__, template_folder="templates")


def _evidence_level(papers: list) -> str:
    """Only report evidence level when we actually find papers.
    Absence of papers means the search didn't find anything — not that evidence doesn't exist."""
    n = len(papers)
    if n == 0:
        return None
    if n == 1:
        return "limited"
    return "documented"


def _adjust_score(score: int, evidence_level: str) -> int:
    """Only boost score when we find papers — never penalise absence."""
    adjustment = {"limited": 3, "documented": 6}
    return max(0, min(100, score + adjustment.get(evidence_level, 0)))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/questions", methods=["POST"])
def questions():
    body = request.json
    problem = body.get("problem", "").strip()
    lang = body.get("lang", "en")
    if not problem:
        return jsonify({"error": "No problem provided"}), 400
    try:
        result = generate_questions(problem, lang)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.json
    problem = body.get("problem", "").strip()
    lang = body.get("lang", "en")
    answers = body.get("answers", {})
    if not problem:
        return jsonify({"error": "No problem provided"}), 400
    try:
        structure = structure_problem(problem, lang, answers)
        analogies = find_analogies(structure, lang, answers)

        mechanisms = structure.get("mechanisms", [])

        # First pass: enrich all analogies with OpenAlex papers + score adjustment
        for a in analogies:
            papers = find_papers(mechanisms, a["domain"])
            evidence = _evidence_level(papers)
            a["papers"] = papers
            a["evidence_level"] = evidence
            a["transferability_score"] = _adjust_score(
                a.get("transferability_score", 50), evidence
            )

        # Sort by score before Wikipedia validation
        analogies.sort(key=lambda x: x["transferability_score"], reverse=True)

        # Second pass: Wikipedia fact-check only the top 2 — halves latency
        for i, a in enumerate(analogies):
            wiki = get_summary(a.get("domain", "")) if i < 2 else {}
            analogies[i] = validate_analogy_with_wikipedia(a, wiki, lang)

        synthesis = synthesise(problem, analogies, lang, answers)
        return jsonify({"structure": structure, "analogies": analogies, "synthesis": synthesis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/deep-dive", methods=["POST"])
def deep_dive_route():
    body = request.json
    problem = body.get("problem", "").strip()
    analogy = body.get("analogy", {})
    lang = body.get("lang", "en")
    if not problem or not analogy:
        return jsonify({"error": "Missing problem or analogy"}), 400
    try:
        # Fetch Wikipedia BEFORE calling LLM — used to ground the deep dive
        wiki = get_summary(analogy.get("domain", ""))
        result = deep_dive(problem, analogy, lang, wikipedia=wiki)
        result["wikipedia"] = wiki
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
