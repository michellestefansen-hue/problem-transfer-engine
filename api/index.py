"""
Routing only. No business logic here.
All LLM logic lives in llm.py.
All external data sources live in sources/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from llm import structure_problem, find_analogies, synthesise, deep_dive
from sources.wikipedia import get_summary
from sources.openalex import find_papers

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.json
    problem = body.get("problem", "").strip()
    lang = body.get("lang", "en")
    if not problem:
        return jsonify({"error": "No problem provided"}), 400
    try:
        structure = structure_problem(problem, lang)
        analogies = find_analogies(structure, lang)
        synthesis = synthesise(problem, analogies, lang)

        # Enrich each analogy with papers (non-blocking — empty list on failure)
        mechanisms = structure.get("mechanisms", [])
        for a in analogies:
            a["papers"] = find_papers(mechanisms, a["domain"])

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
        result = deep_dive(problem, analogy, lang)
        # Add Wikipedia context for the analogy domain
        result["wikipedia"] = get_summary(analogy.get("domain", ""))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
