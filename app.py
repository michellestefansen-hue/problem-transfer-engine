import json
import os
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
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

Given a structured problem, find 4 analogies from 4 DIFFERENT explanation levels:
1. A TECHNICAL/ENGINEERING domain (systems, algorithms, infrastructure)
2. A BEHAVIOURAL/SOCIAL domain (psychology, economics, organisational behaviour)
3. A BIOLOGICAL/NATURAL domain (evolution, ecology, immune systems, ant colonies)
4. An OPERATIONAL domain (logistics, manufacturing, military, aviation)

RULES:
- Do NOT pick domains obvious to practitioners in the user's field.
- Do NOT pick the same industry as the input.
- Be honest about where the analogy breaks down — this is more valuable than overselling it.
- Reject superficial analogies. "Both involve people waiting" is NOT structural similarity.
- action_next_step must be a single concrete action completable within ONE WEEK, not a project description.
  Bad: "Implement a real-time scheduling system..."
  Good: "Run a 2-hour mapping session with staff to draw the actual patient journey as a flowchart, identifying the 3 handoff points with the longest delays."

Return a JSON array of exactly 4 objects:
{
  "domain": "name of the domain",
  "explanation_level": "technical|behavioural|biological|operational",
  "why_similar": "the specific structural mechanism they share — not surface similarity",
  "solution_method": "the concrete technique or approach used in this domain",
  "concrete_example": "a real, named implementation of this solution",
  "transferability_score": 0-100,
  "where_it_holds": "the core reason this analogy is structurally valid",
  "where_it_breaks": "the single most important reason this analogy could fail",
  "action_next_step": "one concrete action completable within one week, written as an instruction"
}

Return ONLY valid JSON array. No explanation, no markdown."""

SYNTHESIS_PROMPT = """You are a strategic insight synthesiser.

Given a problem and 4 cross-domain analogies, identify the single most important meta-insight:
What do all (or most) of these analogies reveal about the UNDERLYING NATURE of this problem that the user probably hasn't articulated yet?

This should be a reframing — not a summary of the analogies, but what they collectively point to.

Return a JSON object with exactly these fields:
{
  "meta_insight": "one punchy sentence (max 30 words) that reframes the problem based on what the analogies reveal",
  "implication": "one sentence on what this reframing implies for how to approach the problem differently"
}

Return ONLY valid JSON. No explanation, no markdown."""

LANG_INSTRUCTION = {
    "en": "Respond in English.",
    "no": "CRITICAL: Keep all JSON keys exactly as specified in English. Only translate the VALUES into Norwegian bokmål. Do not translate any JSON key names.",
}


def structure_problem(problem: str, lang: str = "en") -> dict:
    system = STRUCTURE_PROMPT + "\n\n" + LANG_INSTRUCTION.get(lang, "")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def find_analogies(structure: dict, lang: str = "en") -> list:
    system = ANALOGY_PROMPT + "\n\n" + LANG_INSTRUCTION.get(lang, "")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Find analogies for this problem structure:\n\n{json.dumps(structure, indent=2)}"},
        ],
        temperature=0.4,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def synthesise(problem: str, analogies: list, lang: str = "en") -> dict:
    system = SYNTHESIS_PROMPT + "\n\n" + LANG_INSTRUCTION.get(lang, "")
    summary = [{"domain": a["domain"], "why_similar": a["why_similar"]} for a in analogies]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Problem: {problem}\n\nAnalogies:\n{json.dumps(summary, indent=2)}"},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


DEEP_DIVE_PROMPT = """You are a strategic implementation advisor helping someone apply a cross-domain analogy to their real problem.

Given a problem and one specific analogy, produce a deep-dive analysis with exactly these 7 sections.

Return a JSON object with these exact fields:

{
  "explained_simply": "2-3 sentences explaining how the solution works in the source domain, written for someone who has never heard of it. No jargon.",

  "transfer_steps": [
    {
      "step": "short title",
      "description": "what to do and why",
      "effort": "low|medium|high"
    }
  ],

  "critical_experiment": {
    "hypothesis": "the one assumption that must be true for this analogy to work in your context",
    "experiment": "the fastest, cheapest way to test that assumption — specific enough to schedule",
    "timeframe": "e.g. 1 day, 3 days, 1 week",
    "success_signal": "exactly what you would observe if the hypothesis is correct"
  },

  "resistance": [
    {
      "stakeholder": "who will resist",
      "instinct": "what they will say",
      "real_concern": "what they actually mean underneath",
      "response": "how to address the real concern, not the stated one"
    }
  ],

  "open_questions": [
    "question you must answer before committing to this direction"
  ],

  "warning_signs": [
    "specific, observable sign that this analogy will NOT work in your context"
  ],

  "implementation_guide": [
    {
      "phase": "e.g. Week 1, Month 1, Month 2-3",
      "title": "short phase name",
      "actions": ["concrete action 1", "concrete action 2"],
      "involved": "who needs to be involved",
      "success_criteria": "what does success look like before moving to the next phase"
    }
  ]
}

RULES:
- transfer_steps: 3-5 steps, ordered from least to most effort
- resistance: 2-3 stakeholders
- open_questions: exactly 5 questions
- warning_signs: exactly 3 signs, each specific and observable (not vague)
- implementation_guide: 4-5 phases from "Week 1" to "Month 3-6"
- Everything must be specific to BOTH the analogy domain AND the user's actual problem. No generic advice.
- Return ONLY valid JSON. No explanation, no markdown."""


def deep_dive(problem: str, analogy: dict, lang: str = "en") -> dict:
    system = DEEP_DIVE_PROMPT + "\n\n" + LANG_INSTRUCTION.get(lang, "")
    payload = {
        "problem": problem,
        "analogy_domain": analogy.get("domain"),
        "solution_method": analogy.get("solution_method"),
        "why_similar": analogy.get("why_similar"),
        "where_it_breaks": analogy.get("where_it_breaks"),
    }
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


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
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
