import json
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LANG_INSTRUCTION = {
    "en": "Respond in English.",
    "no": "CRITICAL: Keep all JSON keys exactly as specified in English. Only translate the VALUES into Norwegian bokmål. Do not translate any JSON key names.",
}

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

Given a structured problem, find EXACTLY 4 analogies — one from each of these four roles:

ROLE 1 — THE ESTABLISHED: The most well-documented cross-domain transfer relevant to this problem. This may be a "classic" analogy (lean manufacturing, TCP, swarm intelligence etc.) — include it precisely BECAUSE it is well-documented and evidence-based. Explain why it is the strongest known transfer.

ROLE 2 — THE UNEXPECTED: A domain that practitioners in the user's field would not immediately think of. It must be structurally valid — not merely surprising. If it cannot be both unexpected AND structurally rigorous, choose rigour over novelty.

ROLE 3 — THE NON-HUMAN: A biological, physical, chemical, or ecological system. No human organisations. The mechanism must be non-obvious and specifically relevant to the problem structure.

ROLE 4 — THE EXPERT CHALLENGER: An analogy that would surprise or challenge an expert in the user's own field — something that reframes a core assumption they hold about their problem.

RULES:
- Do NOT pick the same industry as the input.
- Be honest about where the analogy breaks down — this is more valuable than overselling it.
- Reject superficial analogies. "Both involve people waiting" is NOT structural similarity.
- action_next_step must be a single concrete action completable within ONE WEEK by one person with no budget.
  ✗ BAD: "Implement a scheduling system inspired by air traffic control"
  ✓ GOOD: "Spend 90 minutes mapping every handoff between departments on a whiteboard, marking which ones have no agreed owner"
- The four analogies must be genuinely different from each other in their explanatory logic.

For each analogy, assess its EPISTEMIC STATUS:
- "empirical": cross-domain transfer studied and measured in peer-reviewed literature
- "implementation": named organisation implemented this with published results, not formally studied as a transfer
- "structural": strong structural similarity, well-reasoned, but no documented transfer exists
- "speculative": plausible but primarily inferential

Be honest. Most analogies are "structural" or "speculative". Only use "empirical" if confident.

Return a JSON array of exactly 4 objects:
{
  "domain": "name of the domain",
  "explanation_level": "established|unexpected|non-human|expert-challenger",
  "why_similar": "the specific structural mechanism they share — not surface similarity",
  "solution_method": "the concrete technique or approach used in this domain",
  "concrete_example": "a real, named implementation of this solution",
  "transferability_score": 0-100,
  "where_it_holds": "the core reason this analogy is structurally valid",
  "where_it_breaks": "the single most important reason this analogy could fail",
  "action_next_step": "one concrete action completable within one week, written as an instruction",
  "epistemic_status": {
    "basis": "empirical|implementation|structural|speculative",
    "confidence_note": "one honest sentence about what this claim is actually based on and what you are uncertain about"
  }
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

DEEP_DIVE_PROMPT = """You are a strategic implementation advisor helping someone apply a cross-domain analogy to their real problem.

Given a problem and one specific analogy, produce a deep-dive analysis with exactly these 7 sections.

Return a JSON object with these exact fields:
{
  "explained_simply": "2-3 sentences explaining how the solution works in the source domain, written for someone who has never heard of it. No jargon.",
  "transfer_steps": [
    { "step": "short title", "description": "what to do and why", "effort": "low|medium|high" }
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
  "open_questions": ["question you must answer before committing to this direction"],
  "warning_signs": ["specific, observable sign that this analogy will NOT work in your context"],
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


def _call(system: str, user: str, temperature: float = 0.3, json_mode: bool = False):
    kwargs = dict(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def _system(prompt: str, lang: str) -> str:
    return prompt + "\n\n" + LANG_INSTRUCTION.get(lang, "")


VALIDATE_PROMPT = """You proposed an analogy between two domains. You are now given factual information from Wikipedia about the concrete example you cited.

Your job: critically re-examine your analogy in light of this factual context.

Ask yourself:
- Does the Wikipedia description confirm or contradict the mechanism you claimed?
- Is your concrete example accurate, or should it be corrected or replaced?
- Does the structural similarity still hold, or should the analogy be weakened or strengthened?
- Does your transferability score need adjustment based on the facts?

Return a JSON object with these fields:
{
  "domain": "same as before",
  "explanation_level": "same as before",
  "why_similar": "updated if needed — must now be grounded in the Wikipedia facts",
  "solution_method": "updated if the facts revealed something more accurate",
  "concrete_example": "corrected if your original example was inaccurate",
  "transferability_score": updated integer if the facts change your confidence,
  "where_it_holds": "updated based on facts",
  "where_it_breaks": "updated — did the Wikipedia facts reveal a new failure mode?",
  "action_next_step": "same or updated",
  "epistemic_status": {
    "basis": "empirical|implementation|structural|speculative",
    "confidence_note": "updated — now explicitly reference what the Wikipedia source confirmed or contradicted"
  },
  "wikipedia_changed_answer": true or false,
  "what_changed": "one sentence on what the Wikipedia facts caused you to revise, or 'Nothing changed — the facts confirmed the original analogy'"
}

Return ONLY valid JSON. No explanation, no markdown."""


def validate_analogy_with_wikipedia(analogy: dict, wikipedia: dict, lang: str = "en") -> dict:
    """Send analogy + Wikipedia facts back to LLM. Ask it to confirm or revise."""
    if not wikipedia or not wikipedia.get("extract"):
        return analogy

    system = _system(VALIDATE_PROMPT, lang)
    user = (
        f"YOUR ORIGINAL ANALOGY:\n{json.dumps(analogy, indent=2)}\n\n"
        f"WIKIPEDIA FACTS about '{wikipedia.get('title', analogy.get('domain'))}':\n"
        f"{wikipedia['extract']}\n\n"
        f"Re-examine your analogy in light of these facts."
    )
    try:
        revised = _call(system, user, temperature=0.2, json_mode=True)
        # Preserve papers and evidence_level from original
        revised["papers"] = analogy.get("papers", [])
        revised["evidence_level"] = analogy.get("evidence_level")
        return revised
    except Exception:
        return analogy


QUESTIONS_PROMPT = """You are preparing to find cross-domain analogies for a problem. Before doing so, you need 3 pieces of information that will dramatically improve the quality and relevance of the analogies.

Generate exactly 3 short, sharp clarifying questions tailored to THIS specific problem.

Every question set must include one version of each of these three types:
1. ALREADY TRIED: What approaches have already been attempted? (prevents suggesting the obvious)
2. SURPRISING OBSERVATION: What is the most counterintuitive or unexpected thing observed about this problem? (reveals hidden mechanisms the user may not have articulated)
3. ROOT CAUSE UNCERTAINTY: What do they believe the root cause is — and what are they most uncertain about? (separates symptoms from actual problem)

Tailor each question specifically to the problem described. Do not use generic phrasing.

Return a JSON object with exactly this structure:
{
  "questions": [
    { "id": "tried", "question": "...", "placeholder": "e.g. ..." },
    { "id": "surprising", "question": "...", "placeholder": "e.g. ..." },
    { "id": "uncertainty", "question": "...", "placeholder": "e.g. ..." }
  ]
}

Return ONLY valid JSON. No explanation, no markdown."""


def generate_questions(problem: str, lang: str = "en") -> dict:
    return _call(_system(QUESTIONS_PROMPT, lang), problem, temperature=0.3, json_mode=True)


def _answers_block(answers: dict) -> str:
    if not answers:
        return ""
    lines = "\n".join(f"- {k}: {v}" for k, v in answers.items() if v and v.strip())
    return f"\n\nUSER CONTEXT (from clarifying questions):\n{lines}"


def structure_problem(problem: str, lang: str = "en", answers: dict = None) -> dict:
    user = problem + _answers_block(answers)
    return _call(_system(STRUCTURE_PROMPT, lang), user, temperature=0.2, json_mode=True)


def find_analogies(structure: dict, lang: str = "en", answers: dict = None) -> list:
    answers_block = _answers_block(answers)

    # Build active instructions from what the user told us
    active_instructions = ""
    if answers:
        tried = next((v for k, v in answers.items() if "tried" in k.lower() or "already" in k.lower()), None)
        surprising = next((v for k, v in answers.items() if "surpris" in k.lower() or "unexpect" in k.lower() or "counterintuitive" in k.lower()), None)
        uncertainty = next((v for k, v in answers.items() if "uncertain" in k.lower() or "root cause" in k.lower() or "believe" in k.lower()), None)

        parts = []
        if tried:
            parts.append(f"- DO NOT suggest approaches similar to what they have already tried: {tried}")
        if surprising:
            parts.append(f"- The user's surprising observation reveals a hidden mechanism — let it inform especially the UNEXPECTED and EXPERT-CHALLENGER analogies: {surprising}")
        if uncertainty:
            parts.append(f"- Their root cause uncertainty should guide which structural mechanisms you prioritise: {uncertainty}")
        if parts:
            active_instructions = "\n\nINSTRUCTIONS FROM USER ANSWERS:\n" + "\n".join(parts)

    user = (
        f"Find analogies for this problem structure:\n\n{json.dumps(structure, indent=2)}"
        + answers_block
        + active_instructions
    )
    return _call(_system(ANALOGY_PROMPT, lang), user, temperature=0.4)


def synthesise(problem: str, analogies: list, lang: str = "en", answers: dict = None) -> dict:
    summary = [{"domain": a["domain"], "why_similar": a["why_similar"]} for a in analogies]
    user = f"Problem: {problem}\n\nAnalogies:\n{json.dumps(summary, indent=2)}" + _answers_block(answers)
    return _call(_system(SYNTHESIS_PROMPT, lang), user, temperature=0.3, json_mode=True)


def deep_dive(problem: str, analogy: dict, lang: str = "en", wikipedia: dict = None) -> dict:
    payload = {
        "problem": problem,
        "analogy_domain": analogy.get("domain"),
        "solution_method": analogy.get("solution_method"),
        "why_similar": analogy.get("why_similar"),
        "where_it_breaks": analogy.get("where_it_breaks"),
    }

    # Ground the LLM with factual Wikipedia context about the source domain
    system = _system(DEEP_DIVE_PROMPT, lang)
    if wikipedia and wikipedia.get("extract"):
        system += (
            f"\n\nFACTUAL CONTEXT about {analogy.get('domain')} (from Wikipedia):\n"
            f"{wikipedia['extract']}\n\n"
            f"Use this factual context to make your explanations and transfer steps more accurate and specific. "
            f"Do not contradict facts stated above."
        )

    return _call(system, json.dumps(payload, indent=2), temperature=0.3, json_mode=True)
