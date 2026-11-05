"""
Flask REST API for the Interview Trainer Agent.
Exposes endpoints for question generation, answer evaluation, and preparation planning.
"""

import os
import sys
from pathlib import Path

# Ensure src package is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from src import build_agent, ResumeParser, UserProfile

app = Flask(__name__, static_folder="static")
CORS(app)

# ---------------------------------------------------------------------------
# Lazy-initialize agent components on first request
# ---------------------------------------------------------------------------
_agent_cache = {}

def get_agent():
    if not _agent_cache:
        generator, evaluator, planner, rag, parser, status = build_agent()
        _agent_cache.update({
            "generator": generator,
            "evaluator": evaluator,
            "planner": planner,
            "rag": rag,
            "parser": parser,
            "status": status,
        })
    return _agent_cache


# ---------------------------------------------------------------------------
# Serve frontend UI
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder or "static", "index.html")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    agent = get_agent()
    granite = agent["generator"].granite
    return jsonify({
        "status": "ok",
        "rag_status": agent["status"],
        "demo_mode": granite.is_demo_mode(),
        "model": granite.model_id,
        "connection_error": granite._connection_error or None,
    })


# ---------------------------------------------------------------------------
# POST /profile  — parse resume file or build from inputs
# ---------------------------------------------------------------------------
@app.route("/profile", methods=["POST"])
def create_profile():
    """
    Accepts:
      - multipart/form-data with a 'resume' file field, OR
      - JSON body with {name, job_title, experience_level, years_experience, skills[]}
    Returns: parsed UserProfile dict
    """
    agent = get_agent()
    parser: ResumeParser = agent["parser"]

    if request.files.get("resume"):
        file = request.files["resume"]
        tmp_path = Path("/tmp") / (file.filename or "upload")
        file.save(str(tmp_path))
        try:
            profile = parser.parse_file(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        data = request.get_json(force=True) or {}
        profile = parser.build_from_inputs(
            name=data.get("name", ""),
            job_title=data.get("job_title", "Software Engineer"),
            experience_level=data.get("experience_level", "entry"),
            years_experience=int(data.get("years_experience", 0)),
            skills=data.get("skills", []),
        )
    return jsonify(profile.to_dict())


# ---------------------------------------------------------------------------
# POST /questions  — generate tailored question set
# ---------------------------------------------------------------------------
@app.route("/questions", methods=["POST"])
def generate_questions():
    """
    Body: {
      profile: { name, job_title, experience_level, years_experience, skills[] },
      num_questions: 10,
      question_types: ["technical", "behavioral", "hr"]  (optional)
    }
    """
    agent = get_agent()
    data = request.get_json(force=True) or {}
    profile_data = data.get("profile", {})

    profile = UserProfile(
        name=profile_data.get("name", ""),
        job_title=profile_data.get("job_title", "Software Engineer"),
        experience_level=profile_data.get("experience_level", "entry"),
        years_experience=int(profile_data.get("years_experience", 0)),
        skills=profile_data.get("skills", []),
    )

    result = agent["generator"].generate(
        profile=profile,
        num_questions=int(data.get("num_questions", 10)),
        question_types=data.get("question_types"),
    )
    return jsonify(result)


# ---------------------------------------------------------------------------
# POST /evaluate  — evaluate a candidate's answer
# ---------------------------------------------------------------------------
@app.route("/evaluate", methods=["POST"])
def evaluate_answer():
    """
    Body: { question: "...", answer: "..." }
    """
    agent = get_agent()
    data = request.get_json(force=True) or {}
    question = data.get("question", "")
    answer = data.get("answer", "")

    if not question or not answer:
        return jsonify({"error": "Both 'question' and 'answer' are required"}), 400

    result = agent["evaluator"].evaluate(question=question, answer=answer)
    return jsonify(result)


# ---------------------------------------------------------------------------
# POST /plan  — generate preparation plan
# ---------------------------------------------------------------------------
@app.route("/plan", methods=["POST"])
def preparation_plan():
    """
    Body: { profile: { ... } }
    """
    agent = get_agent()
    data = request.get_json(force=True) or {}
    profile_data = data.get("profile", {})

    profile = UserProfile(
        name=profile_data.get("name", ""),
        job_title=profile_data.get("job_title", "Software Engineer"),
        experience_level=profile_data.get("experience_level", "entry"),
        years_experience=int(profile_data.get("years_experience", 0)),
        skills=profile_data.get("skills", []),
    )

    plan = agent["planner"].create_plan(profile)
    tips = agent["planner"].get_improvement_tips(profile)
    return jsonify({"plan": plan["plan"], "tips": tips["tips"], "demo_mode": plan["demo_mode"]})


# ---------------------------------------------------------------------------
# POST /full-session  — end-to-end session: generate questions + plan
# ---------------------------------------------------------------------------
@app.route("/full-session", methods=["POST"])
def full_session():
    """
    Body: { profile: { ... }, num_questions: 10 }
    Combined endpoint for a complete interview prep session.
    """
    agent = get_agent()
    data = request.get_json(force=True) or {}
    profile_data = data.get("profile", {})

    profile = UserProfile(
        name=profile_data.get("name", ""),
        job_title=profile_data.get("job_title", "Software Engineer"),
        experience_level=profile_data.get("experience_level", "entry"),
        years_experience=int(profile_data.get("years_experience", 0)),
        skills=profile_data.get("skills", []),
    )

    questions = agent["generator"].generate(profile, num_questions=int(data.get("num_questions", 10)))
    plan = agent["planner"].create_plan(profile)
    tips = agent["planner"].get_improvement_tips(profile)

    return jsonify({
        "profile_summary": profile.summary(),
        "questions": questions["questions"],
        "preparation_plan": plan["plan"],
        "improvement_tips": tips["tips"],
        "demo_mode": questions["demo_mode"],
    })


if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    app.run(host=os.getenv("APP_HOST", "0.0.0.0"), port=port, debug=debug)
