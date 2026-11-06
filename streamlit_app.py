"""
Streamlit UI for the AI Interview Trainer Agent.
Run with:  streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src import build_agent, ResumeParser, UserProfile

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Interview Trainer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q_idx" not in st.session_state:
    st.session_state.current_q_idx = 0
if "evaluations" not in st.session_state:
    st.session_state.evaluations = {}


# ---------------------------------------------------------------------------
# Lazy agent loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI agent…")
def load_agent():
    generator, evaluator, planner, rag, parser, status = build_agent()
    return {
        "generator": generator,
        "evaluator": evaluator,
        "planner": planner,
        "rag": rag,
        "parser": parser,
        "status": status,
    }


def get_agent():
    if st.session_state.agent is None:
        st.session_state.agent = load_agent()
    return st.session_state.agent


# ---------------------------------------------------------------------------
# Sidebar — profile builder
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("🎯 AI Interview Trainer")
    st.sidebar.markdown("---")

    st.sidebar.header("👤 Your Profile")

    input_mode = st.sidebar.radio(
        "Profile source",
        ["Manual entry", "Upload resume"],
        horizontal=True,
    )

    agent = get_agent()

    if input_mode == "Manual entry":
        name = st.sidebar.text_input("Full name", placeholder="Jane Doe")
        job_title = st.sidebar.text_input("Target job title", placeholder="Software Engineer")
        exp_level = st.sidebar.selectbox(
            "Experience level",
            ["entry", "mid", "senior"],
            format_func=lambda x: x.capitalize(),
        )
        years = st.sidebar.number_input("Years of experience", min_value=0, max_value=40, value=0)
        skills_raw = st.sidebar.text_area(
            "Key skills (comma-separated)",
            placeholder="Python, SQL, Machine Learning",
        )
        skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

        if st.sidebar.button("✅ Set Profile", use_container_width=True):
            parser: ResumeParser = agent["parser"]
            profile = parser.build_from_inputs(
                name=name,
                job_title=job_title or "Software Engineer",
                experience_level=exp_level,
                years_experience=int(years),
                skills=skills,
            )
            st.session_state.profile = profile
            st.session_state.questions = []
            st.session_state.evaluations = {}
            st.sidebar.success("Profile saved!")

    else:
        uploaded = st.sidebar.file_uploader(
            "Upload your resume",
            type=["pdf", "docx", "txt"],
            help="Supported: PDF, DOCX, TXT",
        )
        if uploaded and st.sidebar.button("📄 Parse Resume", use_container_width=True):
            import tempfile, os
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                parser: ResumeParser = agent["parser"]
                profile = parser.parse_file(tmp_path)
                st.session_state.profile = profile
                st.session_state.questions = []
                st.session_state.evaluations = {}
                st.sidebar.success(f"Parsed: {profile.name or 'Unknown'}")
            except Exception as e:
                st.sidebar.error(f"Parse error: {e}")
            finally:
                os.unlink(tmp_path)

    # Show profile summary if set
    if st.session_state.profile:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Current profile**")
        p = st.session_state.profile
        st.sidebar.markdown(
            f"**{p.name or 'Unnamed'}** · {p.job_title}\n\n"
            f"{p.experience_level.capitalize()} · {p.years_experience} yrs\n\n"
            f"Skills: {', '.join(p.skills[:5]) or '—'}"
        )

    st.sidebar.markdown("---")
    # Connection status
    try:
        granite = agent["generator"].granite
        demo = granite.is_demo_mode()
        if demo:
            st.sidebar.warning("⚠️ Demo mode (no live LLM)")
        else:
            st.sidebar.success(f"✅ Connected · {granite.model_id}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tab 1 — Generate Questions
# ---------------------------------------------------------------------------
def tab_questions():
    st.header("📋 Generate Interview Questions")

    if not st.session_state.profile:
        st.info("Set up your profile in the sidebar first.")
        return

    p = st.session_state.profile
    st.markdown(
        f"Generating questions for **{p.name or 'you'}** — "
        f"**{p.job_title}** ({p.experience_level} level)"
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        num_q = st.slider("Number of questions", 3, 20, 10)
    with col2:
        q_types = st.multiselect(
            "Question types",
            ["technical", "behavioral", "hr"],
            default=["technical", "behavioral", "hr"],
        )

    if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
        agent = get_agent()
        with st.spinner("Generating tailored questions with IBM Granite…"):
            try:
                result = agent["generator"].generate(
                    profile=p,
                    num_questions=num_q,
                    question_types=q_types or None,
                )
                st.session_state.questions = result["questions"]
                st.session_state.current_q_idx = 0
                st.session_state.evaluations = {}
                if result.get("demo_mode"):
                    st.warning("⚠️ Running in demo mode — responses are illustrative.")
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return

    questions = st.session_state.questions
    if not questions:
        return

    st.markdown(f"---\n**{len(questions)} questions generated**")

    for i, q in enumerate(questions):
        difficulty_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(
            q.get("difficulty", "medium"), "⚪"
        )
        with st.expander(
            f"{difficulty_color} Q{i + 1}. {q['question'][:90]}{'…' if len(q['question']) > 90 else ''}",
            expanded=(i == 0),
        ):
            st.markdown(f"**Question:** {q['question']}")
            cols = st.columns(3)
            cols[0].markdown(f"**Category:** {q.get('category', '—')}")
            cols[1].markdown(f"**Difficulty:** {q.get('difficulty', '—').capitalize()}")
            cols[2].markdown(f"**Evaluating:** {q.get('evaluating', '—')}")

            # Inline answer & evaluate
            answer_key = f"answer_{i}"
            answer = st.text_area(
                "Your answer",
                key=answer_key,
                placeholder="Type your answer here…",
                height=120,
            )
            if st.button(f"📊 Evaluate Answer", key=f"eval_btn_{i}"):
                if not answer.strip():
                    st.warning("Please type an answer first.")
                else:
                    agent = get_agent()
                    with st.spinner("Evaluating…"):
                        try:
                            eval_result = agent["evaluator"].evaluate(
                                question=q["question"], answer=answer
                            )
                            st.session_state.evaluations[i] = eval_result
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")

            if i in st.session_state.evaluations:
                _render_feedback(st.session_state.evaluations[i])


def _render_feedback(eval_result: dict):
    fb = eval_result.get("structured", {})
    st.markdown("---")
    st.markdown("#### 📊 Feedback")

    score = fb.get("overall_score", "")
    if score:
        st.metric("Score", score)

    col1, col2 = st.columns(2)
    if fb.get("strengths"):
        col1.markdown("**✅ Strengths**")
        col1.markdown(fb["strengths"])
    if fb.get("improvements"):
        col2.markdown("**🔧 Areas for Improvement**")
        col2.markdown(fb["improvements"])

    if fb.get("model_answer"):
        st.markdown("**💡 Model Answer Snippet**")
        st.markdown(fb["model_answer"])

    if fb.get("star_check"):
        st.markdown("**⭐ STAR Check**")
        st.markdown(fb["star_check"])

    if fb.get("key_tip"):
        st.info(f"**One Key Tip:** {fb['key_tip']}")

    # Raw fallback
    if not any(fb.values()):
        st.markdown(eval_result.get("feedback_raw", ""))


# ---------------------------------------------------------------------------
# Tab 2 — Evaluate a Single Answer
# ---------------------------------------------------------------------------
def tab_evaluate():
    st.header("📊 Evaluate Any Answer")
    st.markdown("Paste any interview question and your answer to get structured feedback.")

    question = st.text_area("Interview question", height=80, placeholder="Tell me about a time you…")
    answer = st.text_area("Your answer", height=160, placeholder="In my previous role…")

    if st.button("📊 Get Feedback", type="primary", use_container_width=True):
        if not question.strip() or not answer.strip():
            st.warning("Both question and answer are required.")
            return
        agent = get_agent()
        with st.spinner("Evaluating with IBM Granite…"):
            try:
                result = agent["evaluator"].evaluate(question=question, answer=answer)
                if result.get("demo_mode"):
                    st.warning("⚠️ Running in demo mode — responses are illustrative.")
                _render_feedback(result)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


# ---------------------------------------------------------------------------
# Tab 3 — Preparation Plan
# ---------------------------------------------------------------------------
def tab_plan():
    st.header("📅 Preparation Plan")

    if not st.session_state.profile:
        st.info("Set up your profile in the sidebar first.")
        return

    p = st.session_state.profile
    st.markdown(
        f"Generating a personalized 4-week prep plan for **{p.name or 'you'}** "
        f"targeting **{p.job_title}** ({p.experience_level} level)."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📅 Generate Prep Plan", type="primary", use_container_width=True):
            agent = get_agent()
            with st.spinner("Building your plan…"):
                try:
                    result = agent["planner"].create_plan(p)
                    st.session_state["prep_plan"] = result
                    if result.get("demo_mode"):
                        st.warning("⚠️ Running in demo mode.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    with col2:
        if st.button("💡 Get Improvement Tips", type="primary", use_container_width=True):
            agent = get_agent()
            with st.spinner("Generating tips…"):
                try:
                    tips_result = agent["planner"].get_improvement_tips(p)
                    st.session_state["improvement_tips"] = tips_result
                    if tips_result.get("demo_mode"):
                        st.warning("⚠️ Running in demo mode.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    if "prep_plan" in st.session_state:
        st.markdown("---")
        st.subheader("📅 Your 4-Week Plan")
        st.markdown(st.session_state["prep_plan"]["plan"])

    if "improvement_tips" in st.session_state:
        st.markdown("---")
        st.subheader("💡 Improvement Tips")
        st.markdown(st.session_state["improvement_tips"]["tips"])


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    st.title("🎯 AI Interview Trainer")
    st.markdown(
        "Powered by **IBM Granite** · Personalized questions, answer feedback, and prep plans."
    )
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📋 Interview Questions", "📊 Evaluate Answer", "📅 Prep Plan"]
    )

    with tab1:
        tab_questions()

    with tab2:
        tab_evaluate()

    with tab3:
        tab_plan()


if __name__ == "__main__":
    main()
