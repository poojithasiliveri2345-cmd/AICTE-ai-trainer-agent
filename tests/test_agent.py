"""
Unit tests for the Interview Trainer Agent.
Tests run in demo mode (no IBM credentials required).
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest


# ---------------------------------------------------------------------------
# Resume Parser tests
# ---------------------------------------------------------------------------

class TestResumeParser:

    def setup_method(self):
        from src.resume_parser import ResumeParser
        self.parser = ResumeParser()

    def test_build_from_inputs(self):
        profile = self.parser.build_from_inputs(
            name="Alice Kumar",
            job_title="Software Engineer",
            experience_level="mid",
            years_experience=4,
            skills=["python", "aws", "django"],
        )
        assert profile.name == "Alice Kumar"
        assert profile.job_title == "Software Engineer"
        assert profile.experience_level == "mid"
        assert profile.years_experience == 4
        assert "python" in profile.skills
        assert profile.source == "manual"

    def test_parse_text(self):
        sample = """
        John Doe
        john.doe@email.com | +1-555-123-4567
        Senior Software Engineer
        7 years of experience
        Skills: Python, AWS, Kubernetes, Docker, PostgreSQL
        """
        profile = self.parser.parse_text(sample)
        assert profile.email == "john.doe@email.com"
        assert "python" in profile.skills or "aws" in profile.skills
        assert profile.years_experience == 7
        assert profile.experience_level == "senior"

    def test_experience_level_inference(self):
        p_entry = self.parser.build_from_inputs("A", "SE", "junior", 1, [])
        assert p_entry.experience_level == "entry"

        p_senior = self.parser.build_from_inputs("A", "SE", "senior", 8, [])
        assert p_senior.experience_level == "senior"

    def test_profile_summary(self):
        profile = self.parser.build_from_inputs("Bob", "Data Scientist", "mid", 4, ["python", "ml"])
        summary = profile.summary()
        assert "Bob" in summary
        assert "Data Scientist" in summary
        assert "mid" in summary

    def test_to_dict(self):
        profile = self.parser.build_from_inputs("Alice", "PM", "entry", 1, [])
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "Alice"
        assert d["job_title"] == "PM"

    def test_skill_extraction(self):
        text = "Proficient in Python, TensorFlow, and AWS. Experience with Docker and Kubernetes."
        profile = self.parser.parse_text(text)
        assert "python" in profile.skills
        assert "aws" in profile.skills
        assert "docker" in profile.skills


# ---------------------------------------------------------------------------
# Granite Client tests (demo mode)
# ---------------------------------------------------------------------------

class TestGraniteClientDemoMode:

    def setup_method(self):
        # Ensure demo mode by unsetting env vars temporarily
        self._orig_key = os.environ.pop("WATSONX_API_KEY", None)
        self._orig_proj = os.environ.pop("WATSONX_PROJECT_ID", None)
        from src.granite_client import GraniteClient
        self.client = GraniteClient()

    def teardown_method(self):
        if self._orig_key:
            os.environ["WATSONX_API_KEY"] = self._orig_key
        if self._orig_proj:
            os.environ["WATSONX_PROJECT_ID"] = self._orig_proj

    def test_is_demo_mode(self):
        assert self.client.is_demo_mode() is True

    def test_generate_questions_demo(self):
        response = self.client.generate("generate questions for python developer")
        assert isinstance(response, str)
        assert len(response) > 0
        assert "DEMO MODE" in response

    def test_generate_feedback_demo(self):
        response = self.client.generate("evaluate this answer and provide feedback")
        assert "DEMO MODE" in response
        assert "Feedback" in response or "Score" in response

    def test_generate_plan_demo(self):
        response = self.client.generate("create a preparation plan for the interview")
        assert "DEMO MODE" in response
        assert "Week" in response


# ---------------------------------------------------------------------------
# RAG Engine tests
# ---------------------------------------------------------------------------

class TestRAGEngine:

    def setup_method(self):
        from src.rag_engine import RAGEngine
        self.rag = RAGEngine()
        self.status = self.rag.initialize()

    def test_initialize_loads_documents(self):
        assert self.rag._initialized is True
        assert len(self.rag._docs) > 0

    def test_retrieve_returns_results(self):
        results = self.rag.retrieve("python software engineer technical questions", top_k=3)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_retrieve_behavioral(self):
        results = self.rag.retrieve("conflict resolution behavioral interview", top_k=3)
        assert len(results) >= 1
        texts = " ".join(r["text"] for r in results).lower()
        # Should find something about conflict or behavior
        assert any(kw in texts for kw in ["conflict", "behavior", "scenario", "situation"])

    def test_build_context(self):
        results = self.rag.retrieve("software engineer interview", top_k=2)
        context = self.rag.build_context(results)
        assert isinstance(context, str)
        assert "Context" in context

    def test_knowledge_base_files_exist(self):
        kb_dir = Path(__file__).parent.parent / "data" / "knowledge_base"
        assert (kb_dir / "interview_questions.json").exists()
        assert (kb_dir / "behavioral_scenarios.json").exists()
        assert (kb_dir / "hr_guidelines.json").exists()
        assert (kb_dir / "industry_expectations.json").exists()


# ---------------------------------------------------------------------------
# Question Generator tests (demo mode)
# ---------------------------------------------------------------------------

class TestQuestionGenerator:

    def setup_method(self):
        os.environ.pop("WATSONX_API_KEY", None)
        os.environ.pop("WATSONX_PROJECT_ID", None)
        from src.granite_client import GraniteClient
        from src.rag_engine import RAGEngine
        from src.interview_agent import QuestionGenerator
        from src.resume_parser import ResumeParser
        self.rag = RAGEngine()
        self.rag.initialize()
        self.granite = GraniteClient()
        self.generator = QuestionGenerator(self.granite, self.rag)
        self.parser = ResumeParser()

    def test_generate_returns_dict(self):
        profile = self.parser.build_from_inputs("Alice", "Software Engineer", "entry", 1, ["python"])
        result = self.generator.generate(profile, num_questions=5)
        assert isinstance(result, dict)
        assert "questions" in result
        assert "demo_mode" in result
        assert result["demo_mode"] is True

    def test_generate_questions_list(self):
        profile = self.parser.build_from_inputs("Bob", "Data Scientist", "mid", 3, ["python", "ml"])
        result = self.generator.generate(profile, num_questions=5)
        assert isinstance(result["questions"], list)
        assert len(result["questions"]) >= 1


# ---------------------------------------------------------------------------
# Answer Evaluator tests (demo mode)
# ---------------------------------------------------------------------------

class TestAnswerEvaluator:

    def setup_method(self):
        os.environ.pop("WATSONX_API_KEY", None)
        os.environ.pop("WATSONX_PROJECT_ID", None)
        from src.granite_client import GraniteClient
        from src.rag_engine import RAGEngine
        from src.interview_agent import AnswerEvaluator
        self.rag = RAGEngine()
        self.rag.initialize()
        self.granite = GraniteClient()
        self.evaluator = AnswerEvaluator(self.granite, self.rag)

    def test_evaluate_returns_dict(self):
        result = self.evaluator.evaluate(
            question="Tell me about yourself.",
            answer="I am a software engineer with 3 years of experience in Python and AWS.",
        )
        assert isinstance(result, dict)
        assert "feedback_raw" in result
        assert "structured" in result
        assert result["demo_mode"] is True

    def test_evaluate_non_empty_feedback(self):
        result = self.evaluator.evaluate(
            question="How do you handle conflict?",
            answer="I always try to communicate openly.",
        )
        assert len(result["feedback_raw"]) > 0


# ---------------------------------------------------------------------------
# Knowledge base integrity tests
# ---------------------------------------------------------------------------

class TestKnowledgeBaseIntegrity:

    def test_interview_questions_valid_json(self):
        path = Path("data/knowledge_base/interview_questions.json")
        with open(path) as f:
            data = json.load(f)
        assert "software_engineer" in data
        assert "generic" in data

    def test_behavioral_scenarios_structure(self):
        path = Path("data/knowledge_base/behavioral_scenarios.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        for item in data:
            assert "scenario" in item
            assert "competency" in item
            assert "model_answer_structure" in item

    def test_hr_guidelines_structure(self):
        path = Path("data/knowledge_base/hr_guidelines.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        for item in data:
            assert "guideline" in item
            assert "tips" in item

    def test_industry_expectations_structure(self):
        path = Path("data/knowledge_base/industry_expectations.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        for item in data:
            assert "role" in item
            assert "interview_rounds" in item
