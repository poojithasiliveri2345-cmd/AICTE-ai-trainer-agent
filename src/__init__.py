"""
Interview Trainer Agent — main entry point.
Composes all components: RAG engine, Granite LLM, parser, generator, evaluator.
"""

from .granite_client import GraniteClient
from .rag_engine import RAGEngine
from .resume_parser import ResumeParser, UserProfile
from .interview_agent import QuestionGenerator, AnswerEvaluator, PreparationPlanner

__all__ = [
    "GraniteClient",
    "RAGEngine",
    "ResumeParser",
    "UserProfile",
    "QuestionGenerator",
    "AnswerEvaluator",
    "PreparationPlanner",
    "build_agent",
]


def build_agent():
    """Factory: instantiate and wire all components. Returns (generator, evaluator, planner, rag, parser)."""
    granite = GraniteClient()
    rag = RAGEngine()
    status = rag.initialize()
    parser = ResumeParser()
    generator = QuestionGenerator(granite, rag)
    evaluator = AnswerEvaluator(granite, rag)
    planner = PreparationPlanner(granite, rag)
    return generator, evaluator, planner, rag, parser, status
