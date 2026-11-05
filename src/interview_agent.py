"""
Interview question generator and answer evaluator.
Uses RAG-augmented IBM Granite prompts to produce tailored question sets
and actionable feedback on user answers.
"""

from typing import List, Dict, Any, Optional
from .granite_client import GraniteClient
from .rag_engine import RAGEngine
from .resume_parser import UserProfile


# ---------------------------------------------------------------------------
# Prompt templates for IBM Granite (instruction-tuned format)
# ---------------------------------------------------------------------------

QUESTION_GENERATION_PROMPT = """\
You are an expert interview coach with deep knowledge of technical and behavioral interviews.

## Candidate Profile
{profile_summary}

## Retrieved Interview Context
{context}

## Task
Generate {num_questions} tailored interview questions for this candidate applying for the role of {role} at {level} level.

Requirements:
- Include a mix of: technical (40%), behavioral (40%), HR/motivational (20%)
- Questions must be realistic and relevant to the candidate's background
- For each question, also provide:
  * Category (e.g., Data Structures, Conflict Resolution, Leadership)
  * Difficulty: easy / medium / hard
  * A brief interviewer tip on what they are evaluating

Format each question as:
Q[N]. [Question text]
Category: [category]
Difficulty: [easy/medium/hard]
Evaluating: [what the interviewer looks for]

Begin:
"""

ANSWER_EVALUATION_PROMPT = """\
You are a senior interview coach evaluating a candidate's answer.

## Question Asked
{question}

## Candidate's Answer
{answer}

## Retrieved Best-Practice Context
{context}

## Evaluation Task
Provide structured feedback on the candidate's answer with the following sections:

**Overall Score**: X/10
**Strengths**: What the candidate did well (be specific)
**Areas for Improvement**: What is missing or weak (be specific and actionable)
**Model Answer Snippet**: A brief example of a stronger answer
**STAR Check** (for behavioral questions): Did the candidate cover Situation, Task, Action, Result?
**One Key Tip**: The single most important thing to improve

Be encouraging but honest. Focus on actionable improvements.
"""

PREPARATION_PLAN_PROMPT = """\
You are an expert career coach creating a personalized interview preparation plan.

## Candidate Profile
{profile_summary}

## Target Role
{role} ({level} level)

## Retrieved Industry Context
{context}

## Task
Create a detailed 4-week interview preparation plan tailored to this candidate.

Include:
1. Week-by-week focus areas
2. Daily practice recommendations (time-boxed activities)
3. Resources to study (topics, not specific URLs)
4. Key strengths to leverage in interviews
5. Top 3 weaknesses to address before the interview
6. Mock interview strategy

Make the plan specific, actionable, and realistic for someone with this background.
"""

IMPROVEMENT_TIPS_PROMPT = """\
You are an expert interview coach. Based on the following candidate profile and the interview domain, 
provide 5 highly specific, actionable improvement tips to increase their success rate.

## Candidate Profile
{profile_summary}

## Context
{context}

## Tips Format
For each tip:
TIP [N]: [Short title]
Detail: [Specific, actionable advice — not generic platitudes]
Impact: [Why this will improve interview performance]
"""


# ---------------------------------------------------------------------------
# Question Generator
# ---------------------------------------------------------------------------

class QuestionGenerator:

    def __init__(self, granite: GraniteClient, rag: RAGEngine):
        self.granite = granite
        self.rag = rag

    def generate(
        self,
        profile: UserProfile,
        num_questions: int = 10,
        question_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a tailored question set for the given profile.
        Returns a dict with questions list and metadata.
        """
        query = f"{profile.job_title} {profile.experience_level} interview questions"
        if profile.skills:
            query += f" {' '.join(profile.skills[:5])}"

        results = self.rag.retrieve(query, top_k=5)
        context = self.rag.build_context(results)

        prompt = QUESTION_GENERATION_PROMPT.format(
            profile_summary=profile.summary(),
            context=context,
            num_questions=num_questions,
            role=profile.job_title or "Software Engineer",
            level=profile.experience_level or "entry",
        )

        raw_response = self.granite.generate(prompt)
        questions = self._parse_questions(raw_response)

        return {
            "questions": questions,
            "raw_response": raw_response,
            "profile": profile.to_dict(),
            "context_sources": [r["metadata"].get("source", "kb") for r in results],
            "demo_mode": self.granite.is_demo_mode(),
        }

    def _parse_questions(self, raw: str) -> List[Dict[str, str]]:
        """Parse the LLM's formatted output into structured question objects."""
        questions = []
        blocks = raw.split("\n\n")
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue
            q_line = next((l for l in lines if re.match(r"Q\d+\.", l)), None)
            if not q_line:
                # Try to find any numbered question
                q_line = next((l for l in lines if re.match(r"\d+\.", l)), None)
            if q_line:
                q_text = re.sub(r"^Q?\d+\.\s*", "", q_line)
                category = self._extract_field(lines, "Category")
                difficulty = self._extract_field(lines, "Difficulty")
                evaluating = self._extract_field(lines, "Evaluating")
                questions.append({
                    "question": q_text,
                    "category": category or "General",
                    "difficulty": difficulty or "medium",
                    "evaluating": evaluating or "",
                })
        return questions if questions else [{"question": raw, "category": "General", "difficulty": "medium", "evaluating": ""}]

    def _extract_field(self, lines: List[str], field: str) -> str:
        for line in lines:
            if line.lower().startswith(field.lower() + ":"):
                return line.split(":", 1)[1].strip()
        return ""


# ---------------------------------------------------------------------------
# Answer Evaluator
# ---------------------------------------------------------------------------

class AnswerEvaluator:

    def __init__(self, granite: GraniteClient, rag: RAGEngine):
        self.granite = granite
        self.rag = rag

    def evaluate(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate a candidate's answer and return structured feedback."""
        query = f"best answer feedback {question}"
        results = self.rag.retrieve(query, top_k=3)
        context = self.rag.build_context(results)

        prompt = ANSWER_EVALUATION_PROMPT.format(
            question=question,
            answer=answer,
            context=context,
        )

        feedback_text = self.granite.generate(prompt)
        structured = self._parse_feedback(feedback_text)

        return {
            "question": question,
            "answer": answer,
            "feedback_raw": feedback_text,
            "structured": structured,
            "demo_mode": self.granite.is_demo_mode(),
        }

    def _parse_feedback(self, raw: str) -> Dict[str, str]:
        sections = {
            "overall_score": "",
            "strengths": "",
            "improvements": "",
            "model_answer": "",
            "star_check": "",
            "key_tip": "",
        }
        patterns = {
            "overall_score": r"\*\*Overall Score\*\*:?\s*(.+)",
            "strengths": r"\*\*Strengths?\*\*:?\s*(.+?)(?=\*\*|\Z)",
            "improvements": r"\*\*Areas? for Improvement\*\*:?\s*(.+?)(?=\*\*|\Z)",
            "model_answer": r"\*\*Model Answer[^*]*\*\*:?\s*(.+?)(?=\*\*|\Z)",
            "star_check": r"\*\*STAR Check\*\*:?\s*(.+?)(?=\*\*|\Z)",
            "key_tip": r"\*\*One Key Tip\*\*:?\s*(.+?)(?=\*\*|\Z)",
        }
        import re
        for key, pat in patterns.items():
            m = re.search(pat, raw, re.S | re.I)
            if m:
                sections[key] = m.group(1).strip()
        return sections


# ---------------------------------------------------------------------------
# Preparation Planner
# ---------------------------------------------------------------------------

class PreparationPlanner:

    def __init__(self, granite: GraniteClient, rag: RAGEngine):
        self.granite = granite
        self.rag = rag

    def create_plan(self, profile: UserProfile) -> Dict[str, Any]:
        """Generate a personalized preparation plan."""
        query = f"{profile.job_title} interview preparation strategy {profile.experience_level}"
        results = self.rag.retrieve(query, top_k=5)
        context = self.rag.build_context(results)

        prompt = PREPARATION_PLAN_PROMPT.format(
            profile_summary=profile.summary(),
            role=profile.job_title or "Software Engineer",
            level=profile.experience_level or "entry",
            context=context,
        )

        plan_text = self.granite.generate(prompt)
        return {
            "plan": plan_text,
            "profile": profile.to_dict(),
            "demo_mode": self.granite.is_demo_mode(),
        }

    def get_improvement_tips(self, profile: UserProfile) -> Dict[str, Any]:
        """Generate specific improvement tips for the candidate."""
        query = f"interview tips {profile.job_title} {' '.join(profile.skills[:3])}"
        results = self.rag.retrieve(query, top_k=4)
        context = self.rag.build_context(results)

        prompt = IMPROVEMENT_TIPS_PROMPT.format(
            profile_summary=profile.summary(),
            context=context,
        )

        tips_text = self.granite.generate(prompt)
        return {
            "tips": tips_text,
            "demo_mode": self.granite.is_demo_mode(),
        }


import re  # noqa: E402 — needed at module level for _parse_questions
