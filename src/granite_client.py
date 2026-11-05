"""
IBM Granite LLM client for Interview Trainer Agent.
Wraps ibm-watsonx-ai SDK with interview-specific prompt templates.
"""

import os
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    IBM_AVAILABLE = True
except ImportError:
    Credentials = None  # type: ignore[assignment]
    ModelInference = None  # type: ignore[assignment]
    IBM_AVAILABLE = False


class GraniteClient:
    """Thin wrapper around IBM Granite via watsonx.ai."""

    DEFAULT_MODEL = "ibm/granite-4-h-small"
    FALLBACK_MODEL = "ibm/granite-3-1-8b-base"

    def __init__(self):
        self.api_key = os.getenv("WATSONX_API_KEY", "")
        self.project_id = os.getenv("WATSONX_PROJECT_ID", "")
        self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.model_id = os.getenv("GRANITE_MODEL_ID", self.DEFAULT_MODEL)
        self._model: Optional[Any] = None
        self._demo_mode = not (self.api_key and self.project_id and IBM_AVAILABLE)
        self._connection_error: str = ""

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._demo_mode or Credentials is None or ModelInference is None:
            return None
        try:
            credentials = Credentials(url=self.url, api_key=self.api_key)
            self._model = ModelInference(
                model_id=self.model_id,
                credentials=credentials,
                project_id=self.project_id,
                # Use chat-compatible params (granite-4 uses max_tokens not max_new_tokens)
                params={
                    "max_tokens": int(os.getenv("MAX_TOKENS", 2048)),
                    "temperature": float(os.getenv("TEMPERATURE", 0.7)),
                    "top_p": 0.9,
                },
            )
        except Exception as e:
            # WML service not yet associated with project — fall back to demo mode
            self._demo_mode = True
            self._connection_error = str(e)
            return None
        return self._model

    def generate(self, prompt: str) -> str:
        """Generate text from IBM Granite or demo stub."""
        model = self._get_model()
        if model is None:
            return self._demo_response(prompt)
        # Use chat API (newer endpoint) to avoid deprecation warning
        try:
            response = model.chat(messages=[{"role": "user", "content": prompt}])
            return response["choices"][0]["message"]["content"].strip()
        except Exception:
            # Fallback to generate_text for models that don't support chat
            response = model.generate_text(prompt=prompt)
            return response.strip()

    def is_demo_mode(self) -> bool:
        return self._demo_mode

    # ------------------------------------------------------------------
    # Demo / stub responses for environments without IBM credentials
    # ------------------------------------------------------------------
    def _demo_response(self, prompt: str) -> str:
        """Return a clearly-labeled demo answer when IBM creds are absent."""
        if "generate questions" in prompt.lower():
            return (
                "[DEMO MODE — Connect IBM watsonx credentials for live responses]\n\n"
                "Here are 5 tailored interview questions based on your profile:\n"
                "1. Walk me through your most complex technical project end-to-end.\n"
                "2. How do you approach debugging a production issue under time pressure?\n"
                "3. Describe a time you had to learn a new technology quickly.\n"
                "4. How do you balance technical debt with feature delivery?\n"
                "5. Tell me about a failed project and what you learned from it."
            )
        if "evaluate" in prompt.lower() or "feedback" in prompt.lower():
            return (
                "[DEMO MODE]\n\n"
                "**Feedback on your answer:**\n"
                "✅ Strength: You provided a concrete example which is excellent.\n"
                "⚠️  Improvement: Quantify the impact of your actions — numbers make answers memorable.\n"
                "💡 Tip: Use the STAR format (Situation, Task, Action, Result) for behavioral questions.\n"
                "**Score: 7/10** — Good structure, needs measurable outcomes."
            )
        if "preparation plan" in prompt.lower():
            return (
                "[DEMO MODE]\n\n"
                "**Your 4-Week Interview Preparation Plan:**\n"
                "Week 1: Master fundamentals — review core CS concepts and data structures.\n"
                "Week 2: Practice coding — solve 2 LeetCode problems daily (easy→medium).\n"
                "Week 3: System design — study 3 canonical architectures (URL shortener, chat system, feed).\n"
                "Week 4: Mock interviews — record yourself answering behavioral questions.\n"
                "Daily: Review 1 industry expectation document and 2 STAR stories."
            )
        return (
            "[DEMO MODE — IBM Granite response would appear here]\n"
            "Add your WATSONX_API_KEY and WATSONX_PROJECT_ID to .env to enable live generation."
        )
