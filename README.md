# 🎯 IBM Granite — AI Interview Trainer Agent

> **Problem Statement #22** — Powered by **IBM Granite (watsonx.ai)** + **RAG (ChromaDB)** + **IBM Cloud Lite**

An intelligent interview preparation assistant that uses Retrieval-Augmented Generation to deliver hyper-personalized question sets, real-time answer feedback, and structured preparation plans tailored to a candidate's profile, experience level, and target role.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Interview Trainer Agent                       │
│                                                                 │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │ Resume/      │    │  RAG Engine   │    │  IBM Granite    │  │
│  │ Profile      │───▶│  (ChromaDB +  │───▶│  (watsonx.ai)   │  │
│  │ Parser       │    │  MiniLM-L6)   │    │  granite-13b    │  │
│  └──────────────┘    └───────────────┘    └─────────────────┘  │
│          │                   │                      │           │
│          ▼                   ▼                      ▼           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Interview Agent Core                        │  │
│  │  • QuestionGenerator   • AnswerEvaluator                 │  │
│  │  • PreparationPlanner  • ImprovementTips                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│          │                                                       │
│    ┌─────┴──────┐                                               │
│    │            │                                               │
│    ▼            ▼                                               │
│ Flask API   Rich CLI                                            │
│  + Web UI   (Interactive)                                       │
└─────────────────────────────────────────────────────────────────┘

Knowledge Base (RAG Sources):
  ├── interview_questions.json    (role × level × type)
  ├── behavioral_scenarios.json  (STAR competency scenarios)
  ├── hr_guidelines.json         (recruitment process guidelines)
  └── industry_expectations.json (role-specific interview formats)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configure IBM watsonx.ai

```bash
cp .env.example .env
```

Edit `.env` and fill in your IBM Cloud credentials:

```env
WATSONX_API_KEY=your_ibm_watsonx_api_key
WATSONX_PROJECT_ID=your_ibm_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-13b-instruct-v2
```

> **Get credentials**: Log in to [IBM Cloud](https://cloud.ibm.com), create a **watsonx.ai** project under IBM Cloud Lite (free tier), then generate an API key from **Manage → Access (IAM) → API Keys**.

### 3. Run the Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### 4. Run the CLI

```bash
# Interactive guided session
python cli.py start

# With resume file
python cli.py start --resume path/to/resume.pdf

# Quick question generation (non-interactive)
python cli.py questions --role "Data Scientist" --level mid --skills "python,ml,sql" --num 10

# Evaluate a specific answer
python cli.py evaluate \
  --question "Tell me about a time you resolved a conflict." \
  --answer "I once had a disagreement with a colleague about the architecture..."
```

---

## 📁 Project Structure

```
ai-interview-trainer/
├── app.py                          # Flask REST API + serves Web UI
├── cli.py                          # Rich interactive CLI
├── requirements.txt
├── .env.example                    # IBM credentials template
│
├── src/
│   ├── __init__.py                 # Component factory (build_agent)
│   ├── granite_client.py           # IBM Granite / watsonx.ai wrapper
│   ├── rag_engine.py               # ChromaDB RAG pipeline
│   ├── resume_parser.py            # PDF/DOCX/text resume parser
│   └── interview_agent.py          # Question generator, evaluator, planner
│
├── data/
│   ├── knowledge_base/
│   │   ├── interview_questions.json    # Role × level × type question bank
│   │   ├── behavioral_scenarios.json   # STAR behavioral scenarios
│   │   ├── hr_guidelines.json          # HR recruitment guidelines
│   │   └── industry_expectations.json  # Interview formats by role/company
│   └── chroma_db/                      # Auto-created vector index
│
├── static/
│   └── index.html                  # Single-page web UI
│
└── tests/
    └── test_agent.py               # Pytest test suite (demo mode)
```

---

## 🔌 REST API Reference

### `GET /health`
Returns agent status, model name, and whether demo mode is active.

### `POST /profile`
Parse a resume or build a profile from structured inputs.
```json
{
  "name": "Priya Sharma",
  "job_title": "Software Engineer",
  "experience_level": "mid",
  "years_experience": 4,
  "skills": ["python", "aws", "django"]
}
```

### `POST /questions`
Generate tailored interview questions.
```json
{
  "profile": { "job_title": "Data Scientist", "experience_level": "entry", "skills": ["python", "ml"] },
  "num_questions": 10
}
```

### `POST /evaluate`
Evaluate a candidate's answer with structured AI feedback.
```json
{
  "question": "Tell me about a time you resolved a conflict.",
  "answer": "In my last role, I had a disagreement with..."
}
```

### `POST /plan`
Generate a 4-week preparation plan + improvement tips.
```json
{ "profile": { "job_title": "Software Engineer", "experience_level": "senior" } }
```

### `POST /full-session`
Combined endpoint — questions + plan + tips in one request.

---

## 🤖 IBM Granite Integration

The agent uses **IBM Granite 13B Instruct** (`ibm/granite-13b-instruct-v2`) via the **ibm-watsonx-ai** SDK:

- **RAG-augmented prompts**: Every generation call retrieves top-5 relevant documents from ChromaDB to ground the model's response.
- **Instruction-tuned prompts**: Prompts are structured for Granite's instruction-following format.
- **Demo mode**: If credentials are absent, the agent stubs responses clearly labeled `[DEMO MODE]` — the full architecture still runs.

### IBM Cloud Lite Usage
| Service | Usage |
|---|---|
| IBM watsonx.ai | Granite LLM inference (free tier: 50K tokens/month) |
| IBM Cloud IAM | API key management |

---

## 🧪 Running Tests

```bash
# All tests run in demo mode — no IBM credentials needed
pytest tests/ -v

# With coverage
pytest tests/ -v --tb=short
```

---

## 💡 Features

| Feature | Description |
|---|---|
| **Resume Parsing** | PDF, DOCX, TXT — extracts skills, experience, education |
| **Profile Builder** | Manual input when no resume is available |
| **RAG Retrieval** | ChromaDB + MiniLM embeddings over 4 curated knowledge bases |
| **Question Generation** | 40% technical, 40% behavioral, 20% HR — role + level adaptive |
| **Answer Evaluation** | Score, strengths, improvements, model answer, STAR check |
| **Prep Plan** | 4-week structured plan with daily activities |
| **Improvement Tips** | 5 specific, actionable tips per profile |
| **Web UI** | Dark-mode single-page app served from Flask |
| **Rich CLI** | Interactive terminal session with colors and spinners |
| **IBM Granite** | All generation via `ibm/granite-13b-instruct-v2` on watsonx.ai |

---

## 🛠️ Extending the Knowledge Base

Add entries to any file in `data/knowledge_base/`. The RAG engine auto-reindexes on next startup when the ChromaDB collection is empty. To force a rebuild, delete `data/chroma_db/`.

---

*Built for IBM Cloud Lite · IBM Granite (watsonx.ai) · Problem Statement #22*
