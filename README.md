# IBM Granite — AI Interview Trainer Agent

> **AICTE Problem Statement #22** · Powered by **IBM Granite (watsonx.ai)** · RAG with ChromaDB · IBM Cloud Lite

An intelligent interview preparation assistant that uses Retrieval-Augmented Generation to deliver hyper-personalised question sets, real-time answer feedback, and structured preparation plans — all tailored to a candidate's profile, experience level, and target role.

---

## Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [REST API Reference](#-rest-api-reference)
- [Running Tests](#-running-tests)
- [IBM Granite Integration](#-ibm-granite-integration)
- [Extending the Knowledge Base](#-extending-the-knowledge-base)

---

## Features

| Feature | Description |
|---|---|
| **Resume Parsing** | PDF, DOCX, TXT — extracts skills, experience, and education |
| **Profile Builder** | Manual structured input when no resume is available |
| **RAG Retrieval** | ChromaDB + MiniLM-L6 embeddings over 4 curated knowledge bases |
| **Question Generation** | 40% technical, 40% behavioural, 20% HR — role + level adaptive |
| **Answer Evaluation** | Score, strengths, improvements, model answer, STAR check |
| **Prep Plan** | 4-week structured plan with daily activities |
| **Improvement Tips** | 5 specific, actionable tips per profile |
| **Web UI** | Dark-mode single-page app served from Flask |
| **Rich CLI** | Interactive terminal session with colours and spinners |
| **Demo Mode** | Full pipeline runs without IBM credentials — responses labelled `[DEMO MODE]` |

---

## Architecture

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

## Project Structure

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
│   │   ├── behavioral_scenarios.json   # STAR behavioural scenarios
│   │   ├── hr_guidelines.json          # HR recruitment guidelines
│   │   └── industry_expectations.json  # Interview formats by role/company
│   └── chroma_db/                      # Auto-created vector index (git-ignored)
│
├── static/
│   └── index.html                  # Single-page web UI
│
└── tests/
    └── test_agent.py               # Pytest test suite (demo mode)
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/poojithasiliveri2345-cmd/AICTE-ai-trainer-agent.git
cd AICTE-ai-trainer-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Then edit .env with your IBM credentials (see Configuration below)
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your IBM watsonx.ai credentials:

```env
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

> **No credentials?** The app runs in **Demo Mode** automatically — all features work but LLM responses are stubbed and labelled `[DEMO MODE]`.

To get free credentials:
1. Sign up at [IBM Cloud](https://cloud.ibm.com/registration) (Lite tier — no credit card needed)
2. Create a **watsonx.ai** project
3. Generate an **API key** from IAM → Service credentials

---

## Running the App

### Web UI (Flask)

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Interactive CLI

```bash
python cli.py
```

---

## REST API Reference

### `GET /health`
Returns agent status, model name, and whether demo mode is active.

---

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

---

### `POST /questions`
Generate tailored interview questions.

```json
{
  "profile": {
    "job_title": "Data Scientist",
    "experience_level": "entry",
    "skills": ["python", "ml"]
  },
  "num_questions": 10
}
```

---

### `POST /evaluate`
Evaluate a candidate's answer with structured AI feedback.

```json
{
  "question": "Tell me about a time you resolved a conflict.",
  "answer": "In my last role, I had a disagreement with..."
}
```

---

### `POST /plan`
Generate a 4-week preparation plan + improvement tips.

```json
{
  "profile": {
    "job_title": "Software Engineer",
    "experience_level": "senior"
  }
}
```

---

### `POST /full-session`
Combined endpoint — questions + plan + tips in one request.

---

## Running Tests

```bash
# All tests run in demo mode — no IBM credentials needed
pytest tests/ -v

# With short tracebacks
pytest tests/ -v --tb=short
```

---

## IBM Granite Integration

The agent uses **IBM Granite 13B Instruct** (`ibm/granite-13b-instruct-v2`) via the **ibm-watsonx-ai** SDK:

- **RAG-augmented prompts** — every generation call retrieves the top-5 relevant documents from ChromaDB to ground the model's response.
- **Instruction-tuned prompts** — prompts are structured for Granite's instruction-following format.
- **Demo mode** — if credentials are absent, the agent stubs responses clearly labelled `[DEMO MODE]`; the full architecture still runs.

### IBM Cloud Lite Usage

| Service | Usage |
|---|---|
| IBM watsonx.ai | Granite LLM inference (free tier: 50K tokens/month) |
| IBM Cloud IAM | API key management |

---

## Extending the Knowledge Base

Add entries to any JSON file under `data/knowledge_base/`. The RAG engine auto-reindexes on the next startup when the ChromaDB collection is empty.

To force a full rebuild, delete the `data/chroma_db/` directory:

```bash
rm -rf data/chroma_db/
python app.py   # rebuilds the index on startup
```

---

*Built for AICTE · IBM Cloud Lite · IBM Granite (watsonx.ai) · Problem Statement #22*
