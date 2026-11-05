"""
Resume and profile parser.
Extracts structured information from PDF, DOCX, or plain text resumes,
and from manual profile inputs.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


# Optional heavy parsers — gracefully degraded if not installed
try:
    from PyPDF2 import PdfReader as _PdfReader
    PDF_AVAILABLE = True
except ImportError:
    _PdfReader = None  # type: ignore[assignment]
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DocxDocument = None  # type: ignore[assignment,misc]
    DOCX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    name: str = ""
    email: str = ""
    phone: str = ""
    job_title: str = ""
    experience_level: str = ""       # entry / mid / senior
    years_experience: int = 0
    skills: List[str] = field(default_factory=list)
    education: List[Dict[str, str]] = field(default_factory=list)
    work_history: List[Dict[str, str]] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    raw_text: str = ""
    source: str = "manual"           # manual / pdf / docx / text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        skills_str = ", ".join(self.skills[:10]) if self.skills else "Not specified"
        jobs = "; ".join(w.get("title", "") for w in self.work_history[:2]) if self.work_history else "Not specified"
        return (
            f"Name: {self.name or 'Not provided'}\n"
            f"Target Role: {self.job_title}\n"
            f"Experience Level: {self.experience_level} ({self.years_experience} years)\n"
            f"Key Skills: {skills_str}\n"
            f"Recent Roles: {jobs}"
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Common tech skill keywords for extraction
TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "kotlin", "swift",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring", "rails",
    "aws", "azure", "gcp", "ibm cloud", "docker", "kubernetes", "terraform", "ansible",
    "machine learning", "deep learning", "nlp", "data science", "tensorflow", "pytorch",
    "git", "ci/cd", "devops", "agile", "scrum", "rest", "graphql", "microservices",
    "linux", "bash", "powershell", "html", "css", "spark", "kafka", "airflow",
}

EXPERIENCE_KEYWORDS = {
    "entry": ["junior", "entry", "associate", "graduate", "intern", "fresher", "0-2"],
    "mid": ["mid", "intermediate", "software engineer", "developer", "analyst", "2-5", "3-5"],
    "senior": ["senior", "lead", "principal", "staff", "architect", "manager", "5+", "7+", "10+"],
}

SECTION_HEADERS = {
    "skills": re.compile(r"(skills|technologies|technical skills|expertise)", re.I),
    "experience": re.compile(r"(experience|work history|employment|professional background)", re.I),
    "education": re.compile(r"(education|academic|qualifications|degrees)", re.I),
    "certifications": re.compile(r"(certifications|licenses|certificates)", re.I),
}


class ResumeParser:

    def parse_file(self, file_path: str) -> UserProfile:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._extract_pdf(path)
        elif suffix in (".docx", ".doc"):
            text = self._extract_docx(path)
        elif suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        profile = self._parse_text(text)
        profile.source = suffix.lstrip(".")
        return profile

    def parse_text(self, text: str) -> UserProfile:
        profile = self._parse_text(text)
        profile.source = "text"
        return profile

    def build_from_inputs(
        self,
        name: str,
        job_title: str,
        experience_level: str,
        years_experience: int,
        skills: Optional[List[str]] = None,
    ) -> UserProfile:
        """Build a profile directly from structured user inputs (no resume file needed)."""
        profile = UserProfile(
            name=name,
            job_title=job_title,
            experience_level=self._normalize_level(experience_level),
            years_experience=years_experience,
            skills=skills or [],
            source="manual",
        )
        return profile

    # -----------------------------------------------------------------------
    # Text extraction
    # -----------------------------------------------------------------------
    def _extract_pdf(self, path: Path) -> str:
        if not PDF_AVAILABLE or _PdfReader is None:
            raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
        text_parts = []
        with open(path, "rb") as f:
            reader = _PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    def _extract_docx(self, path: Path) -> str:
        if not DOCX_AVAILABLE or DocxDocument is None:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    # -----------------------------------------------------------------------
    # Text parsing
    # -----------------------------------------------------------------------
    def _parse_text(self, text: str) -> UserProfile:
        profile = UserProfile(raw_text=text)
        profile.email = self._extract_email(text)
        profile.phone = self._extract_phone(text)
        profile.name = self._extract_name(text)
        profile.skills = self._extract_skills(text)
        profile.years_experience = self._extract_years(text)
        profile.experience_level = self._infer_level(text, profile.years_experience)
        profile.job_title = self._infer_job_title(text)
        profile.education = self._extract_education(text)
        profile.work_history = self._extract_work_history(text)
        profile.certifications = self._extract_certifications(text)
        return profile

    def _extract_email(self, text: str) -> str:
        match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
        return match.group() if match else ""

    def _extract_phone(self, text: str) -> str:
        match = re.search(r"(\+?[\d\s\-().]{10,17})", text)
        return match.group().strip() if match else ""

    def _extract_name(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Heuristic: first non-empty line that looks like a name (no @, no digit run)
        for line in lines[:5]:
            if not re.search(r"[@\d{3}]", line) and 1 < len(line.split()) <= 4 and len(line) < 60:
                return line
        return ""

    def _extract_skills(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = [skill for skill in TECH_SKILLS if skill in text_lower]
        return sorted(set(found))

    def _extract_years(self, text: str) -> int:
        patterns = [
            r"(\d+)\+?\s*years?\s+of\s+experience",
            r"(\d+)\+?\s*years?\s+experience",
            r"experience\s+of\s+(\d+)\+?\s*years?",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                return int(m.group(1))
        return 0

    def _infer_level(self, text: str, years: int) -> str:
        # Years-based inference takes priority when explicitly stated
        if years >= 7:
            return "senior"
        if years >= 3:
            return "mid"
        # Fall back to keyword scanning
        text_lower = text.lower()
        for level, keywords in EXPERIENCE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return level
        return "entry"

    def _infer_job_title(self, text: str) -> str:
        common_titles = [
            "software engineer", "data scientist", "product manager",
            "devops engineer", "machine learning engineer", "frontend developer",
            "backend developer", "full stack developer", "cloud engineer",
            "data engineer", "site reliability engineer", "qa engineer",
        ]
        text_lower = text.lower()
        for title in common_titles:
            if title in text_lower:
                return title.title()
        return "Software Engineer"  # sensible default

    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        degrees = re.findall(
            r"(B\.?Tech|B\.?E\.?|B\.?Sc|M\.?Tech|M\.?Sc|MBA|PhD|Bachelor|Master|Doctor)[^\n]{0,80}",
            text, re.I
        )
        return [{"degree": d.strip()} for d in degrees[:3]]

    def _extract_work_history(self, text: str) -> List[Dict[str, str]]:
        # Extract lines that look like job titles with dates
        entries = re.findall(
            r"([A-Z][a-zA-Z\s]+(?:Engineer|Developer|Manager|Analyst|Scientist|Lead|Intern))[^\n]{0,60}",
            text
        )
        return [{"title": e.strip()} for e in entries[:4]]

    def _extract_certifications(self, text: str) -> List[str]:
        cert_patterns = [
            r"(AWS\s+Certified[^\n]{0,50})",
            r"(Google\s+Cloud[^\n]{0,50})",
            r"(Microsoft\s+Certified[^\n]{0,50})",
            r"(IBM\s+Certified[^\n]{0,50})",
            r"(CKA|CKAD|CKS|PMP|CISSP|CEH|CISA)[^\n]{0,40}",
        ]
        certs = []
        for pat in cert_patterns:
            certs.extend(re.findall(pat, text, re.I))
        return [c.strip() for c in certs[:5]]

    def _normalize_level(self, level: str) -> str:
        level_lower = level.lower()
        if any(kw in level_lower for kw in ["senior", "lead", "principal", "staff"]):
            return "senior"
        if any(kw in level_lower for kw in ["mid", "intermediate"]):
            return "mid"
        return "entry"
