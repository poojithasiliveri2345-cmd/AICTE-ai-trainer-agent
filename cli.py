"""
Rich interactive CLI for the Interview Trainer Agent.
Provides a terminal-based interview prep experience with colored output,
progress indicators, and an interactive Q&A session.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from dotenv import load_dotenv

load_dotenv()

console = Console()

IBM_BANNER = """
[bold blue]╔══════════════════════════════════════════════════════════════╗
║      🎯  IBM GRANITE — AI Interview Trainer Agent  🎯          ║
║      Powered by IBM watsonx.ai + RAG (ChromaDB)               ║
╚══════════════════════════════════════════════════════════════╝[/bold blue]
"""

ROLES = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "DevOps Engineer",
    "Machine Learning Engineer",
    "Frontend Developer",
    "Backend Developer",
    "Full Stack Developer",
    "Cloud Engineer",
    "Other",
]

LEVELS = ["entry", "mid", "senior"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_banner():
    console.print(IBM_BANNER)


def print_section(title: str):
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    console.print()


def spinner(message: str):
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{message}...[/bold blue]"),
        transient=True,
    )


def _build_agent_with_status():
    from src import build_agent
    with spinner("Initializing IBM Granite + RAG engine"):
        with Progress(SpinnerColumn(), TextColumn("[bold blue]Loading...[/bold blue]"), transient=True) as prog:
            prog.add_task("init", total=None)
            generator, evaluator, planner, rag, parser, status = build_agent()
    return generator, evaluator, planner, rag, parser, status


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """IBM Granite AI Interview Trainer — Powered by RAG + watsonx.ai"""
    pass


# ---------------------------------------------------------------------------
# Command: interactive session (full guided flow)
# ---------------------------------------------------------------------------

@cli.command("start")
@click.option("--resume", "-r", type=click.Path(exists=True), help="Path to your resume (PDF, DOCX, or TXT)")
def start_session(resume):
    """Start an interactive interview prep session."""
    print_banner()

    if not resume:
        console.print("[yellow]No resume file provided — entering manual profile mode.[/yellow]\n")

    # ---- Collect profile ----
    print_section("👤 Candidate Profile")

    name = Prompt.ask("[bold]Your name[/bold]")

    if resume:
        from src import build_agent
        _, _, _, _, parser, _ = _build_agent_with_status()
        with spinner("Parsing resume"):
            with Progress(SpinnerColumn(), TextColumn("[bold blue]Parsing...[/bold blue]"), transient=True) as prog:
                prog.add_task("parse", total=None)
                profile = parser.parse_file(resume)
        profile.name = name or profile.name
        console.print(f"\n[green]✅ Resume parsed successfully![/green]")
        console.print(Panel(profile.summary(), title="Extracted Profile", border_style="green"))
        if not Confirm.ask("Does this look correct?", default=True):
            profile = _collect_manual_profile(name)
    else:
        profile = _collect_manual_profile(name)

    # ---- Initialize agent ----
    console.print()
    with Progress(SpinnerColumn(), TextColumn("[bold blue]Initializing IBM Granite + RAG...[/bold blue]"), transient=True) as prog:
        prog.add_task("init", total=None)
        generator, evaluator, planner, _, _, status = _build_agent_with_status()

    _show_demo_notice(generator.granite.is_demo_mode())
    console.print(f"[dim]RAG: {status}[/dim]\n")

    # ---- Main menu loop ----
    while True:
        print_section("📋 What would you like to do?")
        console.print("[1] Generate Interview Questions")
        console.print("[2] Practice Q&A (Answer a question and get feedback)")
        console.print("[3] Get My Preparation Plan")
        console.print("[4] Get Improvement Tips")
        console.print("[5] View My Profile")
        console.print("[6] Exit")
        console.print()

        choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "1":
            _run_question_generation(generator, profile)
        elif choice == "2":
            _run_practice_session(generator, evaluator, profile)
        elif choice == "3":
            _run_prep_plan(planner, profile)
        elif choice == "4":
            _run_tips(planner, profile)
        elif choice == "5":
            console.print(Panel(profile.summary(), title="Your Profile", border_style="blue"))
        elif choice == "6":
            console.print("\n[bold green]Good luck with your interview! 🚀[/bold green]\n")
            break


def _collect_manual_profile(name: str):
    from src import ResumeParser

    console.print("[bold]Select your target role:[/bold]")
    for i, r in enumerate(ROLES, 1):
        console.print(f"  [{i}] {r}")
    role_idx = int(Prompt.ask("Role number", default="1")) - 1
    role_idx = max(0, min(role_idx, len(ROLES) - 1))
    role = ROLES[role_idx]
    if role == "Other":
        role = Prompt.ask("Enter your role")

    level = Prompt.ask("Experience level", choices=LEVELS, default="entry")
    years = int(Prompt.ask("Years of experience", default="0"))
    skills_input = Prompt.ask("Key skills (comma-separated)", default="python, git")
    skills = [s.strip() for s in skills_input.split(",") if s.strip()]

    parser = ResumeParser()
    return parser.build_from_inputs(
        name=name,
        job_title=role,
        experience_level=level,
        years_experience=years,
        skills=skills,
    )


def _show_demo_notice(demo: bool):
    if demo:
        console.print(Panel(
            "[yellow]⚠️  DEMO MODE: IBM watsonx credentials not configured.\n"
            "Add WATSONX_API_KEY and WATSONX_PROJECT_ID to .env for live IBM Granite responses.\n"
            "Showing demo/stub responses for illustration.[/yellow]",
            title="Demo Mode Active",
            border_style="yellow",
        ))
    else:
        console.print("[green]✅ IBM Granite (watsonx.ai) — Connected[/green]")


def _run_question_generation(generator, profile):
    print_section("🎯 Interview Questions")
    num = int(Prompt.ask("How many questions to generate?", default="10"))

    with Progress(SpinnerColumn(), TextColumn("[bold blue]IBM Granite generating questions...[/bold blue]"), transient=True) as prog:
        prog.add_task("gen", total=None)
        result = generator.generate(profile, num_questions=num)

    questions = result["questions"]
    console.print(f"\n[bold green]Generated {len(questions)} tailored questions:[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("#", style="dim", width=4)
    table.add_column("Question", min_width=40, max_width=70, overflow="fold")
    table.add_column("Category", width=20)
    table.add_column("Difficulty", width=10)

    for i, q in enumerate(questions, 1):
        diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(q.get("difficulty", "medium"), "white")
        table.add_row(
            str(i),
            q["question"],
            q.get("category", "General"),
            f"[{diff_color}]{q.get('difficulty', 'medium')}[/{diff_color}]",
        )

    console.print(table)

    if result.get("demo_mode"):
        console.print("\n[dim yellow]^ Demo output. Connect IBM watsonx for real tailored questions.[/dim yellow]")


def _run_practice_session(generator, evaluator, profile):
    print_section("💬 Practice Q&A Session")

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Fetching a question for you...[/bold blue]"), transient=True) as prog:
        prog.add_task("q", total=None)
        result = generator.generate(profile, num_questions=1)

    questions = result["questions"]
    if not questions:
        console.print("[red]Could not generate a question. Try again.[/red]")
        return

    q = questions[0]
    console.print(Panel(
        f"[bold white]{q['question']}[/bold white]\n\n"
        f"[dim]Category: {q.get('category')} | Difficulty: {q.get('difficulty')}[/dim]",
        title="❓ Your Question",
        border_style="cyan",
    ))

    console.print("\n[bold]Type your answer below (press Enter twice when done):[/bold]")
    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        pass

    answer = "\n".join(lines).strip()
    if not answer:
        console.print("[yellow]No answer provided. Skipping evaluation.[/yellow]")
        return

    with Progress(SpinnerColumn(), TextColumn("[bold blue]IBM Granite evaluating your answer...[/bold blue]"), transient=True) as prog:
        prog.add_task("eval", total=None)
        feedback = evaluator.evaluate(q["question"], answer)

    console.print()
    console.print(Panel(
        Markdown(feedback["feedback_raw"]),
        title="📊 Feedback from IBM Granite",
        border_style="green",
    ))


def _run_prep_plan(planner, profile):
    print_section("📅 Your Preparation Plan")

    with Progress(SpinnerColumn(), TextColumn("[bold blue]IBM Granite crafting your plan...[/bold blue]"), transient=True) as prog:
        prog.add_task("plan", total=None)
        result = planner.create_plan(profile)

    console.print(Panel(
        Markdown(result["plan"]),
        title=f"📅 4-Week Plan for {profile.name or 'You'} — {profile.job_title}",
        border_style="blue",
    ))


def _run_tips(planner, profile):
    print_section("💡 Improvement Tips")

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Generating personalized tips...[/bold blue]"), transient=True) as prog:
        prog.add_task("tips", total=None)
        result = planner.get_improvement_tips(profile)

    console.print(Panel(
        Markdown(result["tips"]),
        title="💡 Your Personalized Tips",
        border_style="magenta",
    ))


# ---------------------------------------------------------------------------
# Command: quick questions (non-interactive)
# ---------------------------------------------------------------------------

@cli.command("questions")
@click.option("--role", "-r", default="Software Engineer", help="Target job role")
@click.option("--level", "-l", default="entry", type=click.Choice(LEVELS), help="Experience level")
@click.option("--skills", "-s", default="", help="Comma-separated skills")
@click.option("--num", "-n", default=10, help="Number of questions")
@click.option("--name", default="Candidate", help="Your name")
def quick_questions(role, level, skills, num, name):
    """Quickly generate interview questions without full session."""
    print_banner()
    from src import build_agent, ResumeParser

    parser = ResumeParser()
    profile = parser.build_from_inputs(
        name=name,
        job_title=role,
        experience_level=level,
        years_experience=0,
        skills=[s.strip() for s in skills.split(",") if s.strip()],
    )

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Initializing...[/bold blue]"), transient=True) as prog:
        prog.add_task("init", total=None)
        generator, _, _, _, _, status = build_agent()

    _show_demo_notice(generator.granite.is_demo_mode())

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Generating questions...[/bold blue]"), transient=True) as prog:
        prog.add_task("gen", total=None)
        result = generator.generate(profile, num_questions=num)

    questions = result["questions"]
    console.print(f"\n[bold cyan]Interview Questions — {role} ({level})[/bold cyan]\n")
    for i, q in enumerate(questions, 1):
        diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(q.get("difficulty", "medium"), "white")
        console.print(f"[bold]{i}.[/bold] {q['question']}")
        console.print(f"   [dim]Category: {q.get('category')} | Difficulty: [{diff_color}]{q.get('difficulty')}[/{diff_color}][/dim]\n")


# ---------------------------------------------------------------------------
# Command: evaluate an answer from CLI args
# ---------------------------------------------------------------------------

@cli.command("evaluate")
@click.option("--question", "-q", required=True, help="The interview question")
@click.option("--answer", "-a", required=True, help="Your answer to evaluate")
def evaluate_cmd(question, answer):
    """Evaluate a single interview answer."""
    print_banner()
    from src import build_agent

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Initializing...[/bold blue]"), transient=True) as prog:
        prog.add_task("init", total=None)
        _, evaluator, _, _, _, _ = build_agent()

    _show_demo_notice(evaluator.granite.is_demo_mode())

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Evaluating...[/bold blue]"), transient=True) as prog:
        prog.add_task("eval", total=None)
        result = evaluator.evaluate(question, answer)

    console.print(Panel(
        Markdown(result["feedback_raw"]),
        title="📊 Feedback from IBM Granite",
        border_style="green",
    ))


if __name__ == "__main__":
    cli()
