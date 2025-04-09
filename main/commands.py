import typer
from pathlib import Path
from typing import Optional, List
import asyncio # Added for running async functions

from rich.prompt import Confirm

from . import config, task_manager, ui, utils, dependency_manager
from .utils import log

# Lazy import STORM only when needed
_storm_imported = False
def _ensure_storm_imported():
    global _storm_imported
    if not _storm_imported:
        _storm_imported = task_manager._import_storm_lazy() # Call the importer from task_manager
    return _storm_imported


app = typer.Typer(help="Task Researcher: Manage AI-driven development tasks with integrated STORM research.")

@app.callback()
def main_callback(
    ctx: typer.Context,
    tasks_file: Path = typer.Option(lambda: config.TASKS_FILE_PATH, "--tasks-file", "-f", help="Path to the tasks JSON file.", show_default=False),
    complexity_report_file: Path = typer.Option(lambda: config.COMPLEXITY_REPORT_PATH, "--report-file", help="Path to the complexity report JSON file.", show_default=False),
    task_files_dir: Path = typer.Option(lambda: config.TASK_FILES_DIR, "--tasks-dir", help="Directory for generated task files.", show_default=False),
    storm_output_dir: Path = typer.Option(lambda: config.STORM_OUTPUT_DIR, "--storm-output-dir", help="Directory for STORM research output.", show_default=False),
):
    """ Store shared options in the context. """
    ctx.ensure_object(dict)
    # Resolve paths relative to current working directory
    ctx.obj["TASKS_FILE"] = Path.cwd() / tasks_file
    ctx.obj["REPORT_FILE"] = Path.cwd() / complexity_report_file
    ctx.obj["TASKS_DIR"] = Path.cwd() / task_files_dir
    ctx.obj["STORM_DIR"] = Path.cwd() / storm_output_dir

    # Update config paths based on CLI options if they differ from defaults
    config.TASKS_FILE_PATH = ctx.obj["TASKS_FILE"]
    config.COMPLEXITY_REPORT_PATH = ctx.obj["REPORT_FILE"]
    config.TASK_FILES_DIR = ctx.obj["TASKS_DIR"]
    config.STORM_OUTPUT_DIR = ctx.obj["STORM_DIR"]

    # Initial API key check
    missing_keys = config.check_api_keys()
    if missing_keys:
         log.warning(f"Missing required API keys in .env: {', '.join(missing_keys)}")
         ui.console.print(f"[bold yellow]Warning:[/bold yellow] Missing API keys in .env: {', '.join(missing_keys)}")
         ui.console.print("[yellow]Some commands may not function correctly.[/yellow]")


# --- Core Commands ---

@app.command(name="parse-inputs")
def parse_inputs_cmd(
    ctx: typer.Context,
    num_tasks: int = typer.Option(15, "--num-tasks", "-n", help="Approximate number of tasks to generate."),
    func_spec: Path = typer.Option(config.FUNCTIONAL_SPEC_PATH, help="Path to functional spec file."),
    tech_spec: Path = typer.Option(config.TECHNICAL_SPEC_PATH, help="Path to technical spec file."),
    plan: Path = typer.Option(config.PLAN_PATH, help="Path to high-level plan file."),
    research_doc: Path = typer.Option(config.BACKGROUND_PATH, "--research-doc", help="Path to background file."), # Renamed option
):
    """Parses input spec files to generate initial tasks using AI."""
    tasks_file = ctx.obj["TASKS_FILE"]
    ui.display_banner()
    if tasks_file.exists():
        if not Confirm.ask(f"[yellow]Warning:[/yellow] {tasks_file} already exists. Overwrite with new tasks?", default=False):
            print("Operation cancelled.")
            raise typer.Exit()
    # Run async function using asyncio
    asyncio.run(task_manager.parse_inputs(num_tasks, tasks_file, func_spec, tech_spec, plan, research_doc))

@app.command()
def update(
    ctx: typer.Context,
    from_id: int = typer.Option(1, "--from", "-f", help="Task ID to start updating from (inclusive)."),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt describing the changes required."),
    research_hint: bool = typer.Option(False, "--research-hint", help="Hint AI to use its latest knowledge (no live search)."), # Renamed for clarity
):
    """Updates tasks from a specific ID based on new context or requirements."""
    tasks_file = ctx.obj["TASKS_FILE"]
    ui.display_banner()
    asyncio.run(task_manager.update_tasks(from_id, prompt, research_hint, tasks_file))

@app.command()
def generate(ctx: typer.Context):
    """Generates individual task files (.txt) from the main tasks JSON file."""
    tasks_file = ctx.obj["TASKS_FILE"]
    tasks_dir = ctx.obj["TASKS_DIR"]
    ui.display_banner()
    task_manager.generate_task_files(tasks_file, tasks_dir)

@app.command()
def expand(
    ctx: typer.Context,
    task_id: int = typer.Option(None, "--id", "-i", help="ID of the specific task to expand."),
    all_tasks: bool = typer.Option(False, "--all", "-a", help="Expand all eligible pending tasks."),
    num_subtasks: Optional[int] = typer.Option(None, "--num", "-n", help=f"Number of subtasks to generate (default: from report or {config.DEFAULT_SUBTASKS})."),
    research: bool = typer.Option(False, "--research", "-r", help="Perform STORM-based research before generating subtasks."), # Clarified help
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Additional context/prompt for subtask generation."),
    force: bool = typer.Option(False, "--force", help="Force expansion even if task already has subtasks (overwrites existing)."),
):
    """Expands tasks into subtasks. --research triggers multi-step STORM workflow."""
    tasks_file = ctx.obj["TASKS_FILE"]
    ui.display_banner()
    if task_id is not None and all_tasks:
        ui.console.print("[bold red]Error:[/bold red] Cannot use --id and --all together.")
        raise typer.Exit(code=1)
    if task_id is None and not all_tasks:
        ui.console.print("[bold red]Error:[/bold red] Must specify either --id or --all.")
        raise typer.Exit(code=1)

    if research and not _ensure_storm_imported():
        raise typer.Exit(code=1) # Exit if STORM import failed

    # Run async functions using asyncio
    if all_tasks:
        asyncio.run(task_manager.expand_all_tasks(num_subtasks, research, prompt, force, tasks_file))
    else:
        asyncio.run(task_manager.expand_task(task_id, num_subtasks, research, prompt, force, tasks_file))


@app.command(name="analyze-complexity")
def analyze_complexity_cmd(
    ctx: typer.Context,
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path for the report (default: from config)."),
    research_hint: bool = typer.Option(False, "--research-hint", help="Hint AI to use its latest knowledge for analysis."), # Renamed for clarity
    threshold: float = typer.Option(5.0, "--threshold", "-t", min=1.0, max=10.0, help="Complexity threshold (1-10) for highlighting."),
):
    """Analyzes complexity of all tasks and saves a report."""
    tasks_file = ctx.obj["TASKS_FILE"]
    report_file = output_file or ctx.obj["REPORT_FILE"] # Use command option if provided
    ui.display_banner()
    asyncio.run(task_manager.analyze_complexity(report_file, research_hint, threshold, tasks_file))

@app.command(name="complexity-report")
def complexity_report_cmd(ctx: typer.Context):
    """Displays the saved task complexity analysis report."""
    report_file = ctx.obj["REPORT_FILE"]
    ui.display_banner()
    ui.display_complexity_report(report_file)

@app.command(name="validate-deps")
def validate_deps_cmd(ctx: typer.Context):
    """Validates task dependencies for issues like missing refs or cycles."""
    tasks_file = ctx.obj["TASKS_FILE"]
    ui.display_banner()
    tasks_data_dict = utils.read_json(tasks_file)
    if not tasks_data_dict:
        ui.console.print(f"[bold red]Error:[/bold red] Cannot read tasks from {tasks_file}")
        raise typer.Exit(code=1)
    try:
        tasks_data = models.TasksData.model_validate(tasks_data_dict)
    except Exception as e:
        log.error(f"Invalid tasks file structure: {e}")
        ui.console.print(f"[bold red]Error:[/bold red] Invalid tasks file structure in {tasks_file}")
        raise typer.Exit(code=1)

    is_valid, issues = dependency_manager.validate_dependencies(tasks_data)

    if is_valid:
        ui.console.print(Panel("[bold green]All dependencies are valid.[/bold green]", border_style="green"))
    else:
        ui.console.print(Panel(f"[bold yellow]Found {len(issues)} dependency issues:[/bold yellow]", border_style="yellow"))
        table = Table("Type", "Task/Subtask ID", "Details", title="Dependency Issues")
        for issue in issues:
             details = f"Depends on missing: {issue['dep']}" if issue['type'] == 'missing' else \
                       f"Self-dependency" if issue['type'] == 'self' else \
                       f"Part of a cycle (reported at node: {issue['id']})" if issue['type'] == 'cycle' else 'Unknown'
             table.add_row(f"[red]{issue['type'].upper()}[/]", str(issue['id']), details)
        ui.console.print(table)
        ui.console.print("\nRun [cyan]task-researcher fix-deps[/] to attempt automatic fixes.") # Renamed command


@app.command(name="fix-deps")
def fix_deps_cmd(ctx: typer.Context):
    """Automatically fixes invalid dependencies (missing refs, self-deps, simple cycles)."""
    tasks_file = ctx.obj["TASKS_FILE"]
    ui.display_banner()
    tasks_data_dict = utils.read_json(tasks_file)
    if not tasks_data_dict:
        ui.console.print(f"[bold red]Error:[/bold red] Cannot read tasks from {tasks_file}")
        raise typer.Exit(code=1)
    try:
        tasks_data = models.TasksData.model_validate(tasks_data_dict)
    except Exception as e:
        log.error(f"Invalid tasks file structure: {e}")
        ui.console.print(f"[bold red]Error:[/bold red] Invalid tasks file structure in {tasks_file}")
        raise typer.Exit(code=1)

    ui.console.print("Attempting to fix dependencies...")
    made_changes, fixes_summary = dependency_manager.fix_dependencies(tasks_data)

    if made_changes:
        utils.write_json(tasks_file, tasks_data.model_dump(mode='json', exclude_none=True))
        ui.console.print(Panel(
            f"[bold green]Dependencies fixed successfully![/bold green]\n\n"
            f"Missing refs removed: {fixes_summary['missing']}\n"
            f"Self-deps removed: {fixes_summary['self']}\n"
            f"Cycles broken (simple): {fixes_summary['cycle']}\n\n"
            f"Tasks saved to {tasks_file}",
            border_style="green"
        ))
        # Regenerate files to reflect changes
        task_manager.generate_task_files(tasks_file, config.TASK_FILES_DIR)
    else:
        ui.console.print(Panel("[bold blue]No dependency issues found or no automatic fixes applied.[/bold blue]", border_style="blue"))


# --- STORM Command ---
@app.command(name="research-topic")
def research_topic_cmd(
    ctx: typer.Context,
    topic: str = typer.Argument(..., help="The topic to research using STORM."),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for the research report (Markdown). Defaults to '<storm_dir>/<topic_name>.md'."),
    search_top_k: int = typer.Option(config.STORM_SEARCH_TOP_K, "--top-k", "-k", help="Number of search results per query."),
    max_conv_tokens: int = typer.Option(config.STORM_MAX_TOKENS_CONV, "--max-conv-tokens", help="Max tokens for STORM conversation simulation."),
    max_article_tokens: int = typer.Option(config.STORM_MAX_TOKENS_ARTICLE, "--max-article-tokens", help="Max tokens for STORM final article generation."),
    retriever_type: str = typer.Option(config.STORM_RETRIEVER, "--retriever", help="STORM retriever (e.g., 'bing', 'you', 'tavily')."),
):
    """Generates a research report on a topic using the knowledge-storm library."""
    if not _ensure_storm_imported():
        raise typer.Exit(code=1)

    ui.display_banner()
    ui.console.print(f"Starting research on topic: [bold cyan]{topic}[/bold cyan] using STORM...")
    storm_output_dir = ctx.obj["STORM_DIR"] # Use resolved path from context

    # Determine output path
    output_path = output_file or (storm_output_dir / f"{utils.sanitize_filename(topic)}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists

    # --- Configure and Run STORM ---
    # Note: run_storm_research_for_topic expects topic name and list of questions.
    # For the direct command, we only have the topic name. We'll pass an empty list
    # or maybe generate a single high-level question. Let's pass just the topic for now.
    # We refactored the runner setup into task_manager._get_storm_runner
    # and the execution into task_manager.run_storm_research_for_topic
    # Call the synchronous run_storm function directly
    article_content = task_manager.run_storm_research_for_topic(
        topic_name=topic,
        questions=[f"Provide a comprehensive overview of {topic}."], # Generate a default question
        search_top_k=search_top_k,
        max_conv_tokens=max_conv_tokens,
        max_article_tokens=max_article_tokens
        # Retriever type is handled by config inside _get_storm_runner
    )

    if article_content:
        utils.write_file(output_path, article_content)
        ui.console.print(f"[bold green]Success![/bold green] STORM research complete. Report saved to {output_path}")
        # Display snippet
        ui.console.print(Panel(utils.truncate(article_content, 500) + "\n...", title="Report Snippet", border_style="cyan", expand=False))
    else:
        ui.console.print("[bold red]Error:[/bold red] STORM research failed or no content was generated.")


# --- Placeholder Commands (Not implemented in this phase) ---
@app.command(hidden=True) # Hide unimplemented commands
def list(ctx: typer.Context):
    """(Not Implemented) List tasks."""
    ui.console.print("[yellow]Command 'list' is not yet implemented.[/yellow]")

@app.command(hidden=True)
def next_task(ctx: typer.Context):
    """(Not Implemented) Show the next suggested task."""
    ui.console.print("[yellow]Command 'next' is not yet implemented.[/yellow]")

@app.command(hidden=True)
def show(ctx: typer.Context, task_id: str):
    """(Not Implemented) Show details for a specific task or subtask ID."""
    ui.console.print(f"[yellow]Command 'show {task_id}' is not yet implemented.[/yellow]")

@app.command(hidden=True)
def set_status(ctx: typer.Context, task_id: str, status: str):
    """(Not Implemented) Set the status of a task or subtask."""
    ui.console.print(f"[yellow]Command 'set-status {task_id} {status}' is not yet implemented.[/yellow]")