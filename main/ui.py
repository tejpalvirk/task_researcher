from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from typing import List, Optional, Dict, Any, Union
import datetime

from . import models, config # Use models and config
from .utils import truncate # Use utils

console = Console()

# --- Color Mapping ---
STATUS_COLORS = {
    "pending": "yellow",
    "in-progress": "blue",
    "done": "green",
    "deferred": "grey50",
    "blocked": "red",
}
PRIORITY_COLORS = {
    "high": "red",
    "medium": "yellow",
    "low": "green",
}
COMPLEXITY_COLORS = {
    (1, 3.9): "green",    # Low
    (4, 7.9): "yellow",   # Medium
    (8, 10): "red",     # High
}

# --- Formatting Functions ---
def get_status_style(status: Optional[str]) -> str:
    """Gets the Rich style string for a status."""
    status_val = status or "pending"
    return STATUS_COLORS.get(status_val.lower(), "white")

def format_status(status: Optional[str]) -> Text:
    """Formats status with color."""
    status_val = status or "pending"
    style = get_status_style(status_val)
    # Improve display of 'in-progress'
    display_text = status_val.replace("-", " ").title()
    return Text(display_text, style=style)

def get_priority_style(priority: Optional[str]) -> str:
    """Gets the Rich style string for a priority."""
    priority_val = priority or config.DEFAULT_PRIORITY
    return PRIORITY_COLORS.get(priority_val.lower(), "white")

def format_priority(priority: Optional[str]) -> Text:
    """Formats priority with color."""
    priority_val = priority or config.DEFAULT_PRIORITY
    style = get_priority_style(priority_val)
    return Text(priority_val.capitalize(), style=style)

def get_complexity_style(score: Optional[float]) -> str:
    """Gets the Rich style string for a complexity score."""
    if score is None:
        return "grey50"
    # Iterate through ranges defined in COMPLEXITY_COLORS
    for (low, high), color in COMPLEXITY_COLORS.items():
        # Ensure score is compared correctly within float ranges
        # Using >= low and <= high is safer for float comparisons
        if low <= score <= high:
            return color
    return "white" # Default if score is outside defined ranges (e.g., < 1)

def format_complexity(score: Optional[float]) -> Text:
    """Formats complexity score with color."""
    if score is None:
        return Text("N/A", style="grey50")
    style = get_complexity_style(score)
    # Format to one decimal place
    return Text(f"{score:.1f}/10", style=style)


def format_dependencies(
    dependencies: List[Union[int, str]],
    all_tasks: List[models.Task], # Expect list of Task models
    simple: bool = False # Return plain string if True
) -> Union[str, Text]:
    """Formats dependencies, optionally showing status markers."""
    if not dependencies:
        return Text("None", style="grey50") if not simple else "None"

    formatted_deps = []
    # Need find_task_by_id that works with models
    from .utils import find_task_by_id as find_task_dict # Use dict version for now
    all_tasks_dicts = [t.model_dump() for t in all_tasks] # Convert models to dicts for lookup

    for dep_id in dependencies:
        # find_task_dict handles int/str and subtask "p.s" format
        dep_task_dict = find_task_dict(all_tasks_dicts, dep_id)
        dep_id_str = str(dep_id)

        if dep_task_dict:
            status = dep_task_dict.get('status', 'pending')
            is_done = status.lower() == 'done'
            status_marker = "✅" if is_done else "⏳"
            style = "green" if is_done else "yellow"
            if simple:
                formatted_deps.append(f"{status_marker}{dep_id_str}")
            else:
                formatted_deps.append(Text(f"{status_marker}{dep_id_str}", style=style))
        else:
            # Dependency not found
             if simple:
                  formatted_deps.append(f"❓{dep_id_str}")
             else:
                  formatted_deps.append(Text(f"❓{dep_id_str}", style="red"))

    if simple:
        return ", ".join(formatted_deps)
    else:
        text_result = Text("")
        for i, dep_text in enumerate(formatted_deps):
             text_result.append(dep_text)
             if i < len(formatted_deps) - 1:
                  text_result.append(", ")
        return text_result

# --- Display Functions ---
def display_banner():
    """Displays the Task Researcher banner."""
    console.print(Panel(
        Text("Task Researcher", style="bold blue", justify="center"),
        title="[bold cyan]Welcome[/]",
        border_style="cyan"
    ))

def display_tasks_summary(tasks: List[models.Task]): # Use Task model
    """Displays a summary table of tasks."""
    if not tasks:
        return
    table = Table(title="Tasks Overview", show_header=True, header_style="bold magenta", expand=True)
    table.add_column("ID", style="dim", width=7, no_wrap=True)
    table.add_column("Title", min_width=30, ratio=3)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Priority", justify="center", width=10)
    table.add_column("# Sub", justify="right", width=7)
    table.add_column("# Deps", justify="right", width=7)

    for task in tasks:
        # Ensure subtasks and dependencies are lists before checking length
        subtask_count = len(task.subtasks) if task.subtasks else 0
        dep_count = len(task.dependencies) if task.dependencies else 0
        table.add_row(
            str(task.id),
            task.title, # Rich handles wrapping
            format_status(task.status),
            format_priority(task.priority),
            str(subtask_count),
            str(dep_count),
        )
    console.print(table)

def display_subtasks_summary(subtasks: List[models.Subtask], parent_id: int): # Use Subtask model
    """Displays a summary table of subtasks."""
    if not subtasks:
        return
    table = Table(title=f"Generated Subtasks for Task {parent_id}", show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Sub ID", style="dim", width=10, no_wrap=True)
    table.add_column("Title", min_width=40, ratio=3)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Deps (Sibling ID)", justify="center", width=20) # Dependencies

    for subtask in subtasks:
         # Display only sibling dependencies (ints)
         sibling_deps_str = ", ".join(map(str, [d for d in subtask.dependencies if isinstance(d, int)])) if subtask.dependencies else "None"
         table.add_row(
             f"{parent_id}.{subtask.id}",
             subtask.title, # Rich handles wrapping
             format_status(subtask.status),
             sibling_deps_str
         )
    console.print(table)

def display_task_details(task: Union[models.Task, models.Subtask], all_tasks: List[models.Task]): # Use models
    """Displays detailed information about a single task or subtask."""
    is_subtask = isinstance(task, models.Subtask) # Check type directly
    # Get parent info if it's a subtask (assuming it's added by find_task_by_id)
    parent_info = getattr(task, 'parentTaskInfo', None) if is_subtask else None
    parent_id = parent_info['id'] if parent_info else None
    task_id_str = f"{parent_id}.{task.id}" if is_subtask and parent_id else str(task.id)
    title = f"Details: Task #{task_id_str} - {task.title}"

    panel_content = Text()
    panel_content.append("Status: ").append(format_status(task.status)).append("\n")
    if not is_subtask:
         # Task specific fields
         panel_content.append("Priority: ").append(format_priority(task.priority)).append("\n")
         deps = format_dependencies(task.dependencies, all_tasks, simple=False)
         panel_content.append("Dependencies: ").append(deps).append("\n")
    else:
         # Subtask specific fields
         # Display only sibling dependencies (ints)
         sibling_deps = [d for d in task.dependencies if isinstance(d, int)]
         deps_text = ", ".join(map(str, sibling_deps)) if sibling_deps else "None"
         panel_content.append(f"Sibling Dependencies: {deps_text}\n")
         if parent_info:
             panel_content.append(f"Parent Task: #{parent_info['id']} ({parent_info['title']})\n", style="italic cyan")


    panel_content.append("\n[bold]Description:[/bold]\n", style="white")
    panel_content.append(task.description or "N/A").append("\n")

    panel_content.append("\n[bold]Implementation Details:[/bold]\n", style="white")
    panel_content.append(task.details or "N/A").append("\n")

    if not is_subtask:
        panel_content.append("\n[bold]Test Strategy:[/bold]\n", style="white")
        panel_content.append(task.testStrategy or "N/A").append("\n")

    # Display Subtasks if it's a parent task and has subtasks
    if not is_subtask and task.subtasks:
        panel_content.append("\n[bold]Subtasks:[/bold]\n", style="white")
        sub_table = Table(show_header=True, box=None, padding=(0, 1), expand=False)
        sub_table.add_column("ID", style="dim", no_wrap=True)
        sub_table.add_column("Title", ratio=3)
        sub_table.add_column("Status", justify="center")
        sub_table.add_column("Deps (Sibling)", justify="center") # Sibling Deps
        for sub in task.subtasks:
            sibling_deps_text = ", ".join(map(str, [d for d in sub.dependencies if isinstance(d, int)])) if sub.dependencies else "None"
            sub_table.add_row(
                 f"{task.id}.{sub.id}",
                 sub.title,
                 format_status(sub.status),
                 Text(sibling_deps_text, style="cyan")
            )
        panel_content.append(sub_table)

    console.print(Panel(panel_content, title=title, border_style="blue", expand=True))

def format_task_for_file(task: models.Task, all_tasks: List[models.Task]) -> str: # Use Task model
    """Formats a task object into a string suitable for a .txt file."""
    content = f"# Task ID: {task.id}\n"
    content += f"# Title: {task.title}\n"
    content += f"# Status: {task.status or 'pending'}\n"
    content += f"# Priority: {task.priority or config.DEFAULT_PRIORITY}\n"

    # Use simple=True for plain text file
    deps_str = format_dependencies(task.dependencies, all_tasks, simple=True)
    content += f"# Dependencies: {deps_str}\n"
    content += "------------------------------------\n\n"

    content += f"## Description\n{task.description or 'N/A'}\n\n"
    content += f"## Implementation Details\n{task.details or 'N/A'}\n\n"
    content += f"## Test Strategy\n{task.testStrategy or 'N/A'}\n\n"

    if task.subtasks:
        content += "## Subtasks\n"
        content += "------------------------------------\n"
        for sub in task.subtasks:
            content += f"### Subtask {task.id}.{sub.id}: {sub.title}\n"
            content += f"- Status: {sub.status or 'pending'}\n"
            # Use simple=True for plain text file
            sub_deps_str = format_dependencies(sub.dependencies, all_tasks, simple=True)
            content += f"- Dependencies: {sub_deps_str}\n"
            content += f"\n#### Description\n{sub.description or 'N/A'}\n"
            content += f"\n#### Details\n{sub.details or 'N/A'}\n\n"

    return content

def display_complexity_report(report_path: Path):
    """Displays the complexity analysis report in a formatted way."""
    report_data = utils.read_json(report_path)
    if not report_data:
        console.print(f"[bold red]Error:[/bold red] Complexity report not found at {report_path}")
        return
    try:
        report = models.ComplexityReport.model_validate(report_data) # Use model
    except Exception as e:
        log.error(f"Invalid complexity report structure: {e}")
        console.print(f"[bold red]Error:[/bold red] Invalid complexity report file structure.")
        return

    meta = report.meta
    console.print(Panel(f"Complexity Analysis Report ({meta.get('generatedAt', 'N/A')})", style="bold blue", expand=True))
    console.print(f"Project: {meta.get('projectName', 'N/A')} | Version: {meta.get('projectVersion', 'N/A')}")
    console.print(f"Tasks Analyzed: {meta.get('tasksAnalyzed', 'N/A')} | Threshold: {meta.get('thresholdScore', 'N/A')}")
    console.print(f"Research Hint Used: {'Yes' if meta.get('usedResearchHint') else 'No'} | LLM: {meta.get('llm_model', 'N/A')}")

    table = Table(title="Task Complexity Analysis", show_header=True, header_style="bold magenta", expand=True)
    table.add_column("ID", style="dim", width=7, no_wrap=True)
    table.add_column("Title", min_width=30, ratio=2)
    table.add_column("Complexity", justify="center", width=12)
    table.add_column("Rec. Sub", justify="center", width=10) # Shorter header
    table.add_column("Reasoning / Expand Prompt", ratio=3) # Combine reasoning/prompt

    sorted_analysis = sorted(report.complexityAnalysis, key=lambda x: x.complexityScore, reverse=True)

    for item in sorted_analysis:
        complexity_text = format_complexity(item.complexityScore)
        reasoning_prompt = Text()
        reasoning_prompt.append(item.reasoning or "N/A", style="italic")
        reasoning_prompt.append("\nExpand Prompt: ", style="dim")
        reasoning_prompt.append(truncate(item.expansionPrompt, 150))

        # Highlight rows above threshold
        row_style = "on grey19" if item.complexityScore >= meta.get('thresholdScore', 5.0) else ""

        table.add_row(
            str(item.taskId),
            item.taskTitle, # Rich handles wrapping
            complexity_text,
            str(item.recommendedSubtasks),
            reasoning_prompt,
            style=row_style
        )

    console.print(table)

    # Summary Stats
    scores = [item.complexityScore for item in report.complexityAnalysis]
    if scores:
         avg_score = sum(scores) / len(scores)
         high_count = sum(1 for s in scores if s >= 8)
         med_count = sum(1 for s in scores if 4 <= s < 8)
         low_count = sum(1 for s in scores if s < 4)
         console.print("\n[bold]Summary Stats:[/bold]")
         console.print(f"  Average Complexity: {avg_score:.1f}/10")
         console.print(f"  Distribution: [red]High ({high_count})[/], [yellow]Medium ({med_count})[/], [green]Low ({low_count})[/]")


def display_complexity_summary(report: models.ComplexityReport): # Use model
     """Prints a brief summary after generating the report."""
     scores = [item.complexityScore for item in report.complexityAnalysis]
     if not scores:
         console.print("[yellow]No complexity scores generated.[/yellow]")
         return

     avg_score = sum(scores) / len(scores)
     high_count = sum(1 for s in scores if s >= 8)
     med_count = sum(1 for s in scores if 4 <= s < 8)
     low_count = sum(1 for s in scores if s < 4)
     threshold = report.meta.get('thresholdScore', 5.0)
     recommended_for_expansion = sum(1 for item in report.complexityAnalysis if item.complexityScore >= threshold)

     console.print(Panel(
         f"Analyzed [bold]{len(scores)}[/] tasks. Average Complexity: [bold]{avg_score:.1f}/10[/].\n"
         f"Distribution: [red]High ({high_count})[/] | [yellow]Medium ({med_count})[/] | [green]Low ({low_count})[/].\n"
         f"[bold]{recommended_for_expansion}[/] tasks recommended for expansion (score >= {threshold}).",
         title="Complexity Summary",
         border_style="green",
         expand=False
     ))
     report_file_arg = f"--report-file={config.COMPLEXITY_REPORT_PATH}" if config.COMPLEXITY_REPORT_PATH != Path("scripts/task-complexity-report.json") else ""
     console.print(f"View full details: [cyan]task-researcher complexity-report {report_file_arg}[/]")


# --- STORM UI ---
def display_storm_progress(message: str):
    """Displays progress messages specific to STORM operations."""
    console.print(f"🌪️ [cyan]STORM:[/cyan] {message}")