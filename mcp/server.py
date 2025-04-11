import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import MCP components
from mcp.server.fastmcp import FastMCP, Context # Use FastMCP for easier setup

# Import shared configuration and utilities
from .. import config, utils, models, ui

# Import core logic modules
from .. import task_manager
from .. import dependency_manager

# Initialize FastMCP server
mcp_server = FastMCP(
    config.PROJECT_NAME,
    version=config.PROJECT_VERSION,
    dependencies=[
        "task-researcher", # Match package name
        "typer", "instructor", "litellm", "anthropic",
        "google-generativeai", "knowledge-storm", "rich",
        "python-dotenv", "pydantic", "aiohttp", "requests"
    ]
)

# --- MCP Resources ---

@mcp_server.resource("tasks://current")
async def get_current_tasks(ctx: Context) -> Dict[str, Any]:
    """Provides the current content of the main tasks JSON file."""
    # Reading small JSON is likely fast enough to be sync within async handler
    tasks_data_dict = utils.read_json(config.TASKS_FILE_PATH)
    if tasks_data_dict:
        try:
            validated_data = models.TasksData.model_validate(tasks_data_dict)
            return validated_data.model_dump(mode='json', exclude_none=True)
        except Exception as e:
             utils.log.warning(f"tasks.json at {config.TASKS_FILE_PATH} is invalid: {e}. Returning empty.")
             return models.TasksData().model_dump(mode='json', exclude_none=True)
    else:
        # Return default empty structure if file doesn't exist or is invalid
        return models.TasksData().model_dump(mode='json', exclude_none=True)


@mcp_server.resource("report://complexity")
async def get_complexity_report(ctx: Context) -> Optional[Dict[str, Any]]:
    """Provides the content of the task complexity report, if it exists."""
    report_data_dict = utils.read_complexity_report(config.COMPLEXITY_REPORT_PATH)
    if report_data_dict:
         try:
             validated_report = models.ComplexityReport.model_validate(report_data_dict)
             return validated_report.model_dump(mode='json', exclude_none=True)
         except Exception as e:
              utils.log.warning(f"Complexity report is invalid: {e}")
              return None
    return None

@mcp_server.resource("research://{topic_name}")
async def get_storm_research(ctx: Context, topic_name: str) -> Optional[str]:
    """Provides the content of a previously generated STORM research report."""
    safe_filename = utils.sanitize_filename(topic_name) + ".md"
    # Check primary location first
    report_path = config.STORM_OUTPUT_DIR / safe_filename
    content: Optional[str] = None

    if report_path.is_file(): # Check if it's actually a file
        content = await asyncio.to_thread(utils.read_file, report_path)
    else:
        # Fallback: check directory for files starting similarly
        try:
             # Run glob in thread as it can touch the filesystem
             md_files = await asyncio.to_thread(
                 lambda: list(config.STORM_OUTPUT_DIR.glob(f"{utils.sanitize_filename(topic_name)}*.md"))
             )
             if md_files:
                 # Get stat info and find latest file in thread
                 def find_latest(files):
                      return max(files, key=lambda p: p.stat().st_mtime) if files else None
                 latest_md = await asyncio.to_thread(find_latest, md_files)

                 if latest_md:
                      utils.log.info(f"Found potential STORM output via glob for topic '{topic_name}': {latest_md}")
                      content = await asyncio.to_thread(utils.read_file, latest_md)
                 else:
                      utils.log.warning(f"Glob found files, but couldn't determine latest for topic '{topic_name}'")
             else:
                 utils.log.warning(f"Research report not found for topic '{topic_name}' using primary path or glob.")
        except Exception as e:
            utils.log.error(f"Error accessing STORM directory for topic '{topic_name}': {e}")

    return content

@mcp_server.resource("taskfile://{task_id_str}") # Renamed arg for clarity
async def get_task_file(ctx: Context, task_id_str: str) -> Optional[str]:
    """
    Provides the content of a specific generated task file (phase_XX_task_YYY.txt).
    Accepts the task ID as a string (e.g., '5').
    """
    file_path: Optional[Path] = None
    try:
        task_id = int(task_id_str) # Expecting just the ID number as a string
        # Find the corresponding task data to get phase
        tasks_data = utils.read_json(config.TASKS_FILE_PATH)
        task_dict = utils.find_task_by_id(tasks_data.get('tasks', []) if tasks_data else [], task_id)

        if task_dict:
            phase_prefix = utils.format_phase_for_filename(task_dict.get('phase'))
            filename = f"phase_{phase_prefix}_task_{task_id:03d}.txt"
            file_path = config.TASK_FILES_DIR / filename
            utils.log.info(f"Attempting to read task file: {file_path}")
        else:
             utils.log.warning(f"Task ID {task_id} not found in tasks data.")

    except ValueError:
        utils.log.error(f"Invalid task_id '{task_id_str}' provided. Must be an integer.")
        return None # Return None for invalid ID format
    except Exception as e:
        utils.log.exception(f"Error resolving task file path for ID {task_id_str}")
        return None

    content: Optional[str] = None
    if file_path and file_path.exists():
        content = await asyncio.to_thread(utils.read_file, file_path)
    else:
        utils.log.warning(f"Task file not found at expected path: {file_path}")
        # Optional: Could add glob fallback here if really needed

    return content


# --- MCP Tools --- (Ensure they call async task_manager functions)

@mcp_server.tool()
async def parse_inputs(
    ctx: Context,
    num_tasks: int = 15,
    func_spec_path: Optional[str] = None,
    tech_spec_path: Optional[str] = None,
    plan_path: Optional[str] = None,
    research_doc_path: Optional[str] = None
    ) -> str:
    """Parses input specs to generate initial tasks (incl. phase) and overwrites the tasks file."""
    utils.log.info(f"MCP Tool: Running parse_inputs (num_tasks={num_tasks})")
    try:
        f_path = Path(func_spec_path) if func_spec_path else config.FUNCTIONAL_SPEC_PATH
        t_path = Path(tech_spec_path) if tech_spec_path else config.TECHNICAL_SPEC_PATH
        p_path = Path(plan_path) if plan_path else config.PLAN_PATH
        r_path = Path(research_doc_path) if research_doc_path else config.BACKGROUND_PATH

        await task_manager.parse_inputs(
            num_tasks=num_tasks, tasks_file_path=config.TASKS_FILE_PATH,
            func_spec_path=f_path, tech_spec_path=t_path, plan_path=p_path, research_path=r_path,
        )
        # Read back result count (sync ok after await)
        tasks_data = utils.read_json(config.TASKS_FILE_PATH)
        count = len(tasks_data.get('tasks', [])) if tasks_data else 0
        return f"Successfully parsed inputs and generated {count} tasks in {config.TASKS_FILE_PATH}."
    except Exception as e:
        utils.log.exception("MCP Tool: parse_inputs failed")
        return f"Error parsing inputs: {e}"

@mcp_server.tool()
async def update_tasks(
    ctx: Context,
    from_id: int,
    prompt: str,
    research_hint: bool = False
    ) -> str:
    """Updates tasks (title, desc, details, phase, test strat) from ID based on prompt."""
    utils.log.info(f"MCP Tool: Running update_tasks (from_id={from_id}, research_hint={research_hint})")
    try:
        await task_manager.update_tasks(
            from_id=from_id, prompt=prompt, use_research=research_hint,
            tasks_file_path=config.TASKS_FILE_PATH
        )
        return f"Update process completed for tasks starting from ID {from_id}. Check tasks file."
    except Exception as e:
        utils.log.exception("MCP Tool: update_tasks failed")
        return f"Error updating tasks: {e}"

@mcp_server.tool()
async def generate_task_files(ctx: Context) -> str:
    """Generates individual task files (phase_X_task_Y.txt)."""
    utils.log.info(f"MCP Tool: Running generate_task_files")
    try:
        # Run the synchronous function in a thread
        await asyncio.to_thread(
            task_manager.generate_task_files,
            config.TASKS_FILE_PATH,
            config.TASK_FILES_DIR
        )
        return f"Successfully generated task files in {config.TASK_FILES_DIR}."
    except Exception as e:
        utils.log.exception("MCP Tool: generate_task_files failed")
        return f"Error generating task files: {e}"

@mcp_server.tool()
async def expand_task(
    ctx: Context,
    task_id: int,
    num_subtasks: Optional[int] = None,
    research: bool = False,
    prompt: Optional[str] = None,
    force: bool = False
    ) -> str:
    """Expands a task into subtasks (with AC). Use research=True for STORM workflow (requires confirmation via logs/UI if run manually)."""
    utils.log.info(f"MCP Tool: Running expand_task (id={task_id}, research={research}, force={force})")
    if research and not task_manager._ensure_storm_imported():
        return "Error: knowledge-storm package required for --research but not found or failed to import."
    try:
        # NOTE: Confirmation step in task_manager.expand_task relies on interactive console.
        # For MCP, it will proceed without confirmation if research=True.
        # Consider adding a separate MCP tool or flag to *only* generate the research plan first.
        await task_manager.expand_task(
            task_id=task_id, num_subtasks=num_subtasks, use_research=research,
            prompt=prompt, force=force, tasks_file_path=config.TASKS_FILE_PATH
        )
        # Read back subtask count (sync ok after await)
        tasks_data = utils.read_json(config.TASKS_FILE_PATH)
        task_dict = utils.find_task_by_id(tasks_data.get('tasks',[]), task_id) if tasks_data else None
        sub_count = len(task_dict.get('subtasks',[])) if task_dict else 0
        research_msg = " with STORM research" if research else ""
        return f"Successfully expanded task {task_id}{research_msg}. It now has {sub_count} subtasks."
    except Exception as e:
        utils.log.exception(f"MCP Tool: expand_task id={task_id} failed")
        return f"Error expanding task {task_id}: {e}"

@mcp_server.tool()
async def expand_all_tasks(
    ctx: Context,
    num_subtasks: Optional[int] = None,
    research: bool = False,
    prompt: Optional[str] = None,
    force: bool = False
    ) -> str:
    """Expands all eligible pending tasks into subtasks (with AC). Use research=True for STORM workflow (no confirmation via MCP)."""
    utils.log.info(f"MCP Tool: Running expand_all_tasks (research={research}, force={force})")
    if research and not task_manager._ensure_storm_imported():
        return "Error: knowledge-storm package required for --research but not found or failed to import."
    try:
        # NOTE: Confirmation step is bypassed when called via MCP.
        await task_manager.expand_all_tasks(
            num_subtasks=num_subtasks, use_research=research, prompt=prompt,
            force=force, tasks_file_path=config.TASKS_FILE_PATH
        )
        research_msg = " with STORM research" if research else ""
        return f"Bulk expansion process{research_msg} completed. Check logs and tasks file for details."
    except Exception as e:
        utils.log.exception(f"MCP Tool: expand_all_tasks failed")
        return f"Error expanding all tasks: {e}"

@mcp_server.tool()
async def analyze_complexity(
    ctx: Context,
    research_hint: bool = False,
    threshold: float = 5.0,
    output_file: Optional[str] = None
    ) -> str:
    """Analyzes complexity of all tasks using AI and saves a report."""
    utils.log.info(f"MCP Tool: Running analyze_complexity (research_hint={research_hint})")
    try:
        report_path = Path(output_file) if output_file else config.COMPLEXITY_REPORT_PATH
        await task_manager.analyze_complexity(
            output_path=report_path, use_research=research_hint, threshold=threshold,
            tasks_file_path=config.TASKS_FILE_PATH
        )
        if report_path.exists():
             abs_path = str(report_path.resolve())
             return f"Complexity analysis complete. Report saved to: {abs_path}. Use 'get_complexity_report' resource to read."
        else:
             return "Complexity analysis ran, but report file was not found."
    except Exception as e:
        utils.log.exception("MCP Tool: analyze_complexity failed")
        return f"Error analyzing complexity: {e}"

@mcp_server.tool()
async def validate_dependencies(ctx: Context) -> str:
    """Validates task dependencies for issues like missing refs or cycles."""
    utils.log.info("MCP Tool: Running validate_dependencies")
    try:
        tasks_data_dict = utils.read_json(config.TASKS_FILE_PATH)
        if not tasks_data_dict: return "Error: Cannot read tasks file."
        tasks_data = models.TasksData.model_validate(tasks_data_dict)

        is_valid, issues = await asyncio.to_thread(
            dependency_manager.validate_dependencies,
            tasks_data
        )

        if is_valid:
            return "All dependencies are valid."
        else:
            issue_summary = [f"- {issue['type'].upper()} on {issue['id']}" + (f" -> {issue['dep']}" if 'dep' in issue else "") for issue in issues]
            return f"Found {len(issues)} dependency issues:\n" + "\n".join(issue_summary) + "\nRun fix-dependencies tool to attempt fixes."
    except Exception as e:
        utils.log.exception("MCP Tool: validate_dependencies failed")
        return f"Error validating dependencies: {e}"

@mcp_server.tool()
async def fix_dependencies(ctx: Context) -> str:
    """Automatically fixes invalid dependencies."""
    utils.log.info("MCP Tool: Running fix_dependencies")
    try:
        tasks_data_dict = utils.read_json(config.TASKS_FILE_PATH)
        if not tasks_data_dict: return "Error: Cannot read tasks file."
        tasks_data = models.TasksData.model_validate(tasks_data_dict)

        made_changes, fixes_summary = await asyncio.to_thread(
            dependency_manager.fix_dependencies,
            tasks_data
        )

        if made_changes:
            utils.write_json(config.TASKS_FILE_PATH, tasks_data.model_dump(mode='json', exclude_none=True))
            await asyncio.to_thread(
                task_manager.generate_task_files,
                config.TASKS_FILE_PATH,
                config.TASK_FILES_DIR
            )
            summary_str = ", ".join([f"{v} {k}" for k, v in fixes_summary.items() if v > 0])
            return f"Dependencies fixed ({summary_str}). Tasks file updated and task files regenerated."
        else:
            return "No dependency issues found or no automatic fixes applied."
    except Exception as e:
        utils.log.exception("MCP Tool: fix_dependencies failed")
        return f"Error fixing dependencies: {e}"

@mcp_server.tool()
async def research_topic(
    ctx: Context,
    topic: str,
    output_file: Optional[str] = None,
    search_top_k: int = config.STORM_SEARCH_TOP_K,
    ) -> str:
    """Generates a research report on a given topic using knowledge-storm."""
    utils.log.info(f"MCP Tool: Running research_topic (topic='{topic}')")
    if not task_manager._ensure_storm_imported():
        return "Error: knowledge-storm package required but not found or failed to import."
    try:
        storm_output_dir = config.STORM_OUTPUT_DIR
        report_path = Path(output_file) if output_file else (storm_output_dir / f"{utils.sanitize_filename(topic)}.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        article_content = await task_manager.run_storm_research_for_topic(
            topic_name=topic,
            questions=[f"Provide a comprehensive overview of {topic}."],
            search_top_k=search_top_k,
            # Use default conv/article tokens from config inside the function
        )

        if article_content is not None:
            await asyncio.to_thread(utils.write_file, report_path, article_content)
            abs_path = str(report_path.resolve())
            return f"STORM research complete. Report saved to: {abs_path}. Use 'get_storm_research' resource with topic '{topic}' to read."
        else:
            return "STORM research failed or no content was generated."
    except Exception as e:
        utils.log.exception(f"MCP Tool: research_topic failed for topic '{topic}'")
        return f"Error during STORM research for topic '{topic}': {e}"

# --- Main Execution ---
def run_server():
    """Runs the MCP server."""
    utils.log.info("Starting Task Researcher MCP Server...")
    try:
         mcp_server.run() # FastMCP's run handles the event loop
    except KeyboardInterrupt:
         utils.log.info("MCP Server shutting down...")
    except Exception as e:
         utils.log.exception("MCP Server encountered an unhandled exception.")
    finally:
         utils.log.info("MCP Server stopped.")

if __name__ == "__main__":
    run_server()
