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
        "task-researcher",
        "typer", "instructor", "litellm", "anthropic",
        "google-generativeai", "knowledge-storm", "rich",
        "python-dotenv", "pydantic", "aiohttp", "requests"
    ]
)

# --- MCP Resources ---
# Resource handlers remain async, calling sync utils.read functions.
# For very large files, these could block, consider adding to_thread if necessary.

@mcp_server.resource("tasks://current")
async def get_current_tasks(ctx: Context) -> Dict[str, Any]:
    """Provides the current content of the main tasks JSON file."""
    # Sync file read - usually fast enough. Could wrap in to_thread if files are huge.
    tasks_data_dict = utils.read_json(config.TASKS_FILE_PATH)
    if tasks_data_dict:
        try:
            validated_data = models.TasksData.model_validate(tasks_data_dict)
            return validated_data.model_dump(mode='json', exclude_none=True)
        except Exception as e:
             utils.log.warning(f"tasks.json at {config.TASKS_FILE_PATH} is invalid: {e}. Returning empty.")
             return models.TasksData().model_dump(mode='json', exclude_none=True)
    else:
        return models.TasksData().model_dump(mode='json', exclude_none=True)

@mcp_server.resource("report://complexity")
async def get_complexity_report(ctx: Context) -> Optional[Dict[str, Any]]:
    """Provides the content of the task complexity report, if it exists."""
    # Sync file read
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
    # Sync file read
    safe_filename = utils.sanitize_filename(topic_name) + ".md"
    report_path = config.STORM_OUTPUT_DIR / safe_filename
    content: Optional[str] = None
    if report_path.exists():
        content = utils.read_file(report_path)
    else:
        md_files = list(config.STORM_OUTPUT_DIR.glob(f"{utils.sanitize_filename(topic_name)}*.md"))
        if md_files:
            latest_md = max(md_files, key=lambda p: p.stat().st_mtime)
            utils.log.info(f"Found potential STORM output via glob for topic '{topic_name}': {latest_md}")
            content = utils.read_file(latest_md)
        else:
            utils.log.warning(f"Research report not found for topic '{topic_name}'")
    return content

@mcp_server.resource("taskfile://{task_id}")
async def get_task_file(ctx: Context, task_id: int) -> Optional[str]:
    """Provides the content of a specific generated task file (task_XXX.txt)."""
    # Sync file read
    file_path = config.TASK_FILES_DIR / f"task_{task_id:03d}.txt"
    content: Optional[str] = None
    if file_path.exists():
        content = utils.read_file(file_path)
    else:
        utils.log.warning(f"Task file not found: {file_path}")
    return content

# --- MCP Tools ---
# All tool handlers that perform potentially blocking IO or significant CPU work
# should be async and use await asyncio.to_thread for sync calls,
# or await direct async calls.

@mcp_server.tool()
async def parse_inputs(
    ctx: Context, # Keep ctx even if unused
    num_tasks: int = 15,
    func_spec_path: Optional[str] = None,
    tech_spec_path: Optional[str] = None,
    plan_path: Optional[str] = None,
    research_doc_path: Optional[str] = None
    ) -> str:
    """
    Parses input specs to generate initial tasks and overwrites the tasks file.
    (Calls async task_manager.parse_inputs)
    """
    utils.log.info(f"MCP Tool: Running parse_inputs (num_tasks={num_tasks})")
    try:
        f_path = Path(func_spec_path) if func_spec_path else config.FUNCTIONAL_SPEC_PATH
        t_path = Path(tech_spec_path) if tech_spec_path else config.TECHNICAL_SPEC_PATH
        p_path = Path(plan_path) if plan_path else config.PLAN_PATH
        r_path = Path(research_doc_path) if research_doc_path else config.DEEP_RESEARCH_PATH

        await task_manager.parse_inputs( # Already async
            num_tasks=num_tasks, tasks_file_path=config.TASKS_FILE_PATH,
            func_spec_path=f_path, tech_spec_path=t_path, plan_path=p_path, research_path=r_path,
        )
        tasks_data = utils.read_json(config.TASKS_FILE_PATH) # Sync read after async write
        count = len(tasks_data.get('tasks', [])) if tasks_data else 0
        return f"Successfully parsed inputs and generated {count} tasks in {config.TASKS_FILE_PATH}."
    except Exception as e:
        utils.log.exception("MCP Tool: parse_inputs failed")
        return f"Error parsing inputs: {e}"

@mcp_server.tool()
async def update_tasks(
    ctx: Context, # Keep ctx
    from_id: int,
    prompt: str,
    research_hint: bool = False
    ) -> str:
    """
    Updates tasks from a specific ID based on the provided prompt.
    (Calls async task_manager.update_tasks)
    """
    utils.log.info(f"MCP Tool: Running update_tasks (from_id={from_id}, research_hint={research_hint})")
    try:
        await task_manager.update_tasks( # Already async
            from_id=from_id, prompt=prompt, use_research=research_hint,
            tasks_file_path=config.TASKS_FILE_PATH
        )
        return f"Update process completed for tasks starting from ID {from_id}. Check tasks file."
    except Exception as e:
        utils.log.exception("MCP Tool: update_tasks failed")
        return f"Error updating tasks: {e}"

@mcp_server.tool()
async def generate_task_files(ctx: Context) -> str: # Uses to_thread
    """
    Generates individual task description files (task_XXX.txt).
    (Calls sync task_manager.generate_task_files)
    """
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
    ctx: Context, # Keep ctx
    task_id: int,
    num_subtasks: Optional[int] = None,
    research: bool = False,
    prompt: Optional[str] = None,
    force: bool = False
    ) -> str:
    """
    Expands a specific task into subtasks using AI (optional STORM research).
    (Calls async task_manager.expand_task)
    """
    utils.log.info(f"MCP Tool: Running expand_task (id={task_id}, research={research}, force={force})")
    if research and not task_manager._ensure_storm_imported():
        return "Error: knowledge-storm package required for --research but not found or failed to import."
    try:
        await task_manager.expand_task( # Already async
            task_id=task_id, num_subtasks=num_subtasks, use_research=research,
            prompt=prompt, force=force, tasks_file_path=config.TASKS_FILE_PATH
        )
        tasks_data = utils.read_json(config.TASKS_FILE_PATH) # Sync read ok after await
        task_dict = utils.find_task_by_id(tasks_data.get('tasks',[]), task_id) if tasks_data else None
        sub_count = len(task_dict.get('subtasks',[])) if task_dict else 0
        return f"Successfully expanded task {task_id}. It now has {sub_count} subtasks."
    except Exception as e:
        utils.log.exception(f"MCP Tool: expand_task id={task_id} failed")
        return f"Error expanding task {task_id}: {e}"

@mcp_server.tool()
async def expand_all_tasks(
    ctx: Context, # Keep ctx
    num_subtasks: Optional[int] = None,
    research: bool = False,
    prompt: Optional[str] = None,
    force: bool = False
    ) -> str:
    """
    Expands all eligible pending tasks into subtasks using AI (optional STORM research).
    (Calls async task_manager.expand_all_tasks)
    """
    utils.log.info(f"MCP Tool: Running expand_all_tasks (research={research}, force={force})")
    if research and not task_manager._ensure_storm_imported():
        return "Error: knowledge-storm package required for --research but not found or failed to import."
    try:
        await task_manager.expand_all_tasks( # Already async
            num_subtasks=num_subtasks, use_research=research, prompt=prompt,
            force=force, tasks_file_path=config.TASKS_FILE_PATH
        )
        return f"Bulk expansion process completed. Check logs and tasks file for details."
    except Exception as e:
        utils.log.exception(f"MCP Tool: expand_all_tasks failed")
        return f"Error expanding all tasks: {e}"

@mcp_server.tool()
async def analyze_complexity(
    ctx: Context, # Keep ctx
    research_hint: bool = False,
    threshold: float = 5.0,
    output_file: Optional[str] = None
    ) -> str:
    """
    Analyzes complexity of all tasks using AI and saves a report.
    (Calls async task_manager.analyze_complexity)
    """
    utils.log.info(f"MCP Tool: Running analyze_complexity (research_hint={research_hint})")
    try:
        report_path = Path(output_file) if output_file else config.COMPLEXITY_REPORT_PATH
        await task_manager.analyze_complexity( # Already async
            output_path=report_path, use_research=research_hint, threshold=threshold,
            tasks_file_path=config.TASKS_FILE_PATH
        )
        if report_path.exists():
             # Return absolute path for clarity
             abs_path = str(report_path.resolve())
             return f"Complexity analysis complete. Report saved to: {abs_path}. Use 'get_complexity_report' resource to read."
        else:
             return "Complexity analysis ran, but report file was not found."
    except Exception as e:
        utils.log.exception("MCP Tool: analyze_complexity failed")
        return f"Error analyzing complexity: {e}"

@mcp_server.tool()
async def validate_dependencies(ctx: Context) -> str: # Uses to_thread
    """
    Validates task dependencies for issues like missing refs or cycles.
    (Calls sync dependency_manager.validate_dependencies)
    """
    utils.log.info("MCP Tool: Running validate_dependencies")
    try:
        # Read synchronously before thread work
        tasks_data_dict = utils.read_json(config.TASKS_FILE_PATH)
        if not tasks_data_dict: return "Error: Cannot read tasks file."
        tasks_data = models.TasksData.model_validate(tasks_data_dict)

        # Run synchronous validation in a thread
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
async def fix_dependencies(ctx: Context) -> str: # Uses to_thread
    """
    Automatically fixes invalid dependencies (missing refs, self-deps, simple cycles).
    (Calls sync dependency_manager.fix_dependencies and sync task_manager.generate_task_files)
    """
    utils.log.info("MCP Tool: Running fix_dependencies")
    try:
        # Read synchronously before starting thread work
        tasks_data_dict = utils.read_json(config.TASKS_FILE_PATH)
        if not tasks_data_dict: return "Error: Cannot read tasks file."
        tasks_data = models.TasksData.model_validate(tasks_data_dict)

        # Run synchronous fixing logic in a thread
        made_changes, fixes_summary = await asyncio.to_thread(
            dependency_manager.fix_dependencies,
            tasks_data
        )

        if made_changes:
            # Write and generate files synchronously after thread completes
            utils.write_json(config.TASKS_FILE_PATH, tasks_data.model_dump(mode='json', exclude_none=True))
            # Also run file generation in a thread as it involves IO
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
async def research_topic( # Uses await on async task_manager function
    ctx: Context, # Keep ctx
    topic: str,
    output_file: Optional[str] = None,
    search_top_k: int = config.STORM_SEARCH_TOP_K,
    ) -> str:
    """
    Generates a research report on a given topic using knowledge-storm.
    (Calls async task_manager.run_storm_research_for_topic)
    """
    utils.log.info(f"MCP Tool: Running research_topic (topic='{topic}')")
    if not task_manager._ensure_storm_imported():
        return "Error: knowledge-storm package required but not found or failed to import."
    try:
        storm_output_dir = config.STORM_OUTPUT_DIR
        report_path = Path(output_file) if output_file else (storm_output_dir / f"{utils.sanitize_filename(topic)}.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Await the async function which uses to_thread internally for STORM's sync run
        article_content = await task_manager.run_storm_research_for_topic(
            topic_name=topic,
            questions=[f"Provide a comprehensive overview of {topic}."],
            search_top_k=search_top_k,
        )

        if article_content is not None: # Check for None on failure
            # Write file sync after await
            utils.write_file(report_path, article_content)
            abs_path = str(report_path.resolve())
            # Return path and mention the resource
            return f"STORM research complete. Report saved to: {abs_path}. Use 'get_storm_research' resource with topic '{topic}' to read."
        else:
            return "STORM research failed or no content was generated."
    except Exception as e:
        utils.log.exception(f"MCP Tool: research_topic failed for topic '{topic}'")
        return f"Error during STORM research for topic '{topic}': {e}"

# --- Main Execution --- (remains the same)
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