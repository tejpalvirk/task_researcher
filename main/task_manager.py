import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import asyncio # For running STORM maybe? STORM itself isn't async yet.

from . import config, utils, ai_services, dependency_manager, ui, models
from .models import TasksData, Task, Subtask, ComplexityReport
from .utils import log, read_json, write_json, task_exists, find_task_by_id, get_next_task_id, get_next_subtask_id, read_complexity_report, find_task_in_complexity_report, sanitize_filename, read_file, write_file
from rich.prompt import Confirm

# Lazy import STORM stuff
KnowledgeStormRunner = None
STORMWikiRunnerArguments = None
STORMWikiLMConfigs = None
LitellmModel = None
RM_CLASSES = {} # Dictionary to hold imported RM classes

def _import_storm_lazy():
    global KnowledgeStormRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs, LitellmModel, RM_CLASSES
    if KnowledgeStormRunner: # Already imported
        return True
    try:
        from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs
        from knowledge_storm.lm import LitellmModel
        # Import all known RM classes we might need
        from knowledge_storm import rm as storm_rm
        RM_CLASSES['bing'] = getattr(storm_rm, 'BingSearch', None)
        RM_CLASSES['you'] = getattr(storm_rm, 'YouRM', None)
        RM_CLASSES['google'] = getattr(storm_rm, 'GoogleSearch', None)
        RM_CLASSES['serper'] = getattr(storm_rm, 'SerperRM', None)
        RM_CLASSES['tavily'] = getattr(storm_rm, 'TavilySearchRM', None)
        RM_CLASSES['duckduckgo'] = getattr(storm_rm, 'DuckDuckGoSearchRM', None)
        # Add others as needed...
        log.info("knowledge-storm package imported successfully.")
        return True
    except ImportError:
        log.error("Failed to import 'knowledge-storm'. Research command requires 'pip install knowledge-storm'")
        ui.console.print("[bold red]Error:[/bold red] The 'knowledge-storm' package is not installed. Cannot use research features.")
        ui.console.print("Please install it using: [cyan]pip install knowledge-storm[/]")
        return False
    except Exception as e:
        log.exception(f"An unexpected error occurred while importing knowledge-storm: {e}")
        return False

# --- Core Task Functions --- (parse_inputs, update_tasks, generate_task_files unchanged)
async def parse_inputs( # Make async
    num_tasks: int,
    tasks_file_path: Path = config.TASKS_FILE_PATH,
    func_spec_path: Path = config.FUNCTIONAL_SPEC_PATH,
    tech_spec_path: Path = config.TECHNICAL_SPEC_PATH,
    plan_path: Path = config.PLAN_PATH,
    research_path: Path = config.BACKGROUND_PATH,
) -> None:
    """Parses input files and generates initial tasks using an LLM."""
    log.info(f"Parsing inputs to generate ~{num_tasks} tasks.")
    func_spec = read_file(func_spec_path)
    tech_spec = read_file(tech_spec_path)
    plan = read_file(plan_path)
    research = read_file(research_path)
    source_files = [str(p.resolve()) for p in [func_spec_path, tech_spec_path, plan_path, research_path] if p.exists()]

    if not any([func_spec, tech_spec, plan, research]):
        log.error("No input content found in specification files. Cannot generate tasks.")
        ui.console.print("[bold red]Error:[/bold red] No content found in input specification files.")
        return

    if not plan:
         log.warning("Plan file (`plan.md`) is empty or missing. Task generation might lack structure.")
         ui.console.print("[yellow]Warning:[/yellow] Plan file is empty or missing. Task generation might lack structure.")

    ui.console.print(f"Calling AI to generate ~{num_tasks} tasks...")
    spinner = ui.console.status("[bold green]Generating tasks with AI...", spinner="dots")
    spinner.start()
    try:
        tasks_data = await ai_services.call_llm_for_tasks( # Use await
            functional_spec=func_spec,
            technical_spec=tech_spec,
            plan=plan,
            background=research,
            num_tasks=num_tasks,
            source_files=source_files,
        )
        spinner.stop()

        if not tasks_data:
            raise Exception("AI failed to return valid task data.")

        # Validate and fix dependencies immediately after generation
        log.info("Validating and fixing dependencies for newly generated tasks...")
        dep_made_changes, dep_fixes = dependency_manager.fix_dependencies(tasks_data)
        if dep_made_changes:
             log.info(f"Dependency fixes applied: {dep_fixes}")
        else:
             log.info("No dependency fixes needed.")

        tasks_data.meta.generatedAt = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(tasks_file_path, tasks_data.model_dump(mode='json', exclude_none=True))

        ui.console.print(f"[bold green]Success:[/bold green] Generated {len(tasks_data.tasks)} tasks and saved to {tasks_file_path}")
        generate_task_files(tasks_file_path, config.TASK_FILES_DIR) # Generate files automatically

    except Exception as e:
        spinner.stop()
        log.exception("Failed to parse inputs and generate tasks.")
        ui.console.print(f"[bold red]Error:[/bold red] Failed to generate tasks: {e}")

async def update_tasks( # Make async
    from_id: int,
    prompt: str,
    use_research: bool = False, # This flag now only affects the prompt for update
    tasks_file_path: Path = config.TASKS_FILE_PATH,
) -> None:
    """Updates tasks from a specific ID based on a prompt."""
    log.info(f"Updating tasks from ID {from_id} using prompt: '{prompt}'")
    tasks_data_dict = read_json(tasks_file_path)
    if not tasks_data_dict:
        ui.console.print(f"[bold red]Error:[/bold red] Cannot read tasks from {tasks_file_path}")
        return
    try:
        tasks_data = models.TasksData.model_validate(tasks_data_dict)
    except Exception as e:
        log.error(f"Invalid tasks file structure: {e}")
        ui.console.print(f"[bold red]Error:[/bold red] Invalid tasks file structure in {tasks_file_path}")
        return

    tasks_to_update_models = [
        task for task in tasks_data.tasks
        if task.id >= from_id and task.status != "done"
    ]

    if not tasks_to_update_models:
        ui.console.print(f"[yellow]Info:[/yellow] No pending tasks found with ID >= {from_id} to update.")
        return

    ui.console.print(f"Found {len(tasks_to_update_models)} tasks to update (ID >= {from_id} and not 'done').")
    ui.display_tasks_summary(tasks_to_update_models) # Show summary before update

    if not Confirm.ask(f"Proceed with updating {len(tasks_to_update_models)} tasks?", default=True):
        ui.console.print("Update cancelled.")
        return

    ui.console.print(f"Calling AI to update tasks (Research prompt hint: {use_research})...")
    spinner = ui.console.status("[bold green]Updating tasks with AI...", spinner="dots")
    spinner.start()
    try:
        # Pass Pydantic models directly
        updated_task_models = await ai_services.update_tasks_with_llm(
            tasks_to_update_models, prompt, use_research_prompt=use_research
        )
        spinner.stop()

        # Merge updates back into tasks_data
        updated_count = 0
        if updated_task_models:
            updates_dict = {task.id: task for task in updated_task_models}
            for i, task in enumerate(tasks_data.tasks):
                if task.id in updates_dict:
                    # Preserve original subtasks if not returned
                    original_subtasks = task.subtasks
                    tasks_data.tasks[i] = updates_dict[task.id]
                    # Ensure subtasks are lists of Subtask models if restored
                    if not tasks_data.tasks[i].subtasks and original_subtasks:
                         tasks_data.tasks[i].subtasks = [Subtask(**st.model_dump()) if isinstance(st, models.Subtask) else Subtask(**st) for st in original_subtasks]
                    elif tasks_data.tasks[i].subtasks: # Ensure loaded subtasks are models too
                         tasks_data.tasks[i].subtasks = [Subtask(**st.model_dump()) if isinstance(st, models.Subtask) else Subtask(**st) for st in tasks_data.tasks[i].subtasks]

                    updated_count +=1

        if updated_count > 0:
            log.info("Validating and fixing dependencies after update...")
            dep_made_changes, dep_fixes = dependency_manager.fix_dependencies(tasks_data)
            if dep_made_changes:
                log.info(f"Dependency fixes applied after update: {dep_fixes}")

            write_json(tasks_file_path, tasks_data.model_dump(mode='json', exclude_none=True))
            ui.console.print(f"[bold green]Success:[/bold green] Updated {updated_count} tasks in {tasks_file_path}")
            generate_task_files(tasks_file_path, config.TASK_FILES_DIR) # Regenerate files
        else:
             ui.console.print("[yellow]Warning:[/yellow] AI did not return valid updates or no tasks were updated.")

    except Exception as e:
        spinner.stop()
        log.exception("Failed to update tasks.")
        ui.console.print(f"[bold red]Error:[/bold red] Failed to update tasks: {e}")


def generate_task_files(
    tasks_file_path: Path = config.TASKS_FILE_PATH,
    output_dir: Path = config.TASK_FILES_DIR,
) -> None:
    """Generates individual .txt files for each task."""
    log.info(f"Generating task files in directory: {output_dir}")
    tasks_data_dict = read_json(tasks_file_path)
    if not tasks_data_dict:
        ui.console.print(f"[bold red]Error:[/bold red] Cannot read tasks from {tasks_file_path}")
        return
    try:
        # Validate data using Pydantic models before proceeding
        tasks_data = models.TasksData.model_validate(tasks_data_dict)
    except Exception as e:
        log.error(f"Invalid tasks file structure: {e}")
        ui.console.print(f"[bold red]Error:[/bold red] Invalid tasks file structure in {tasks_file_path}")
        return

    if not tasks_data.tasks:
        ui.console.print("[yellow]No tasks found to generate files for.[/yellow]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    log.info("Validating dependencies before generating files...")
    # Run validation (but don't necessarily fix here, assume fixed previously)
    # _, issues = dependency_manager.validate_dependencies(tasks_data)
    # if issues:
    #      log.warning(f"Found {len(issues)} dependency issues before generating files. Files might contain invalid references.")

    for task in tasks_data.tasks:
        file_path = output_dir / f"task_{task.id:03d}.txt"
        # Pass the validated Pydantic Task model and the full list for context
        content = ui.format_task_for_file(task, tasks_data.tasks)
        try:
            file_path.write_text(content, encoding='utf-8')
            generated_count += 1
            log.debug(f"Generated file: {file_path.name}")
        except Exception as e:
            log.error(f"Failed to write task file {file_path.name}: {e}")

    ui.console.print(f"[bold green]Success:[/bold green] Generated {generated_count} task files in {output_dir}")


# --- Research & Expansion Workflow ---

def _get_storm_runner(search_top_k: int, max_conv_tokens: int, max_article_tokens: int) -> Optional[Any]:
    """Configures and returns a STORM runner instance. (Sync setup)"""
    if not _import_storm_lazy():
        return None

    # 1. LM Configs
    lm_configs = STORMWikiLMConfigs()
    storm_llm_kwargs = {"model": config.STORM_LLM_MODEL, "temperature": config.TEMPERATURE}
    # Create LitellmModel instances for STORM components
    lm_powerful = LitellmModel(**storm_llm_kwargs, max_tokens=max_article_tokens)
    lm_fast = LitellmModel(**storm_llm_kwargs, max_tokens=max_conv_tokens)
    lm_configs.set_conv_simulator_lm(lm_fast)
    lm_configs.set_question_asker_lm(lm_fast)
    lm_configs.set_outline_gen_lm(lm_powerful)
    lm_configs.set_article_gen_lm(lm_powerful)
    lm_configs.set_article_polish_lm(lm_powerful)

    # 2. Retriever Setup
    rm = None
    retriever_type = config.STORM_RETRIEVER
    RMClass = RM_CLASSES.get(retriever_type)
    if not RMClass:
        ui.console.print(f"[bold red]Error:[/bold red] Unsupported STORM retriever type configured: {retriever_type}")
        return None
    try:
        if retriever_type == 'bing':
            if not config.BING_SEARCH_API_KEY: raise ValueError("BING_SEARCH_API_KEY not set.")
            rm = RMClass(bing_search_api_key=config.BING_SEARCH_API_KEY, k=search_top_k)
        elif retriever_type == 'you':
            if not config.YDC_API_KEY: raise ValueError("YDC_API_KEY not set.")
            rm = RMClass(ydc_api_key=config.YDC_API_KEY, k=search_top_k)
        elif retriever_type == 'tavily':
             if not config.TAVILY_API_KEY: raise ValueError("TAVILY_API_KEY not set.")
             rm = RMClass(tavily_api_key=config.TAVILY_API_KEY, k=search_top_k)
        elif retriever_type == 'serper':
              if not config.SERPER_API_KEY: raise ValueError("SERPER_API_KEY not set.")
              rm = RMClass(serper_api_key=config.SERPER_API_KEY, k=search_top_k)
        else: # Attempt generic init
             rm = RMClass(k=search_top_k)
    except ValueError as ve:
        ui.console.print(f"[bold red]Error:[/bold red] Missing API key for {retriever_type} retriever: {ve}")
        return None
    except Exception as e:
        ui.console.print(f"[bold red]Error:[/bold red] Failed to initialize {retriever_type} retriever: {e}")
        return None


    # 3. Engine Args
    engine_args = STORMWikiRunnerArguments(
        output_dir=config.STORM_OUTPUT_DIR,
        remove_intermediate_files=False,
    )

    return STORMWikiRunner(engine_args, lm_configs, rm)

def _run_storm_sync(runner: Any, storm_topic_input: str):
    """Synchronous wrapper for runner.run() to be used with to_thread."""
    runner.run(
        topic=storm_topic_input,
        do_research=True,
        do_generate_outline=True,
        do_generate_article=True,
        do_polish_article=True,
    )
    # Note: STORM runner doesn't directly return the article content from run()
    # We need to find the file afterwards. runner.run() returns None.

async def run_storm_research_for_topic(
    topic_name: str,
    questions: List[str],
    search_top_k: int = config.STORM_SEARCH_TOP_K,
    max_conv_tokens: int = config.STORM_MAX_TOKENS_CONV,
    max_article_tokens: int = config.STORM_MAX_TOKENS_ARTICLE,
) -> Optional[str]:
    """Runs the STORM pipeline (sync part in thread) for a specific topic and questions."""
    runner = _get_storm_runner(search_top_k, max_conv_tokens, max_article_tokens)
    if not runner:
        return None # Setup failed

    storm_topic_input = f"{topic_name}: Research Summary focusing on:\n" + "\n".join(f"- {q}" for q in questions)
    storm_topic_filename_base = sanitize_filename(topic_name) # Used for finding the file

    ui.console.print(f"🌪️ Running STORM in background thread for topic: [bold cyan]{topic_name}[/bold cyan]...")
    try:
        # Run the synchronous STORM run method in a separate thread
        await asyncio.to_thread(
            _run_storm_sync, # Pass the wrapper function
            runner,          # Pass the runner instance
            storm_topic_input # Pass the topic input
        )
        log.info(f"STORM synchronous run completed for topic '{topic_name}'.")

        # --- Find the generated article file (remains the same logic) ---
        expected_file_path = config.STORM_OUTPUT_DIR / f"{sanitize_filename(storm_topic_input)}.md"
        fallback_file_path = config.STORM_OUTPUT_DIR / f"{storm_topic_filename_base}.md"
        article_content = None

        if expected_file_path.exists():
            article_content = read_file(expected_file_path)
            log.info(f"Found STORM output at: {expected_file_path}")
        elif fallback_file_path.exists():
            article_content = read_file(fallback_file_path)
            log.info(f"Found STORM output at fallback path: {fallback_file_path}")
        else:
            md_files = list(config.STORM_OUTPUT_DIR.glob(f"{storm_topic_filename_base}*.md"))
            if md_files:
                 latest_md = max(md_files, key=lambda p: p.stat().st_mtime)
                 article_content = read_file(latest_md)
                 log.warning(f"Found potential STORM output via glob: {latest_md}")
            else:
                log.error(f"STORM finished for topic '{topic_name}', but couldn't find the output markdown file.")
                ui.console.print(f"[yellow]Warning:[/yellow] STORM couldn't find output for topic '{topic_name}'.")

        # runner.post_run() # If cleanup is needed and potentially blocking, wrap in to_thread too

        return article_content if article_content else ""

    except Exception as e:
        log.exception(f"STORM research thread failed for topic '{topic_name}'.")
        ui.console.print(f"[bold red]Error during STORM research for topic '{topic_name}':[/bold red] {e}")
        return None # Indicate failure


async def expand_task(
    task_id: int,
    num_subtasks: Optional[int] = None,
    use_research: bool = False,
    prompt: Optional[str] = None,
    force: bool = False,
    tasks_file_path: Path = config.TASKS_FILE_PATH,
) -> None:
    """Expands a single task into subtasks, optionally using async STORM research."""
    log.info(f"Attempting to expand task {task_id}...")
    tasks_data_dict = read_json(tasks_file_path)
    if not tasks_data_dict:
        ui.console.print(f"[bold red]Error:[/bold red] Cannot read tasks from {tasks_file_path}")
        return
    try:
        tasks_data = models.TasksData.model_validate(tasks_data_dict)
    except Exception as e:
        log.error(f"Invalid tasks file structure: {e}")
        ui.console.print(f"[bold red]Error:[/bold red] Invalid tasks file structure in {tasks_file_path}")
        return

    task_index = next((i for i, t in enumerate(tasks_data.tasks) if t.id == task_id), -1)
    if task_index == -1:
        ui.console.print(f"[bold red]Error:[/bold red] Task with ID {task_id} not found.")
        return
    task = tasks_data.tasks[task_index]
    if task.status == "done":
        ui.console.print(f"[yellow]Info:[/yellow] Task {task_id} is already 'done'. Skipping expansion.")
        return
    if task.subtasks and not force:
        ui.console.print(f"[yellow]Info:[/yellow] Task {task_id} already has {len(task.subtasks)} subtasks. Use --force to overwrite.")
        return
    elif task.subtasks and force:
        ui.console.print(f"[yellow]Warning:[/yellow] Overwriting existing {len(task.subtasks)} subtasks for task {task_id} due to --force flag.")
        task.subtasks = []

    # --- Determine parameters (remains the same) ---
    complexity_report = read_complexity_report()
    task_analysis = find_task_in_complexity_report(complexity_report, task.id) if complexity_report else None
    final_num_subtasks = num_subtasks if num_subtasks is not None else config.DEFAULT_SUBTASKS
    final_context = prompt or "" # User prompt takes precedence
    if task_analysis:
        log.info(f"Using complexity analysis for task {task.id} (Score: {task_analysis.get('complexityScore', 'N/A')})")
        if num_subtasks is None and task_analysis.get('recommendedSubtasks'):
            final_num_subtasks = task_analysis['recommendedSubtasks']
            log.info(f"Using recommended number of subtasks: {final_num_subtasks}")
        if not prompt and task_analysis.get('expansionPrompt'):
            if not use_research:
                final_context = task_analysis['expansionPrompt']
                log.info("Using expansion prompt from complexity report.")
            else:
                 log.info("Ignoring complexity report prompt because --research (STORM) is enabled.")

    ui.console.print(f"Expanding task {task.id} '{task.title}' into {final_num_subtasks} subtasks...")

    # --- Research Workflow (if --research) ---
    storm_research_summary = ""
    if use_research:
        ui.console.print("🔬 [bold]Starting Research Phase...[/bold]")
        research_spinner = ui.console.status("[cyan]Generating research questions...", spinner="dots")
        research_spinner.start()
        try:
            # 1. Generate Questions (Async)
            questions = await ai_services.generate_research_questions(task) # Assuming this is async now
            if not questions: raise Exception("Failed to generate research questions.")
            research_spinner.update("[cyan]Grouping questions into topics...")
            ui.console.print(f"  💬 Generated {len(questions)} research questions.")

            # 2. Group into Topics (Async)
            topics = await ai_services.group_questions_into_topics(questions, task) # Assuming async
            if not topics:
                 topics = {"General Research": questions}
                 log.warning("Failed to group questions; using single topic.")
                 ui.console.print("[yellow]  ⚠️ Could not group questions; proceeding with single topic.[/yellow]")
            else:
                 ui.console.print(f"  📑 Grouped into {len(topics)} topics: {', '.join(topics.keys())}")

            # 3. Run STORM for each topic (Using await on the async wrapper)
            research_spinner.stop()
            storm_outputs = []
            topic_count = len(topics)
            # Run STORM topics concurrently? Could be faster but resource intensive.
            # Let's run sequentially for now to manage resources/API limits.
            for i, (topic_name, topic_questions) in enumerate(topics.items()):
                 ui.console.print(f"\n🔄 [bold]Researching Topic {i+1}/{topic_count}: {topic_name}[/bold] ({len(topic_questions)} questions)")
                 # Await the async function which uses to_thread internally
                 storm_output = await run_storm_research_for_topic(topic_name, topic_questions)
                 if storm_output is not None: # Check for None on failure
                      storm_outputs.append(f"## Research Topic: {topic_name}\n\n{storm_output}\n\n---\n")
                      ui.console.print(f"  ✅ Completed research for '{topic_name}'.")
                 else:
                      ui.console.print(f"  ❌ Failed research for '{topic_name}'.")

            # 4. Combine results (remains the same)
            if storm_outputs:
                storm_research_summary = "\n".join(storm_outputs)
                ui.console.print("\n✅ [bold green]Research Phase Complete.[/bold green]")
            else:
                 ui.console.print("\n⚠️ [bold yellow]Research Phase completed with no successful STORM outputs.[/bold yellow]")

        except Exception as research_error:
            research_spinner.stop()
            log.exception(f"Research phase failed for task {task_id}: {research_error}")
            ui.console.print(f"\n❌ [bold red]Research Phase Failed:[/bold red] {research_error}")
            storm_research_summary = "" # Ensure empty on failure

    # --- Generate Subtasks (remains the same, but combined_context might be larger) ---
    combined_context = f"{storm_research_summary}\n\nUser Provided Context:\n{final_context}".strip()
    ui.console.print(f"\n🧠 [bold]Generating {final_num_subtasks} Subtasks...[/bold]")
    spinner = ui.console.status("[bold green]Generating subtasks with AI...", spinner="dots")
    spinner.start()
    try:
        next_sub_id = get_next_subtask_id(task)
        # Await the async subtask generation
        new_subtasks = await ai_services.generate_subtasks_with_llm(
            task, final_num_subtasks, next_sub_id, combined_context
        )
        spinner.stop()

        if new_subtasks:
            # Add subtasks (ensure they are Subtask model instances)
            tasks_data.tasks[task_index].subtasks.extend(
                [models.Subtask(**sub.model_dump(exclude_none=True)) for sub in new_subtasks]
            )
            log.info("Validating and fixing dependencies after expansion...")
            dep_made_changes, dep_fixes = dependency_manager.fix_dependencies(tasks_data)
            if dep_made_changes: log.info(f"Dependency fixes applied after expansion: {dep_fixes}")

            write_json(tasks_file_path, tasks_data.model_dump(mode='json', exclude_none=True))
            ui.console.print(f"✅ [bold green]Success:[/bold green] Added {len(new_subtasks)} subtasks to task {task_id}.")
            ui.display_subtasks_summary(new_subtasks, task.id)
            generate_task_files(tasks_file_path, config.TASK_FILES_DIR)
        else:
             ui.console.print("⚠️ [yellow]Warning:[/yellow] AI did not return any subtasks.")

    except Exception as e:
        spinner.stop()
        log.exception(f"Failed to expand task {task_id}.")
        ui.console.print(f"❌ [bold red]Error:[/bold red] Failed to expand task {task_id}: {e}")


async def expand_all_tasks(
    num_subtasks: Optional[int] = None,
    use_research: bool = False,
    prompt: Optional[str] = None,
    force: bool = False,
    tasks_file_path: Path = config.TASKS_FILE_PATH,
) -> None:
    """Expands all eligible pending tasks, calling async expand_task."""
    log.info("Attempting to expand all eligible tasks...")
    tasks_data_dict = read_json(tasks_file_path)
    if not tasks_data_dict:
        ui.console.print(f"[bold red]Error:[/bold red] Cannot read tasks from {tasks_file_path}")
        return
    try:
        tasks_data = models.TasksData.model_validate(tasks_data_dict)
    except Exception as e:
        log.error(f"Invalid tasks file structure: {e}")
        ui.console.print(f"[bold red]Error:[/bold red] Invalid tasks file structure in {tasks_file_path}")
        return

    tasks_to_expand_models = []
    for task_model in tasks_data.tasks: # Iterate over models
        if task_model.status != "done":
            if not task_model.subtasks or force:
                tasks_to_expand_models.append(task_model)
            else:
                 log.info(f"Skipping task {task_model.id} (already has subtasks and --force not used).")

    if not tasks_to_expand_models:
        ui.console.print("[yellow]Info:[/yellow] No tasks eligible for expansion found.")
        return

    ui.console.print(f"Found {len(tasks_to_expand_models)} tasks eligible for expansion.")
    ui.display_tasks_summary(tasks_to_expand_models) # Expects list of Task models

    complexity_report = read_complexity_report()
    if complexity_report:
        complexity_map = {analysis.get('taskId'): analysis.get('complexityScore', 0)
                          for analysis in complexity_report.get('complexityAnalysis', [])}
        tasks_to_expand_models.sort(key=lambda t: complexity_map.get(t.id, 0), reverse=True)
        ui.console.print("[italic]Tasks sorted by complexity (highest first).[/italic]")
    else:
        tasks_to_expand_models.sort(key=lambda t: t.id)

    if not Confirm.ask(f"Proceed with expanding {len(tasks_to_expand_models)} tasks?", default=True):
        ui.console.print("Expansion cancelled.")
        return

    expanded_count = 0
    failed_count = 0
    ui.console.print(f"\n--- Starting Expansion for {len(tasks_to_expand_models)} Tasks ---")

    # Run expansions sequentially to avoid overwhelming resources/APIs
    for i, task_model in enumerate(tasks_to_expand_models):
        ui.console.print(f"\n[{i+1}/{len(tasks_to_expand_models)}] Expanding Task {task_model.id}: '{task_model.title}'...")
        try:
            # Await the async single expand function
            await expand_task( # await the call
                 task_id=task_model.id,
                 num_subtasks=num_subtasks,
                 use_research=use_research,
                 prompt=prompt,
                 force=force, # Pass force flag correctly
                 tasks_file_path=tasks_file_path
             )
            expanded_count += 1
            # Small delay? Might help rate limits, but adds time. Optional.
            # await asyncio.sleep(1)
        except Exception as e:
            # expand_task already prints errors, just log and count failure
            log.error(f"Expansion failed for task {task_model.id} in bulk operation: {e}")
            failed_count += 1

    ui.console.print(f"\n--- Bulk Expansion Complete ---")
    ui.console.print(f"[bold green]Successfully processed {expanded_count} tasks for expansion.[/bold green]")
    if failed_count > 0:
        ui.console.print(f"[bold red]Failed to initiate or complete expansion for {failed_count} tasks.[/bold red]")
    # Final dependency check and file generation is handled within each `expand_task` call now.


async def analyze_complexity( # Make async
    output_path: Optional[Path] = None,
    use_research: bool = False, # Affects prompt hint only
    threshold: float = 5.0,
    tasks_file_path: Path = config.TASKS_FILE_PATH,
) -> None:
    """Analyzes task complexity using an LLM and saves a report."""
    log.info("Analyzing task complexity...")
    output_path = output_path or config.COMPLEXITY_REPORT_PATH
    tasks_data_dict = read_json(tasks_file_path)
    if not tasks_data_dict:
        ui.console.print(f"[bold red]Error:[/bold red] Cannot read tasks from {tasks_file_path}")
        return
    try:
        tasks_data = models.TasksData.model_validate(tasks_data_dict)
    except Exception as e:
        log.error(f"Invalid tasks file structure: {e}")
        ui.console.print(f"[bold red]Error:[/bold red] Invalid tasks file structure in {tasks_file_path}")
        return

    if not tasks_data.tasks:
        ui.console.print("[yellow]No tasks found to analyze.[/yellow]")
        return

    ui.console.print(f"Found {len(tasks_data.tasks)} tasks. Calling AI for complexity analysis (Research hint: {use_research})...")
    spinner = ui.console.status("[bold green]Analyzing complexity with AI...", spinner="dots")
    spinner.start()
    try:
        # Pass the validated Pydantic model
        analysis_results = await ai_services.analyze_task_complexity_with_llm(
            tasks_data, use_research_prompt=use_research
        )
        spinner.stop()

        if not analysis_results:
             ui.console.print("[yellow]Warning:[/yellow] AI did not return any complexity analysis results.")
             return

        report = models.ComplexityReport(
            meta={
                "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tasksAnalyzed": len(analysis_results),
                "inputFile": str(tasks_file_path.resolve()),
                "thresholdScore": threshold,
                "projectName": tasks_data.meta.projectName,
                "projectVersion": tasks_data.meta.projectVersion,
                "usedResearchHint": use_research, # Renamed for clarity
                "llm_model": config.LLM_MODEL,
            },
            complexityAnalysis=sorted(analysis_results, key=lambda x: x.complexityScore, reverse=True)
        )

        write_json(output_path, report.model_dump(mode='json', exclude_none=True))
        ui.console.print(f"[bold green]Success:[/bold green] Complexity analysis saved to {output_path}")
        ui.display_complexity_summary(report) # Show summary

    except Exception as e:
        spinner.stop()
        log.exception("Failed to analyze task complexity.")
        ui.console.print(f"[bold red]Error:[/bold red] Failed to analyze complexity: {e}")
