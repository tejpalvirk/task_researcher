import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import asyncio

from . import config, utils, ai_services, dependency_manager, ui, models
from .models import TasksData, Task, Subtask, ComplexityReport, ResearchTopics
from .utils import log, read_json, write_json, task_exists, find_task_by_id, get_next_task_id, get_next_subtask_id, read_complexity_report, find_task_in_complexity_report, sanitize_filename, read_file, write_file, format_phase_for_filename
from rich.prompt import Confirm

from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs

# Lazy import STORM stuff
KnowledgeStormRunner = None
STORMWikiRunnerArguments = None
STORMWikiLMConfigs = None
LitellmModel = None
RM_CLASSES = {} # Dictionary to hold imported RM classes

def _import_storm_lazy():
    global KnowledgeStormRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs, LitellmModel, RM_CLASSES
    if KnowledgeStormRunner: return True
    try:
        from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments, STORMWikiLMConfigs
        from knowledge_storm.lm import LitellmModel
        from knowledge_storm import rm as storm_rm
        RM_CLASSES['bing'] = getattr(storm_rm, 'BingSearch', None)
        RM_CLASSES['you'] = getattr(storm_rm, 'YouRM', None)
        RM_CLASSES['google'] = getattr(storm_rm, 'GoogleSearch', None)
        RM_CLASSES['serper'] = getattr(storm_rm, 'SerperRM', None)
        RM_CLASSES['tavily'] = getattr(storm_rm, 'TavilySearchRM', None)
        RM_CLASSES['duckduckgo'] = getattr(storm_rm, 'DuckDuckGoSearchRM', None)
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

# --- Core Task Functions ---
async def parse_inputs(
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
         log.warning("Plan file (`plan.md`) is empty or missing. Task generation might lack phase information.")
         ui.console.print("[yellow]Warning:[/yellow] Plan file is empty or missing. Task generation might lack phase information.")

    ui.console.print(f"Calling AI to generate ~{num_tasks} tasks...")
    spinner = ui.console.status("[bold green]Generating tasks with AI...", spinner="dots")
    spinner.start()
    try:
        tasks_data = await ai_services.call_llm_for_tasks(
            functional_spec=func_spec,
            technical_spec=tech_spec,
            plan=plan,
            deep_research=research,
            num_tasks=num_tasks,
            source_files=source_files,
        )
        spinner.stop()

        if not tasks_data:
            raise Exception("AI failed to return valid task data.")

        log.info("Validating and fixing dependencies for newly generated tasks...")
        dep_made_changes, dep_fixes = dependency_manager.fix_dependencies(tasks_data)
        if dep_made_changes: log.info(f"Dependency fixes applied: {dep_fixes}")
        else: log.info("No dependency fixes needed.")

        tasks_data.meta.generatedAt = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(tasks_file_path, tasks_data.model_dump(mode='json', exclude_none=True))

        ui.console.print(f"[bold green]Success:[/bold green] Generated {len(tasks_data.tasks)} tasks and saved to {tasks_file_path}")
        generate_task_files(tasks_file_path, config.TASK_FILES_DIR) # Generate files automatically

    except Exception as e:
        spinner.stop()
        log.exception("Failed to parse inputs and generate tasks.")
        ui.console.print(f"[bold red]Error:[/bold red] Failed to generate tasks: {e}")

async def update_tasks(
    from_id: int,
    prompt: str,
    use_research: bool = False,
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
    ui.display_tasks_summary(tasks_to_update_models)

    if not Confirm.ask(f"Proceed with updating {len(tasks_to_update_models)} tasks?", default=True):
        ui.console.print("Update cancelled.")
        return

    ui.console.print(f"Calling AI to update tasks (Research hint: {use_research})...")
    spinner = ui.console.status("[bold green]Updating tasks with AI...", spinner="dots")
    spinner.start()
    try:
        updated_task_models = await ai_services.update_tasks_with_llm(
            tasks_to_update_models, prompt, use_research_prompt=use_research
        )
        spinner.stop()

        updated_count = 0
        if updated_task_models:
            updates_dict = {task.id: task for task in updated_task_models}
            for i, task in enumerate(tasks_data.tasks):
                if task.id in updates_dict:
                    original_subtasks = task.subtasks
                    tasks_data.tasks[i] = updates_dict[task.id]
                    if not tasks_data.tasks[i].subtasks and original_subtasks:
                         tasks_data.tasks[i].subtasks = [models.Subtask(**st.model_dump()) if isinstance(st, models.Subtask) else models.Subtask(**st) for st in original_subtasks]
                    elif tasks_data.tasks[i].subtasks:
                         tasks_data.tasks[i].subtasks = [models.Subtask(**st.model_dump()) if isinstance(st, models.Subtask) else models.Subtask(**st) for st in tasks_data.tasks[i].subtasks]
                    updated_count +=1

        if updated_count > 0:
            log.info("Validating and fixing dependencies after update...")
            dep_made_changes, dep_fixes = dependency_manager.fix_dependencies(tasks_data)
            if dep_made_changes: log.info(f"Dependency fixes applied after update: {dep_fixes}")

            write_json(tasks_file_path, tasks_data.model_dump(mode='json', exclude_none=True))
            ui.console.print(f"[bold green]Success:[/bold green] Updated {updated_count} tasks in {tasks_file_path}")
            generate_task_files(tasks_file_path, config.TASK_FILES_DIR)
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
    """Generates individual .txt files for each task with phase number in filename.""" # Updated docstring
    log.info(f"Generating task files in directory: {output_dir}")
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
        ui.console.print("[yellow]No tasks found to generate files for.[/yellow]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    log.info("Validating dependencies before generating files...")

    for task in tasks_data.tasks:
        # Use numeric phase for filename
        phase_prefix = format_phase_for_filename(task.phase) # Use helper
        file_path = output_dir / f"phase_{phase_prefix}_task_{task.id:03d}.txt" # Updated filename format
        content = ui.format_task_for_file(task, tasks_data.tasks)
        try:
            file_path.write_text(content, encoding='utf-8')
            generated_count += 1
            log.debug(f"Generated file: {file_path.name}")
        except Exception as e:
            log.error(f"Failed to write task file {file_path.name}: {e}")

    ui.console.print(f"[bold green]Success:[/bold green] Generated {generated_count} task files in {output_dir}")

# --- Research & Expansion Workflow ---

def _estimate_storm_tokens(topics: Dict[str, List[str]]) -> int:
    """Provides a *very* rough estimate of token usage for STORM research."""
    # Extremely simplified heuristic:
    # Assume each question triggers some conv simulation and article gen part.
    # This doesn't account for actual search result length, conversation depth, etc.
    num_questions = sum(len(q_list) for q_list in topics.values())
    num_topics = len(topics)

    # Estimate based on configured max tokens per component per topic/question
    # Assume conv_sim per question and article_gen per topic (very rough)
    estimated_tokens = (num_questions * config.STORM_MAX_TOKENS_CONV) + \
                       (num_topics * config.STORM_MAX_TOKENS_ARTICLE)

    # Add a buffer, as actual usage can vary wildly
    buffer_factor = 1.5
    total_estimate = int(estimated_tokens * buffer_factor)

    log.info(f"Rough STORM token estimate: {total_estimate} (based on {num_topics} topics, {num_questions} questions)")
    return total_estimate


def _get_storm_runner(search_top_k: int, max_conv_tokens: int, max_article_tokens: int) -> Optional[Any]:
    """Configures and returns a STORM runner instance."""
    if not _import_storm_lazy():
        return None

    # 1. LM Configs
    lm_configs = STORMWikiLMConfigs()
    big_llm_kwargs = {"model": config.BIG_STORM_MODEL, "temperature": config.BIG_STORM_TEMP}
    small_llm_kwargs = {"model": config.SMALL_STORM_MODEL, "temperature": config.SMALL_STORM_TEMP}

    # Allow overriding via specific STORM env vars if needed in the future
    # STORM models use LiteLLM model instances
    lm_powerful = LitellmModel(
        **big_llm_kwargs,
        max_tokens=max_article_tokens,
    )
    lm_fast = LitellmModel(
        **small_llm_kwargs,
        max_tokens=max_conv_tokens,
    )
    lm_configs.set_conv_simulator_lm(lm_fast)
    lm_configs.set_question_asker_lm(lm_fast)
    lm_configs.set_outline_gen_lm(lm_powerful)
    lm_configs.set_article_gen_lm(lm_powerful)
    lm_configs.set_article_polish_lm(lm_powerful)
    # Add embedding model config if needed for specific retrievers
    # if config.STORM_EMBEDDING_MODEL:
    #      embedding_model = LitellmModel(model=config.STORM_EMBEDDING_MODEL) # Assuming LitellmModel works for embeddings too
    #      lm_configs.set_embedding_model(embedding_model) # Check STORM's actual method name

    # 2. Retriever
    rm = None
    retriever_type = config.STORM_RETRIEVER
    RMClass = RM_CLASSES.get(retriever_type)

    if not RMClass:
        ui.console.print(f"[bold red]Error:[/bold red] Unsupported STORM retriever type configured: {retriever_type}")
        return None

    try:
        # Map config keys to STORM RM kwargs dynamically if possible
        # This reduces explicit checks for every retriever
        retriever_keys = {
            'bing': ('bing_search_api_key', config.BING_SEARCH_API_KEY),
            'you': ('ydc_api_key', config.YDC_API_KEY),
            'tavily': ('tavily_api_key', config.TAVILY_API_KEY),
            'serper': ('serper_api_key', config.SERPER_API_KEY),
            # Add others...
        }
        rm_kwargs = {'k': search_top_k}
        if retriever_type in retriever_keys:
            key_name, key_value = retriever_keys[retriever_type]
            if not key_value:
                 raise ValueError(f"{key_name.upper()} not set in .env for {retriever_type} retriever.")
            rm_kwargs[key_name] = key_value

        rm = RMClass(**rm_kwargs)
        log.info(f"Initialized STORM retriever: {retriever_type}")

    except ValueError as ve:
        ui.console.print(f"[bold red]Error:[/bold red] Missing API key for {retriever_type} retriever: {ve}")
        return None
    except Exception as e:
        ui.console.print(f"[bold red]Error:[/bold red] Failed to initialize {retriever_type} retriever: {e}")
        return None

    # 3. Engine Args
    engine_args = STORMWikiRunnerArguments(
        output_dir=config.STORM_OUTPUT_DIR,
        remove_intermediate_files=False, # Keep intermediate files for debugging
    )

    return STORMWikiRunner(engine_args, lm_configs, rm)


# NOTE: STORM is synchronous. Running it within an async function needs care.
# Using asyncio.to_thread allows running sync code in a separate thread
# without blocking the main async event loop.
async def run_storm_research_for_topic( # Changed to async, uses to_thread
    topic_name: str,
    questions: List[str],
    search_top_k: int = config.STORM_SEARCH_TOP_K,
    max_conv_tokens: int = config.STORM_MAX_TOKENS_CONV,
    max_article_tokens: int = config.STORM_MAX_TOKENS_ARTICLE,
) -> Optional[str]:
    """Runs the STORM pipeline for a specific topic and questions in a thread."""

    def storm_sync_runner():
        """Synchronous wrapper for STORM execution."""
        runner = _get_storm_runner(search_top_k, max_conv_tokens, max_article_tokens)
        if not runner:
            return None # Return None if setup fails

        storm_topic_input = f"{topic_name}: Research Summary focusing on:\n" + "\n".join(f"- {q}" for q in questions)
        storm_topic_filename_base = sanitize_filename(topic_name)

        log.info(f"Starting synchronous STORM run for topic: {topic_name}")
        try:
            runner.run(
                topic=storm_topic_input,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
            )

            # Find the generated article file
            expected_file_path = config.STORM_OUTPUT_DIR / f"{sanitize_filename(storm_topic_input)}.md"
            fallback_file_path = config.STORM_OUTPUT_DIR / f"{storm_topic_filename_base}.md"
            output_file_to_read = None

            if expected_file_path.exists():
                output_file_to_read = expected_file_path
            elif fallback_file_path.exists():
                output_file_to_read = fallback_file_path
            else:
                md_files = list(config.STORM_OUTPUT_DIR.glob(f"{storm_topic_filename_base}*.md"))
                if md_files:
                    output_file_to_read = max(md_files, key=lambda p: p.stat().st_mtime)

            if output_file_to_read:
                log.info(f"Found STORM output at: {output_file_to_read}")
                return read_file(output_file_to_read)
            else:
                log.error(f"STORM finished for topic '{topic_name}', but couldn't find the output markdown file.")
                return "" # Return empty string if file not found

        except Exception as e:
            log.exception(f"STORM synchronous run failed for topic '{topic_name}'.")
            # Re-raise or handle appropriately? For now, log and return None.
            return None # Indicate failure

    # Run the synchronous STORM function in a separate thread
    ui.console.print(f"🌪️ Running STORM research for topic: [bold cyan]{topic_name}[/bold cyan]...")
    article_content = await asyncio.to_thread(storm_sync_runner)

    if article_content is None: # Check for None indicating an error during run
         ui.console.print(f"  ❌ Failed research for '{topic_name}'. Check logs.")
    elif not article_content: # Empty string means file not found after run
         ui.console.print(f"  ⚠️ STORM ran for '{topic_name}', but output file not found.")
    else:
         ui.console.print(f"  ✅ Completed research for '{topic_name}'.")

    return article_content


async def expand_task(
    task_id: int,
    num_subtasks: Optional[int] = None,
    use_research: bool = False,
    prompt: Optional[str] = None,
    force: bool = False,
    tasks_file_path: Path = config.TASKS_FILE_PATH,
) -> None:
    """Expands a single task into subtasks, optionally using STORM research with confirmation."""
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
        tasks_data.tasks[task_index].subtasks = [] # Clear existing subtasks before proceeding

    # --- Determine parameters ---
    complexity_report = read_complexity_report()
    task_analysis = find_task_in_complexity_report(complexity_report, task.id) if complexity_report else None
    final_num_subtasks = num_subtasks if num_subtasks is not None else config.DEFAULT_SUBTASKS
    final_context = prompt or ""

    if task_analysis:
        log.info(f"Using complexity analysis for task {task.id} (Score: {task_analysis.get('complexityScore', 'N/A')})")
        if num_subtasks is None and task_analysis.get('recommendedSubtasks'):
            final_num_subtasks = task_analysis['recommendedSubtasks']
            log.info(f"Using recommended number of subtasks: {final_num_subtasks}")
        if not prompt and task_analysis.get('expansionPrompt'):
            if not use_research: # Only use analysis prompt if not doing STORM research
                final_context = task_analysis['expansionPrompt']
                log.info("Using expansion prompt from complexity report.")
            else:
                 log.info("Ignoring complexity report prompt because --research (STORM) is enabled.")

    ui.console.print(f"\nPreparing to expand task {task.id} '{task.title}' into {final_num_subtasks} subtasks...")

    # --- Research Workflow (if --research) ---
    storm_research_summary = ""
    if use_research:
        ui.console.print("🔬 [bold cyan]Research Phase Initiated[/bold cyan]")
        research_spinner = ui.console.status("[cyan]Generating research questions...", spinner="dots")
        research_spinner.start()
        try:
            # 1. Generate Questions
            questions = await ai_services.generate_research_questions(task)
            if not questions: raise Exception("Failed to generate research questions.")
            research_spinner.update("[cyan]Grouping questions into topics...")
            ui.console.print(f"  💬 Generated {len(questions)} research questions.")

            # 2. Group into Topics
            topics_dict = await ai_services.group_questions_into_topics(questions, task)
            if not topics_dict:
                topics_dict = {"General Research": questions}
                log.warning("Failed to group questions; using single topic.")
                ui.console.print("[yellow]  ⚠️ Could not group questions; proceeding with single topic.[/yellow]")
            research_spinner.stop()
            ui.console.print(f"  📑 Grouped into {len(topics_dict)} topics.")

            # 3. Estimate Tokens & Confirm with User
            estimated_tokens = _estimate_storm_tokens(topics_dict)
            ui.display_research_plan(topics_dict, estimated_tokens) # Show plan and estimate

            if not Confirm.ask(f"\nProceed with STORM research for {len(topics_dict)} topics?", default=True):
                ui.console.print("[yellow]Research cancelled by user.[/yellow] Proceeding to generate subtasks without research.")
                storm_research_summary = "" # Ensure empty
            else:
                # 4. Run STORM for each topic (now uses asyncio.to_thread)
                storm_outputs = []
                topic_count = len(topics_dict)
                ui.console.print("\n--- Starting STORM Research Runs ---")
                for i, (topic_name, topic_questions) in enumerate(topics_dict.items()):
                    ui.console.rule(f"[bold]Topic {i+1}/{topic_count}: {topic_name}[/]")
                    # Await the thread running the sync STORM code
                    storm_output = await run_storm_research_for_topic(topic_name, topic_questions)
                    if storm_output is not None: # Check for None (error) vs "" (file not found)
                        storm_outputs.append(f"## Research Topic: {topic_name}\n\n{storm_output}\n\n---\n")
                    # Status already printed by run_storm_research_for_topic

                # 5. Combine results
                if storm_outputs:
                    storm_research_summary = "\n".join(storm_outputs)
                    ui.console.print("\n✅ [bold green]Research Phase Complete. Combined results.[/bold green]")
                    # Optionally save combined research to a file?
                    research_file = config.STORM_OUTPUT_DIR / f"task_{task.id}_research_summary.md"
                    write_file(research_file, storm_research_summary)
                    ui.console.print(f"  Combined research saved to: {research_file}")
                else:
                     ui.console.print("\n⚠️ [bold yellow]Research Phase completed with no successful STORM outputs.[/bold yellow]")

        except Exception as research_error:
            research_spinner.stop() # Ensure spinner stops on error
            log.exception(f"Research phase failed for task {task_id}: {research_error}")
            ui.console.print(f"\n❌ [bold red]Research Phase Failed:[/bold red] {research_error}")
            ui.console.print("Proceeding to generate subtasks without STORM research.")
            storm_research_summary = "" # Ensure empty on failure

    # --- Generate Subtasks ---
    combined_context = f"{storm_research_summary}\n\nUser Provided Context:\n{final_context}".strip()

    ui.console.print(f"\n🧠 [bold]Generating {final_num_subtasks} Subtasks...[/bold]")
    spinner = ui.console.status("[bold green]Generating subtasks with AI...", spinner="dots")
    spinner.start()
    try:
        next_sub_id = get_next_subtask_id(tasks_data.tasks[task_index].model_dump())
        new_subtasks = await ai_services.generate_subtasks_with_llm(
            tasks_data.tasks[task_index], final_num_subtasks, next_sub_id, combined_context
        )
        spinner.stop()

        if new_subtasks:
            tasks_data.tasks[task_index].subtasks.extend(
                [Subtask(**sub.model_dump()) for sub in new_subtasks]
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
    """Expands all eligible pending tasks, with confirmation for research."""
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
    for task in tasks_data.tasks:
        if task.status != "done":
            if not task.subtasks or force:
                tasks_to_expand_models.append(task)
            else:
                 log.info(f"Skipping task {task.id} (already has subtasks and --force not used).")

    if not tasks_to_expand_models:
        ui.console.print("[yellow]Info:[/yellow] No tasks eligible for expansion found.")
        return

    ui.console.print(f"Found {len(tasks_to_expand_models)} tasks eligible for expansion.")
    ui.display_tasks_summary(tasks_to_expand_models)

    complexity_report = read_complexity_report()
    if complexity_report:
        complexity_map = {analysis.get('taskId'): analysis.get('complexityScore', 0)
                          for analysis in complexity_report.get('complexityAnalysis', [])}
        tasks_to_expand_models.sort(key=lambda t: complexity_map.get(t.id, 0), reverse=True)
        ui.console.print("[italic]Tasks sorted by complexity (highest first).[/italic]")
    else:
        tasks_to_expand_models.sort(key=lambda t: t.id)

    research_confirmation_needed = use_research and _import_storm_lazy()
    if research_confirmation_needed:
         ui.console.print(f"\n[bold yellow]Research Mode Enabled:[/bold yellow] This will run the STORM research workflow for each of the {len(tasks_to_expand_models)} tasks.")
         ui.console.print("This involves multiple AI calls per task (questions, topics, STORM runs) and can take significant time and tokens.")
         # Estimate total tokens for ALL tasks - very rough
         # Assume complexity report helps estimate subtasks per task, else use default
         total_est_tokens = 0
         total_est_topics = 0
         for task in tasks_to_expand_models:
              task_analysis = find_task_in_complexity_report(complexity_report, task.id) if complexity_report else None
              num_q_est = 7 # Rough guess for questions per task
              num_topics_est = 3 # Rough guess for topics per task
              # A simple heuristic based on complexity score?
              # if task_analysis:
              #     num_q_est = int(task_analysis.get('complexityScore', 5) * 1.5)
              #     num_topics_est = max(2, int(task_analysis.get('complexityScore', 5) / 2))

              est_q_tokens = num_q_est * config.STORM_MAX_TOKENS_CONV # Questions lead to conv sim
              est_a_tokens = num_topics_est * config.STORM_MAX_TOKENS_ARTICLE # Topics lead to article gen
              total_est_tokens += int((est_q_tokens + est_a_tokens) * 1.5) # Add buffer
              total_est_topics += num_topics_est

         ui.console.print(f"Rough total estimated STORM tokens: ~[bold]{total_est_tokens:,}[/bold] across ~{total_est_topics} topics.")

    if not Confirm.ask(f"\nProceed with expanding {len(tasks_to_expand_models)} tasks?", default=True):
        ui.console.print("Expansion cancelled.")
        return

    expanded_count = 0
    failed_count = 0
    ui.console.print(f"\n--- Starting Bulk Expansion for {len(tasks_to_expand_models)} Tasks ---")

    for i, task_model in enumerate(tasks_to_expand_models):
        ui.console.print(f"\n[{i+1}/{len(tasks_to_expand_models)}] Expanding Task {task_model.id}: '{task_model.title}'...")
        try:
            # Call the async single expand function
            await expand_task(
                 task_id=task_model.id,
                 num_subtasks=num_subtasks,
                 use_research=use_research, # Pass the flag
                 prompt=prompt,
                 force=force,
                 tasks_file_path=tasks_file_path
             )
            expanded_count += 1
            # Re-read data in case expand_task modified it - needed if not passing data around
            tasks_data_dict = read_json(tasks_file_path)
            tasks_data = models.TasksData.model_validate(tasks_data_dict)

        except Exception as e:
            log.error(f"Expansion failed for task {task_model.id} in bulk operation: {e}")
            failed_count += 1
            # Potentially add retry logic here if desired

    ui.console.print(f"\n--- Bulk Expansion Complete ---")
    ui.console.print(f"[bold green]Successfully attempted expansion for {expanded_count} tasks.[/bold green]")
    if failed_count > 0:
        ui.console.print(f"[bold red]Failed during expansion process for {failed_count} tasks.[/bold red] Check logs for details.")
    # Final dependency check and file generation happened within each expand_task call

async def analyze_complexity(
    output_path: Optional[Path] = None,
    use_research: bool = False, # Research HINT only
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
                "usedResearchHint": use_research,
                "llm_model": config.LLM_MODEL,
            },
            complexityAnalysis=sorted(analysis_results, key=lambda x: x.complexityScore, reverse=True)
        )

        write_json(output_path, report.model_dump(mode='json', exclude_none=True))
        ui.console.print(f"[bold green]Success:[/bold green] Complexity analysis saved to {output_path}")
        ui.display_complexity_summary(report)

    except Exception as e:
        spinner.stop()
        log.exception("Failed to analyze task complexity.")
        ui.console.print(f"[bold red]Error:[/bold red] Failed to analyze complexity: {e}")
