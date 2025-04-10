import instructor
from litellm import completion as litellm_completion, acompletion as litellm_acompletion
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel
import litellm
import json
import re

from . import config, utils
from .models import (
    Task, Subtask, TasksData, TaskFileMetadata, # Added TaskFileMetadata
    ComplexityAnalysisItem, ComplexityReport,
    ResearchQuestions, ResearchTopics
)
from .utils import log

# --- Instructor Client Setup ---
# Use JSON mode for better compatibility, especially with complex nested models
client = instructor.from_litellm(litellm_completion, mode=instructor.Mode.JSON)
# async_client = instructor.from_litellm(litellm_acompletion, mode=instructor.Mode.JSON) # If needed

# --- Helper Function ---
def _get_llm_kwargs(
    max_tokens_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    model_override: Optional[str] = None
) -> Dict[str, Any]:
    """Returns common kwargs for LLM calls via litellm."""
    return {
        "model": model_override or config.LLM_MODEL,
        "max_tokens": max_tokens_override or config.MAX_TOKENS,
        "temperature": temperature_override if temperature_override is not None else config.TEMPERATURE,
        "response_format": {"type": "json_object"} # Request JSON mode
    }

# --- Core AI Functions ---

async def call_llm_with_instructor(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    max_retries: int = 2,
    **kwargs # Passthrough for model, temp, max_tokens etc.
) -> Optional[BaseModel]:
    """Generic function to call LLM using Instructor."""
    llm_kwargs = _get_llm_kwargs(**kwargs)
    log.debug(f"Calling LLM with Instructor: Model={llm_kwargs['model']}, ResponseModel={response_model.__name__}")
    try:
        # Add a validation context to help Instructor recover if parsing fails initially
        response = client.chat.completions.create(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context={"context": "Trying to parse LLM response."},
            **llm_kwargs,
        )
        return response
    except Exception as e:
        log.exception(f"Error calling LLM via Instructor for {response_model.__name__}: {e}")
        # Consider adding more specific exception handling if needed
        # e.g., handling API errors, validation errors from Instructor
        return None


async def call_llm_for_tasks(
    functional_spec: str,
    technical_spec: str,
    plan: str,
    deep_research: str,
    num_tasks: int,
    source_files: List[str],
) -> Optional[TasksData]:
    """Calls LLM to generate tasks from input specifications."""
    log.info(f"Generating ~{num_tasks} tasks from input files using {config.LLM_MODEL}...")

    # Updated prompt for numeric phase
    system_prompt = f"""You are an AI assistant helping to break down project specifications into sequential development tasks.
Your goal is to create approximately {num_tasks} well-structured, actionable development tasks based on the provided plan and specifications.
Assign a relevant numeric 'phase' (e.g., 1, 2, 3) to each task based on the structure outlined in the `plan.md` content. Look for numbered phases or sections in the plan. If no numeric phase is identifiable, use `null`.

Each task should have the following fields, inferred from the inputs:
- id: (sequential integer starting from 1)
- title: (string)
- phase: (integer or null) - The numeric phase (e.g., 1, 2) derived from the plan.
- description: (string)
- details: (string - implementation details)
- status: "pending"
- dependencies: List[int] (IDs of tasks this depends on, ensure they exist and are lower IDs)
- priority: "high" | "medium" | "low"
- testStrategy: (string - validation approach)
- subtasks: [] (initially empty)

Guidelines:
1. Generate roughly {num_tasks} tasks, numbered sequentially starting from 1. Assign IDs correctly.
2. Base tasks on the high-level plan provided, breaking down each planned item and associating it with its corresponding phase NUMBER found in the plan.
3. Use the functional spec, technical spec, and deep research for details, implementation guidance, and test strategies.
4. Ensure tasks are atomic and focus on a single responsibility.
5. Order tasks logically, respecting phases in the plan and technical dependencies. Prioritize setup and core functionality.
6. Assign appropriate dependencies (only tasks with lower IDs). Ensure no circular dependencies.
7. Assign priority based on the plan and criticality.
8. Include detailed implementation guidance in "details" and a clear validation approach in "testStrategy".
"""

    # Ensure plan formatting is preserved in the user prompt
    user_content = f"""Please generate approximately {num_tasks} development tasks based on the following project inputs:

## High-Level Plan (`plan.md`)
```markdown
{plan}
```

## Functional Specification (`functional_spec.md`)
```markdown
{functional_spec}
```

## Technical Specification (`technical_spec.md`)
```markdown
{technical_spec}
```

## Background Research (`deep_research.md`)
```markdown
{deep_research}
```

Generate the tasks. Ensure task IDs are sequential starting from 1, dependencies are valid, and each task has a relevant numeric 'phase' derived from the Plan section (or null if none).
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    response = await call_llm_with_instructor(messages, TasksData)

    if response:
        log.info(f"Successfully generated {len(response.tasks)} tasks.")
        response.meta.sourceFiles = source_files
        response.meta.totalTasks = len(response.tasks)
        # Post-process IDs, dependencies, phase, status
        current_max_id = 0
        processed_tasks = []
        for i, task_model in enumerate(response.tasks):
             # Validate with Pydantic model before adding
             try:
                task = Task(**task_model.model_dump()) # Includes phase validation
                task.id = i + 1
                task.dependencies = [dep for dep in task.dependencies if isinstance(dep, int) and 0 < dep < task.id]
                task.status = "pending"
                task.subtasks = []
                # Phase validation already happened in Pydantic model
                processed_tasks.append(task)
                current_max_id = task.id
             except Exception as val_err:
                 log.warning(f"Skipping task due to validation error: {val_err}. Original data: {task_model.model_dump()}")

        response.tasks = processed_tasks # Replace with validated tasks
        response.meta.totalTasks = current_max_id
        if not response.meta.projectName: response.meta.projectName = config.PROJECT_NAME
        if not response.meta.projectVersion: response.meta.projectVersion = config.PROJECT_VERSION

        return response
    else:
        log.error("LLM failed to generate tasks.")
        return None


async def generate_research_questions(task: Task) -> Optional[List[str]]:
    log.info(f"Generating research questions for task {task.id}...")
    system_prompt = """You are an AI research assistant helping to break down complex software tasks.
Given a task description, identify key areas requiring further research before implementation or subtask creation.
Focus on technical challenges, unknown implementation details, best practices, relevant libraries/tools, or potential design choices.
Generate a concise list of specific, actionable research questions.
"""
    user_content = f"""Analyze the following task and generate 5-10 specific research questions to guide its breakdown and implementation:

**Task ID:** {task.id}
**Title:** {task.title}
**Phase:** {task.phase or 'N/A'}
**Description:** {task.description or 'N/A'}
**Details:**
```
{task.details or 'No details provided.'}
```
**Test Strategy:**
```
{task.testStrategy or 'No test strategy provided.'}
```
Generate the list of research questions.
"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    response = await call_llm_with_instructor(messages, ResearchQuestions)
    return response.questions if response else None

async def group_questions_into_topics(questions: List[str], task: Task) -> Optional[Dict[str, List[str]]]:
    if not questions: return None
    log.info(f"Grouping {len(questions)} research questions into topics for task {task.id}...")
    system_prompt = """You are an AI assistant skilled at organizing information.
Given a list of research questions related to a software task, group them into 2-4 high-level, logical research topics.
Each topic should represent a distinct area of investigation needed for the task. Assign each question to exactly one topic.
Ensure the keys are descriptive topic names and the values are lists of the original question strings.
"""
    question_list_str = "\n".join([f"- {q}" for q in questions])
    user_content = f"""Group the following research questions related to the task "{task.title}" (Phase {task.phase or 'N/A'}) into 2-4 logical topics:

**Research Questions:**
{question_list_str}

**Original Task Context:**
Title: {task.title}
Description: {utils.truncate(task.description, 100)}

Generate the topics and their corresponding questions.
"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    response = await call_llm_with_instructor(messages, ResearchTopics)
    return response.topics if response else None

async def generate_subtasks_with_llm(
    parent_task: Task,
    num_subtasks: int,
    next_subtask_id: int,
    additional_context: str = "",
) -> Optional[List[Subtask]]:
    log.info(f"Generating {num_subtasks} subtasks for task {parent_task.id} using {config.LLM_MODEL}...")
    has_research_context = bool(additional_context)

    system_prompt = f"""You are an AI assistant expert in breaking down software development tasks.
Your goal is to generate {num_subtasks} specific, actionable subtasks for the given parent task.
{'You have been provided with research findings or context relevant to this task. Use this information extensively to inform the subtask details, implementation guidance, technical considerations, and acceptance criteria.' if has_research_context else ''}

Subtasks should include:
- id: (sequential integer starting from {next_subtask_id})
- title: (string)
- description: (string)
- details: (string - implementation guidance, referencing research if applicable)
- acceptanceCriteria: (string - how to verify this subtask is done)
- status: "pending"
- dependencies: List[int] (IDs of *sibling* subtasks generated in this batch, starting from {next_subtask_id})

Guidelines:
1. Create {num_subtasks} specific, actionable implementation steps.
2. Ensure a logical sequence for implementation.
3. Collectively cover the parent task's requirements, incorporating context/research.
4. Provide clear implementation guidance in 'details'.
5. Define clear 'acceptanceCriteria' for each subtask.
6. Define dependencies between subtasks using their sequential IDs (starting from {next_subtask_id}). Use `[]` for no dependencies.
"""
    user_content = f"""Please break down the following parent task into exactly {num_subtasks} specific, actionable subtasks:

**Parent Task ID:** {parent_task.id}
**Parent Task Title:** {parent_task.title}
**Parent Task Phase:** {parent_task.phase or 'N/A'}
**Parent Task Description:** {parent_task.description or 'N/A'}
**Parent Task Details:**
```
{parent_task.details or 'No details provided.'}
```
**Parent Task Test Strategy:**
```
{parent_task.testStrategy or 'No test strategy provided.'}
```

**{'Research Context & Findings:' if has_research_context else 'Additional Context:'}**
```
{additional_context or 'None provided.'}
```

Generate a list of {num_subtasks} subtask objects. Assign IDs sequentially starting from {next_subtask_id}. Define dependencies relative to these new IDs. Include 'acceptanceCriteria' for each.
"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    max_tokens = config.MAX_TOKENS * 2 if has_research_context else config.MAX_TOKENS

    response = await call_llm_with_instructor(messages, List[Subtask], max_tokens_override=max_tokens)

    if response:
        log.info(f"Successfully generated {len(response)} subtasks for task {parent_task.id}.")
        # Post-process IDs and dependencies (same as before)
        validated_subtasks = []
        for i, subtask_model in enumerate(response):
             subtask = Subtask(**subtask_model.model_dump())
             subtask.id = next_subtask_id + i
             subtask.status = "pending"
             valid_deps = []
             for dep in subtask.dependencies:
                 try:
                     dep_id = int(dep)
                     if next_subtask_id <= dep_id < subtask.id: valid_deps.append(dep_id)
                     else: log.warning(f"Subtask {subtask.id} (parent {parent_task.id}) has invalid dependency {dep_id}. Removing.")
                 except (ValueError, TypeError): log.warning(f"Subtask {subtask.id} (parent {parent_task.id}) has non-integer dependency '{dep}'. Removing.")
             subtask.dependencies = valid_deps
             subtask.acceptanceCriteria = str(subtask.acceptanceCriteria) if subtask.acceptanceCriteria is not None else None
             validated_subtasks.append(subtask)
        if len(validated_subtasks) != num_subtasks:
             log.warning(f"LLM generated {len(validated_subtasks)} subtasks, expected {num_subtasks}.")
        return validated_subtasks
    else:
        log.error(f"LLM failed to generate subtasks for task {parent_task.id}.")
        return None

async def analyze_task_complexity_with_llm(
    tasks_data: TasksData,
    use_research_prompt: bool = False,
) -> Optional[List[ComplexityAnalysisItem]]:
    log.info(f"Analyzing complexity for {len(tasks_data.tasks)} tasks using {config.LLM_MODEL}...")
    research_guidance = """
Leverage your knowledge of software engineering best practices, potential complexities, and common implementation patterns related to each task's domain.
Provide insightful reasoning for your complexity score and tailored prompts for effective subtask breakdown.
""" if use_research_prompt else ""

    system_prompt = f"""You are an expert software architect analyzing task complexity. Assess each task based on ambiguity, technical difficulty, dependencies, and scope.

For each task, provide:
- taskId: The original task ID.
- taskTitle: The original task title.
- complexityScore: Float score from 1.0 (simple) to 10.0 (very complex).
- recommendedSubtasks: Integer number of subtasks (e.g., 3-7) appropriate for the complexity.
- expansionPrompt: A concise, specific prompt to guide an AI in generating high-quality subtasks for *this specific task*.
- reasoning: Brief justification for the complexity score.
{research_guidance}
"""
    tasks_input_str = "\n---\n".join([
        f"Task ID: {task.id}\nTitle: {task.title}\nPhase: {task.phase or 'N/A'}\nDescription: {utils.truncate(task.description, 150)}\nDetails: {utils.truncate(task.details, 200)}\nDependencies: {task.dependencies or []}\nPriority: {task.priority or ''}"
        for task in tasks_data.tasks
    ])
    user_content = f"""Please analyze the complexity of the following tasks:

{tasks_input_str}

Generate the analysis for each task.
"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    response = await call_llm_with_instructor(messages, List[ComplexityAnalysisItem], max_tokens_override=config.MAX_TOKENS * 2)

    if response:
        log.info(f"Successfully generated complexity analysis for {len(response)} tasks.")
        # Validation logic remains the same
        analyzed_ids = {item.taskId for item in response}
        input_ids = {task.id for task in tasks_data.tasks}
        missing_ids = input_ids - analyzed_ids
        if missing_ids: log.warning(f"Missing complexity analysis for task IDs: {missing_ids}")
        extra_ids = analyzed_ids - input_ids
        if extra_ids:
            log.warning(f"Extra complexity analysis for unknown task IDs: {extra_ids}. Filtering.")
            response = [item for item in response if item.taskId in input_ids]
        for item in response:
             item.complexityScore = max(1.0, min(10.0, item.complexityScore))
             item.recommendedSubtasks = max(1, item.recommendedSubtasks)
        return response
    else:
        log.error("LLM failed to generate complexity analysis.")
        return None

async def update_tasks_with_llm(
    tasks_to_update: List[Task],
    update_prompt: str,
    use_research_prompt: bool = False,
) -> Optional[List[Task]]:
    if not tasks_to_update: return []
    log.info(f"Updating {len(tasks_to_update)} tasks using {config.LLM_MODEL}...")
    research_guidance = """
Leverage your knowledge of current best practices and implementation details related to the requested update. Ensure the tasks remain coherent and technically sound after the changes.
""" if use_research_prompt else ""

    system_prompt = f"""You are an AI assistant helping to update software development tasks based on new context or requirements changes.
Update the tasks provided based on the user's prompt. Modify ONLY the necessary fields (`title`, `description`, `details`, `testStrategy`, potentially `phase`).
Do NOT change `id`, `status`, `dependencies`, `priority`, or existing `subtasks` unless explicitly instructed.
Apply the changes consistently across all relevant tasks. Ensure updated tasks remain actionable and clear.
{research_guidance}
"""
    # Include phase in input context
    tasks_input_str = json.dumps(
        [t.model_dump(exclude={'subtasks'}) for t in tasks_to_update],
        indent=2
    )
    user_content = f"""Here are the tasks to update:
```json
{tasks_input_str}
```
Please update these tasks based on the following new context or requirement change:
```
{update_prompt}
```
Return the updated task objects for ALL the tasks listed above. Preserve fields like `id`, `status`, `dependencies`, `priority`, and existing `subtasks` unless the update prompt specifically instructs otherwise. If the prompt implies a phase change, update the numeric `phase` field.
"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    response = await call_llm_with_instructor(messages, List[Task], max_tokens_override=config.MAX_TOKENS * 2)

    if response:
        log.info(f"Successfully received updated task data for {len(response)} tasks.")
        # Validation and merging logic (ensure phase is handled correctly)
        original_tasks_map = {t.id: t for t in tasks_to_update}
        final_updated_tasks = []
        processed_ids = set()

        for updated_task_model in response:
            try:
                # Validate incoming data and ensure phase is int/None
                updated_task = Task(**updated_task_model.model_dump())
            except Exception as val_err:
                log.warning(f"Skipping update for task ID {updated_task_model.id} due to validation error: {val_err}")
                continue

            original_task = original_tasks_map.get(updated_task.id)
            if not original_task:
                log.warning(f"LLM returned update for unknown task ID {updated_task.id}. Skipping.")
                continue

            # Preserve fields
            updated_task.id = original_task.id
            updated_task.status = original_task.status
            updated_task.dependencies = original_task.dependencies
            updated_task.priority = original_task.priority
            updated_task.subtasks = original_task.subtasks
            # Phase is handled by Pydantic validator during Task instantiation

            final_updated_tasks.append(updated_task)
            processed_ids.add(updated_task.id)

        missed_tasks = [t for t in tasks_to_update if t.id not in processed_ids]
        if missed_tasks:
            log.warning(f"LLM did not return updates for {len(missed_tasks)} tasks. Keeping original versions.")
            final_updated_tasks.extend(missed_tasks)

        final_updated_tasks.sort(key=lambda t: tasks_to_update.index(original_tasks_map[t.id]) if t.id in original_tasks_map else float('inf'))
        return final_updated_tasks
    else:
        log.error("LLM failed to update tasks.")
        return None