import instructor
from litellm import completion as litellm_completion, acompletion as litellm_acompletion
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel
import litellm
import json # Added for potential fallback parsing
import re # Added for fallback parsing

from . import config, utils
from .models import (
    Task, Subtask, TasksData,
    ComplexityAnalysisItem, ComplexityReport,
    ResearchQuestions, ResearchTopics # Added research models
)
from .utils import log

# Configure litellm logging if needed
# litellm.set_verbose = config.DEBUG

# --- Instructor Client Setup ---
# Use the primary LLM model for most operations
client = instructor.from_litellm(litellm_completion, mode=instructor.Mode.JSON) # Ensure JSON mode
# async_client = instructor.from_litellm(litellm_acompletion, mode=instructor.Mode.JSON) # If needed

# --- Helper Function ---
def _get_llm_kwargs(
    max_tokens_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    model_override: Optional[str] = None
) -> Dict[str, Any]:
    """Returns common kwargs for LLM calls via litellm."""
    # litellm handles API keys via environment variables by default
    return {
        "model": model_override or config.LLM_MODEL,
        "max_tokens": max_tokens_override or config.MAX_TOKENS,
        "temperature": temperature_override if temperature_override is not None else config.TEMPERATURE,
        # litellm automatically selects provider based on model string prefix
        # and uses corresponding env vars (e.g., ANTHROPIC_API_KEY for "claude-...")
        "response_format": {"type": "json_object"} # Request JSON output where supported
    }

def _attempt_json_extraction(raw_text: str) -> Optional[Any]:
    """Tries different ways to extract JSON from potentially messy LLM output."""
    log.debug(f"Attempting JSON extraction from raw text (first 200 chars): {raw_text[:200]}")

    # 1. Check for markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        log.debug("Found JSON in markdown block.")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse JSON from markdown block: {e}")
            # Continue trying other methods

    # 2. Look for top-level JSON object or array
    json_str = raw_text.strip()
    if json_str.startswith('{') and json_str.endswith('}'):
        log.debug("Attempting to parse as JSON object.")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse as direct JSON object: {e}")
    elif json_str.startswith('[') and json_str.endswith(']'):
        log.debug("Attempting to parse as JSON array.")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse as direct JSON array: {e}")

    # 3. More aggressive extraction if it looks like JSON
    start_index = -1
    end_index = -1
    if '{' in json_str:
        start_index = json_str.find('{')
    if '[' in json_str and (start_index == -1 or json_str.find('[') < start_index):
         start_index = json_str.find('[')

    if start_index != -1:
         brace_balance = 0
         bracket_balance = 0
         in_string = False
         start_char = json_str[start_index]
         end_char_expected = '}' if start_char == '{' else ']'

         for i in range(start_index, len(json_str)):
             char = json_str[i]
             if char == '"': # Basic string detection
                 # Could add escape handling if needed
                 in_string = not in_string
             elif not in_string:
                  if char == '{': brace_balance += 1
                  elif char == '}': brace_balance -= 1
                  elif char == '[': bracket_balance += 1
                  elif char == ']': bracket_balance -= 1

             # Check if we found the end corresponding to the start
             if start_char == '{' and char == '}' and brace_balance == 0:
                  end_index = i
                  break
             elif start_char == '[' and char == ']' and bracket_balance == 0:
                  end_index = i
                  break

         if end_index != -1:
              potential_json = json_str[start_index : end_index + 1]
              log.debug(f"Aggressively extracted potential JSON (indices {start_index}-{end_index}).")
              try:
                  return json.loads(potential_json)
              except json.JSONDecodeError as e:
                   log.warning(f"Failed to parse aggressively extracted JSON: {e}")


    log.error("Failed to extract valid JSON from the response.")
    return None


async def call_llm_with_instructor(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    max_retries: int = 2,
    **kwargs # Passthrough for model, temp, max_tokens etc.
) -> Optional[BaseModel]:
    """Generic function to call LLM using Instructor."""
    llm_kwargs = _get_llm_kwargs(**kwargs)
    log.debug(f"Calling LLM with Instructor: Model={llm_kwargs['model']}, ResponseModel={response_model.__name__}, MaxTokens={llm_kwargs['max_tokens']}")
    try:
        response = client.chat.completions.create(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **llm_kwargs,
        )
        return response
    except Exception as e:
        log.exception(f"Error calling LLM via Instructor for {response_model.__name__}: {e}")
        # Check for specific errors like context length, API errors etc. if needed
        # Example using litellm exception types if available:
        # if isinstance(e, litellm.exceptions.ContextWindowExceededError):
        #    log.error("Context window exceeded for the request.")
        # elif isinstance(e, litellm.exceptions.APIError):
        #    log.error(f"API Error: {e.status_code} - {e.message}")
        return None # Return None on failure


# --- Core AI Functions ---

async def call_llm_for_tasks(
    functional_spec: str,
    technical_spec: str,
    plan: str,
    background: str,
    num_tasks: int,
    source_files: List[str],
) -> Optional[TasksData]:
    """Calls LLM to generate tasks from input specifications."""
    log.info(f"Generating ~{num_tasks} tasks from input files using {config.LLM_MODEL}...")
    system_prompt = f"""You are an AI assistant helping to break down project specifications into a set of sequential development tasks.
Your goal is to create approximately {num_tasks} well-structured, actionable development tasks based on the provided specifications and plan.

Each task should follow this Pydantic model structure:
```python
class Task(BaseModel):
    id: int
    title: str
    description: Optional[str]
    details: Optional[str]
    status: Literal["pending", ...] = "pending"
    dependencies: List[Union[int, str]] = []
    priority: Literal["high", "medium", "low"] = "{config.DEFAULT_PRIORITY}"
    testStrategy: Optional[str]
    subtasks: List[Subtask] = [] # Initially empty
```
Ensure the final output is a single JSON object conforming to the `TasksData` model:
```python
class TasksData(BaseModel):
    meta: TaskFileMetadata
    tasks: List[Task]
```
Guidelines:
1. Generate roughly {num_tasks} tasks, numbered sequentially starting from 1. Assign IDs correctly.
2. Base tasks on the high-level plan provided, breaking down each planned item.
3. Use the functional spec, technical spec, and background for details, implementation guidance, and test strategies.
4. Ensure tasks are atomic and focus on a single responsibility.
5. Order tasks logically, respecting phases in the plan and technical dependencies. Prioritize setup and core functionality.
6. Assign appropriate dependencies (only tasks with lower IDs). Ensure no circular dependencies.
7. Assign priority based on the plan and criticality.
8. Include detailed implementation guidance in "details" and a clear validation approach in "testStrategy".

Respond ONLY with the valid JSON object conforming to the TasksData model. Do not include ```json markdown.
"""
    user_content = f"""Please generate approximately {num_tasks} development tasks based on the following project inputs:

## High-Level Plan (`plan.md`):
```
{plan}
```

## Functional Specification (`functional_spec.md`):
```
{functional_spec}
```

## Technical Specification (`technical_spec.md`):
```
{technical_spec}
```

## Background Research (`background.md`):
```
{background}
```

Generate the tasks as a single JSON object conforming to the `TasksData` model. Ensure task IDs are sequential starting from 1, and dependencies are valid.
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
        # Post-process IDs and dependencies
        current_max_id = 0
        for i, task in enumerate(response.tasks):
            task.id = i + 1
            task.dependencies = [dep for dep in task.dependencies if isinstance(dep, int) and 0 < dep < task.id]
            task.status = "pending"
            task.subtasks = []
            current_max_id = task.id
        response.meta.totalTasks = current_max_id
        return response
    else:
        log.error("LLM failed to generate tasks.")
        return None

async def generate_research_questions(task: Task) -> Optional[List[str]]:
    """Generates research questions for a given task."""
    log.info(f"Generating research questions for task {task.id}...")
    system_prompt = """You are an AI research assistant helping to break down complex software tasks.
Given a task description, identify key areas requiring further research before implementation or subtask creation.
Focus on technical challenges, unknown implementation details, best practices, relevant libraries/tools, or potential design choices.
Generate a concise list of specific, actionable research questions.
Respond ONLY with a valid JSON object conforming to the `ResearchQuestions` model:
```python
class ResearchQuestions(BaseModel):
    questions: List[str]
```
Do not include ```json markdown.
"""
    user_content = f"""Analyze the following task and generate 5-10 specific research questions to guide its breakdown and implementation:

**Task ID:** {task.id}
**Title:** {task.title}
**Description:** {task.description or 'N/A'}
**Details:**
```
{task.details or 'No details provided.'}
```
**Test Strategy:**
```
{task.testStrategy or 'No test strategy provided.'}
```

Generate the list of research questions as a JSON object matching the `ResearchQuestions` model.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = await call_llm_with_instructor(messages, ResearchQuestions)
    return response.questions if response else None


async def group_questions_into_topics(questions: List[str], task: Task) -> Optional[Dict[str, List[str]]]:
    """Groups research questions into logical topics using an LLM."""
    if not questions:
        return None
    log.info(f"Grouping {len(questions)} research questions into topics for task {task.id}...")
    system_prompt = """You are an AI assistant skilled at organizing information.
Given a list of research questions related to a software task, group them into 2-4 high-level, logical research topics.
Each topic should represent a distinct area of investigation needed for the task.
Assign each question to exactly one topic.
Respond ONLY with a valid JSON object conforming to the `ResearchTopics` model:
```python
class ResearchTopics(BaseModel):
    topics: Dict[str, List[str]] # Key: Topic Name, Value: List of question strings
```
Ensure the keys are descriptive topic names and the values are lists of the original question strings. Do not include ```json markdown.
"""
    question_list_str = "\n".join([f"- {q}" for q in questions])
    user_content = f"""Group the following research questions related to the task "{task.title}" into 2-4 logical topics:

**Research Questions:**
{question_list_str}

**Original Task Context:**
Title: {task.title}
Description: {utils.truncate(task.description, 100)}

Generate the topics and their corresponding questions as a JSON object matching the `ResearchTopics` model.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = await call_llm_with_instructor(messages, ResearchTopics)
    return response.topics if response else None


async def generate_subtasks_with_llm(
    parent_task: Task,
    num_subtasks: int,
    next_subtask_id: int,
    additional_context: str = "", # This context can now include extensive research
) -> Optional[List[Subtask]]:
    """Generates subtasks for a given task using an LLM, potentially with research context."""
    log.info(f"Generating {num_subtasks} subtasks for task {parent_task.id} using {config.LLM_MODEL}...")
    has_research_context = bool(additional_context)

    system_prompt = f"""You are an AI assistant expert in breaking down software development tasks.
Your goal is to generate {num_subtasks} specific, actionable subtasks for the given parent task.
{'You have been provided with research findings relevant to this task. Use this information extensively to inform the subtask details, implementation guidance, and technical considerations.' if has_research_context else ''}

Subtasks should:
1. Be specific, actionable implementation steps, small enough for a focused coding session.
2. Follow a logical sequence for implementation.
3. Collectively cover the parent task's requirements, incorporating insights from the provided context/research.
4. Include clear, detailed implementation guidance in the 'details' field, referencing research where applicable.
5. Define dependencies between subtasks using their sequential IDs (starting from {next_subtask_id}). A subtask can only depend on subtasks with lower IDs within this batch. Use `[]` for no dependencies.

Respond ONLY with a valid JSON list containing exactly {num_subtasks} subtask objects, conforming to the `Subtask` model:
```python
class Subtask(BaseModel):
    id: int
    title: str
    description: Optional[str]
    details: Optional[str]
    status: Literal["pending", ...] = "pending"
    dependencies: List[Union[int, str]] = []
```
Assign sequential IDs starting from {next_subtask_id}. Do not include ```json markdown.
"""

    user_content = f"""Please break down the following parent task into exactly {num_subtasks} specific, actionable subtasks:

**Parent Task ID:** {parent_task.id}
**Parent Task Title:** {parent_task.title}
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

Generate a JSON list of {num_subtasks} subtask objects. Assign IDs sequentially starting from {next_subtask_id}. Define dependencies relative to these new IDs.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    # Allow more tokens as research context can be large
    max_tokens = config.MAX_TOKENS * 2 if has_research_context else config.MAX_TOKENS

    response = await call_llm_with_instructor(messages, List[Subtask], max_tokens_override=max_tokens)

    if response:
        log.info(f"Successfully generated {len(response)} subtasks for task {parent_task.id}.")
        # Post-process: Assign correct sequential IDs and validate dependencies
        validated_subtasks = []
        for i, subtask_model in enumerate(response):
             # Create Subtask instance to trigger potential default values
             subtask = Subtask(**subtask_model.model_dump())
             subtask.id = next_subtask_id + i
             subtask.status = "pending"
             # Validate dependencies: ensure they refer to IDs within this generated batch
             valid_deps = []
             for dep in subtask.dependencies:
                 try:
                     dep_id = int(dep)
                     if next_subtask_id <= dep_id < subtask.id:
                         valid_deps.append(dep_id)
                     else:
                          log.warning(f"Subtask {subtask.id} for parent {parent_task.id} has invalid dependency {dep_id}. Removing.")
                 except (ValueError, TypeError):
                      log.warning(f"Subtask {subtask.id} for parent {parent_task.id} has non-integer dependency '{dep}'. Removing.")
             subtask.dependencies = valid_deps
             validated_subtasks.append(subtask)

        if len(validated_subtasks) != num_subtasks:
             log.warning(f"LLM generated {len(validated_subtasks)} subtasks, expected {num_subtasks}. Check LLM output.")

        return validated_subtasks
    else:
        log.error(f"LLM failed to generate subtasks for task {parent_task.id}.")
        return None

async def analyze_task_complexity_with_llm(
    tasks_data: TasksData,
    use_research_prompt: bool = False, # Keep this simpler approach for analyze
) -> Optional[List[ComplexityAnalysisItem]]:
    """Analyzes complexity for a list of tasks using an LLM."""
    log.info(f"Analyzing complexity for {len(tasks_data.tasks)} tasks using {config.LLM_MODEL}...")
    research_guidance = """
Leverage your knowledge of software engineering best practices, potential complexities, and common implementation patterns related to each task's domain.
Provide insightful reasoning for your complexity score and tailored prompts for effective subtask breakdown.
""" if use_research_prompt else ""

    system_prompt = f"""You are an expert software architect and project manager analyzing task complexity.
Your goal is to assess each provided task and generate complexity analysis data.

For each task, provide:
- taskId: The original ID of the task.
- taskTitle: The original title of the task.
- complexityScore: A float score from 1.0 (very simple) to 10.0 (very complex).
- recommendedSubtasks: An integer number of subtasks (typically 3-7) appropriate for the complexity. More complex tasks need more subtasks.
- expansionPrompt: A concise, specific prompt tailored to the task, designed to guide an AI in generating high-quality subtasks for *this specific task*. Focus on key challenges or areas needing detail.
- reasoning: A brief justification for the complexity score, highlighting key factors.
{research_guidance}
Respond ONLY with a valid JSON list containing an analysis object for EVERY task provided in the input, conforming to the `ComplexityAnalysisItem` model. Do not include ```json markdown.
"""
    tasks_input_str = "\n---\n".join([
        f"Task ID: {task.id}\nTitle: {task.title}\nDescription: {utils.truncate(task.description, 150)}\nDetails: {utils.truncate(task.details, 200)}\nDependencies: {task.dependencies or []}\nPriority: {task.priority or ''}"
        for task in tasks_data.tasks
    ])
    user_content = f"""Please analyze the complexity of the following tasks:

{tasks_input_str}

Return a JSON list containing an analysis object for each task, following the specified `ComplexityAnalysisItem` structure. Ensure every input task has a corresponding analysis object in the output list.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = await call_llm_with_instructor(messages, List[ComplexityAnalysisItem], max_tokens_override=config.MAX_TOKENS * 2)

    if response:
        log.info(f"Successfully generated complexity analysis for {len(response)} tasks.")
        # Validate results
        analyzed_ids = {item.taskId for item in response}
        input_ids = {task.id for task in tasks_data.tasks}
        missing_ids = input_ids - analyzed_ids
        if missing_ids:
            log.warning(f"LLM did not provide analysis for all tasks. Missing IDs: {missing_ids}")
        extra_ids = analyzed_ids - input_ids
        if extra_ids:
             log.warning(f"LLM provided analysis for unexpected task IDs: {extra_ids}. Filtering them out.")
             response = [item for item in response if item.taskId in input_ids]
        # Basic validation of scores/counts
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
    use_research_prompt: bool = False, # Keep this simple for update
) -> Optional[List[Task]]:
    """Updates a list of tasks based on a new prompt using an LLM."""
    if not tasks_to_update: return []
    log.info(f"Updating {len(tasks_to_update)} tasks using {config.LLM_MODEL}...")
    research_guidance = """
Leverage your knowledge of current best practices and implementation details related to the requested update. Ensure the tasks remain coherent and technically sound after the changes.
""" if use_research_prompt else ""

    system_prompt = f"""You are an AI assistant helping to update software development tasks based on new context or requirements changes.
You will be given a list of existing tasks and a prompt describing the changes.
Your job is to update the tasks (`title`, `description`, `details`, `testStrategy`) to reflect these changes accurately, while preserving the overall structure (`id`, `status`, `dependencies`, `priority`, `subtasks`).

Guidelines:
1. Modify ONLY the necessary fields (`title`, `description`, `details`, `testStrategy`) to align with the update prompt.
2. Do NOT change task `id`, `status`, `dependencies`, `priority`, or `subtasks` unless the prompt explicitly requires it and gives clear instructions.
3. Apply the changes specified in the update prompt consistently across all relevant tasks provided.
4. Ensure the updated tasks remain actionable and clear.
{research_guidance}
Respond ONLY with a valid JSON list containing the updated task objects for ALL tasks provided in the input, conforming to the `Task` model. The list should contain the same number of tasks as the input. Do not include ```json markdown.
"""
    tasks_input_str = json.dumps(
        [t.model_dump(exclude={'subtasks'}) for t in tasks_to_update], # Exclude subtasks for brevity
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
Return a JSON list containing the updated task objects for ALL the tasks listed above. Ensure the output conforms to the `Task` model structure. Preserve fields like `id`, `status`, `dependencies`, `priority`, and existing `subtasks` unless the update prompt specifically instructs otherwise.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = await call_llm_with_instructor(messages, List[Task], max_tokens_override=config.MAX_TOKENS * 2)

    if response:
        log.info(f"Successfully received updated task data for {len(response)} tasks.")
        # Validation and merging logic (ensure IDs match, preserve subtasks, etc.)
        original_tasks_map = {t.id: t for t in tasks_to_update}
        final_updated_tasks = []
        processed_ids = set()

        for updated_task_model in response:
            # Create Task instance to trigger defaults/validation
            updated_task = Task(**updated_task_model.model_dump())
            original_task = original_tasks_map.get(updated_task.id)

            if not original_task:
                log.warning(f"LLM returned update for unknown task ID {updated_task.id}. Skipping.")
                continue

            # Preserve fields that should not change
            updated_task.id = original_task.id
            updated_task.status = original_task.status
            updated_task.dependencies = original_task.dependencies
            updated_task.priority = original_task.priority
            updated_task.subtasks = original_task.subtasks # Crucial: preserve original subtasks

            final_updated_tasks.append(updated_task)
            processed_ids.add(updated_task.id)

        # Check if any original tasks were missed by the LLM
        missed_tasks = [t for t in tasks_to_update if t.id not in processed_ids]
        if missed_tasks:
            log.warning(f"LLM did not return updates for {len(missed_tasks)} tasks. Keeping original versions.")
            final_updated_tasks.extend(missed_tasks) # Add back the originals

        # Ensure original order if possible (though map lookup might disrupt it)
        final_updated_tasks.sort(key=lambda t: tasks_to_update.index(original_tasks_map[t.id]) if t.id in original_tasks_map else float('inf'))

        return final_updated_tasks
    else:
        log.error("LLM failed to update tasks.")
        return None