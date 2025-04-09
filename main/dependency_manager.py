from typing import List, Dict, Any, Set, Optional, Tuple, Union
from .utils import log, read_json, write_json, task_exists, find_task_by_id, find_cycles
from .models import TasksData, Task, Subtask # Use models
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def validate_dependencies(tasks_data: TasksData) -> Tuple[bool, List[Dict[str, Any]]]:
    """Validates dependencies for all tasks and subtasks."""
    issues = []
    all_tasks = tasks_data.tasks
    all_task_ids = {task.id for task in all_tasks}
    all_subtask_ids = set()
    for task in all_tasks:
        if task.subtasks:
            for subtask in task.subtasks:
                all_subtask_ids.add(f"{task.id}.{subtask.id}")

    valid_ids_str = {str(tid) for tid in all_task_ids}.union(all_subtask_ids) # Use strings for comparison
    dependency_map: Dict[str, List[str]] = {} # Store string IDs

    for task in all_tasks:
        task_id_str = str(task.id)
        task_deps_str_map = []
        # Validate task dependencies
        if task.dependencies:
            for dep_id in task.dependencies:
                dep_id_str = str(dep_id) # Work with string representation
                if dep_id_str == task_id_str:
                    issues.append({"type": "self", "id": task_id_str, "dep": dep_id_str})
                elif dep_id_str not in valid_ids_str:
                     # Check if it's maybe a subtask ID written as int?
                     is_maybe_subtask_ref = isinstance(dep_id, int) and dep_id < 100 # Heuristic
                     if not is_maybe_subtask_ref: # Only report if not likely a subtask int ref
                         issues.append({"type": "missing", "id": task_id_str, "dep": dep_id_str})
                task_deps_str_map.append(dep_id_str)
            dependency_map[task_id_str] = task_deps_str_map
        else:
             dependency_map[task_id_str] = []

        # Validate subtask dependencies
        if task.subtasks:
            for subtask in task.subtasks:
                subtask_id_str = f"{task.id}.{subtask.id}"
                subtask_deps_str_map = []
                if subtask.dependencies:
                    for dep_id in subtask.dependencies:
                        # Normalize potential int subtask references TO STRING
                        if isinstance(dep_id, int) and dep_id < 100: # Heuristic: likely refers to sibling subtask
                             norm_dep_id_str = f"{task.id}.{dep_id}"
                        else:
                             norm_dep_id_str = str(dep_id)

                        if norm_dep_id_str == subtask_id_str:
                            issues.append({"type": "self", "id": subtask_id_str, "dep": norm_dep_id_str})
                        elif norm_dep_id_str not in valid_ids_str:
                             issues.append({"type": "missing", "id": subtask_id_str, "dep": norm_dep_id_str})
                        subtask_deps_str_map.append(norm_dep_id_str)
                    dependency_map[subtask_id_str] = subtask_deps_str_map
                else:
                     dependency_map[subtask_id_str] = []


    # Check for cycles using the map (IDs are strings now)
    visited: Set[str] = set()
    recursion_stack: Set[str] = set()
    all_node_ids = list(dependency_map.keys()) # Get all task/subtask IDs from the map

    for node_id_str in all_node_ids:
        if node_id_str not in visited:
            cycle_paths = find_cycles(node_id_str, dependency_map, visited, recursion_stack)
            if cycle_paths:
                # Report cycle involving the starting node
                # Just mark the node as part of *a* cycle for simplicity
                if not any(issue['type'] == 'cycle' and issue['id'] == node_id_str for issue in issues):
                     issues.append({"type": "cycle", "id": node_id_str})

    return len(issues) == 0, issues


def fix_dependencies(tasks_data: TasksData) -> Tuple[bool, Dict[str, int]]:
    """Finds and fixes invalid dependencies."""
    fixes_summary = {"missing": 0, "self": 0, "cycle": 0}
    made_changes = False

    all_tasks = tasks_data.tasks
    all_task_ids = {task.id for task in all_tasks}
    all_subtask_ids = set()
    for task in all_tasks:
        if task.subtasks:
            for subtask in task.subtasks:
                all_subtask_ids.add(f"{task.id}.{subtask.id}")

    valid_ids_str = {str(tid) for tid in all_task_ids}.union(all_subtask_ids) # Use strings
    dependency_map: Dict[str, List[str]] = {} # Use string IDs

    # --- Pass 1: Remove missing and self dependencies ---
    for task in all_tasks:
        task_id_str = str(task.id)
        original_deps = list(task.dependencies) if task.dependencies else []
        valid_task_deps_pass1 = []
        task_deps_str_map_pass1 = []

        if task.dependencies:
            for dep_id in task.dependencies:
                dep_id_str = str(dep_id)
                is_valid = True
                if dep_id_str == task_id_str:
                    log.warning(f"Removing self-dependency from task {task_id_str}")
                    fixes_summary["self"] += 1
                    is_valid = False
                elif dep_id_str not in valid_ids_str:
                     is_maybe_subtask_ref = isinstance(dep_id, int) and dep_id < 100
                     if not is_maybe_subtask_ref:
                         log.warning(f"Removing missing dependency '{dep_id_str}' from task {task_id_str}")
                         fixes_summary["missing"] += 1
                         is_valid = False
                     else: # Keep potential subtask ref for now
                         pass

                if is_valid:
                    valid_task_deps_pass1.append(dep_id) # Keep original type
                    task_deps_str_map_pass1.append(dep_id_str)

            if len(valid_task_deps_pass1) != len(original_deps):
                task.dependencies = valid_task_deps_pass1
                made_changes = True
            dependency_map[task_id_str] = task_deps_str_map_pass1
        else:
             dependency_map[task_id_str] = []


        if task.subtasks:
            for subtask in task.subtasks:
                subtask_id_str = f"{task.id}.{subtask.id}"
                original_sub_deps = list(subtask.dependencies) if subtask.dependencies else []
                valid_subtask_deps_pass1 = []
                subtask_deps_str_map_pass1 = []

                if subtask.dependencies:
                    for dep_id in subtask.dependencies:
                        # Normalize potential int subtask references TO STRING
                        if isinstance(dep_id, int) and dep_id < 100:
                             norm_dep_id_str = f"{task.id}.{dep_id}"
                        else:
                             norm_dep_id_str = str(dep_id)

                        is_valid = True
                        if norm_dep_id_str == subtask_id_str:
                             log.warning(f"Removing self-dependency from subtask {subtask_id_str}")
                             fixes_summary["self"] += 1
                             is_valid = False
                        elif norm_dep_id_str not in valid_ids_str:
                             log.warning(f"Removing missing dependency '{norm_dep_id_str}' from subtask {subtask_id_str}")
                             fixes_summary["missing"] += 1
                             is_valid = False

                        if is_valid:
                            valid_subtask_deps_pass1.append(dep_id) # Keep original type
                            subtask_deps_str_map_pass1.append(norm_dep_id_str)

                    if len(valid_subtask_deps_pass1) != len(original_sub_deps):
                        subtask.dependencies = valid_subtask_deps_pass1
                        made_changes = True
                    dependency_map[subtask_id_str] = subtask_deps_str_map_pass1
                else:
                    dependency_map[subtask_id_str] = []


    # --- Pass 2: Detect and break cycles ---
    visited: Set[str] = set()
    recursion_stack: Set[str] = set()
    all_node_ids = list(dependency_map.keys())

    for node_id_str in all_node_ids:
        if node_id_str not in visited:
             # find_cycles returns list of edges [from, to] that form cycles
             cycle_edges = find_cycles(node_id_str, dependency_map, visited, recursion_stack)
             if cycle_edges:
                 for cycle_from_str, cycle_to_str in cycle_edges:
                     log.warning(f"Breaking cycle: Removing dependency {cycle_from_str} -> {cycle_to_str}")
                     # Find the task/subtask object `cycle_from_str` (using dict version for now)
                     node_obj_dict = find_task_by_id(tasks_data.model_dump()['tasks'], cycle_from_str) # Find in dict representation

                     # Find the corresponding object in the Pydantic model list to modify it
                     node_obj_model = None
                     if '.' in cycle_from_str:
                          parent_id, sub_id = map(int, cycle_from_str.split('.'))
                          parent_task_model = next((t for t in tasks_data.tasks if t.id == parent_id), None)
                          if parent_task_model and parent_task_model.subtasks:
                               node_obj_model = next((st for st in parent_task_model.subtasks if st.id == sub_id), None)
                     else:
                          node_obj_model = next((t for t in tasks_data.tasks if t.id == int(cycle_from_str)), None)


                     if node_obj_model and node_obj_model.dependencies:
                         # Determine the original type of the dependency to remove
                         dep_to_remove_original_type: Union[int, str]
                         try: # Is 'cycle_to_str' a simple task ID?
                             dep_to_remove_original_type = int(cycle_to_str)
                         except ValueError: # It's a subtask like "p.s" or other string
                             dep_to_remove_original_type = cycle_to_str

                         # Refine removal logic based on original type stored in the model
                         original_len = len(node_obj_model.dependencies)
                         node_obj_model.dependencies = [
                             d for d in node_obj_model.dependencies if d != dep_to_remove_original_type
                         ]

                         # Check if value was numeric sibling subtask reference that got normalized
                         if isinstance(dep_to_remove_original_type, str) and '.' in dep_to_remove_original_type:
                             try:
                                  parent_part, sub_part = dep_to_remove_original_type.split('.')
                                  # If the node being modified is also a subtask from the same parent...
                                  if '.' in cycle_from_str and cycle_from_str.startswith(parent_part + '.'):
                                       # ...then the original dependency might have been just the integer sub_id
                                       sub_id_int = int(sub_part)
                                       # Try removing the integer version as well
                                       node_obj_model.dependencies = [
                                           d for d in node_obj_model.dependencies if d != sub_id_int
                                       ]
                             except (ValueError, IndexError):
                                  pass # Ignore parsing errors


                         if len(node_obj_model.dependencies) < original_len:
                             fixes_summary["cycle"] += 1
                             made_changes = True
                             # Update dependency map in memory for subsequent checks
                             dependency_map[cycle_from_str] = [
                                 d for d in dependency_map.get(cycle_from_str, []) if d != cycle_to_str
                             ]

    return made_changes, fixes_summary