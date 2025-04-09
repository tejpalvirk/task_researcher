import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
import re
import datetime # Added for timestamping

from rich.logging import RichHandler
from . import config

# --- Logging Setup ---
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=config.DEBUG)]
)
log = logging.getLogger("main") # Changed logger name

# --- File Operations ---
def read_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Reads and parses a JSON file."""
    try:
        with filepath.open('r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        log.error(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON from {filepath}: {e}")
        return None
    except Exception as e:
        log.error(f"Error reading file {filepath}: {e}")
        return None

def write_json(filepath: Path, data: Dict[str, Any]):
    """Writes data to a JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        log.debug(f"Successfully wrote JSON to {filepath}")
    except Exception as e:
        log.error(f"Error writing JSON to {filepath}: {e}")

def read_file(filepath: Path) -> Optional[str]:
    """Reads content from a text file."""
    try:
        return filepath.read_text(encoding='utf-8')
    except FileNotFoundError:
        log.warning(f"Optional input file not found: {filepath}")
        return "" # Return empty string instead of None
    except Exception as e:
        log.error(f"Error reading file {filepath}: {e}")
        return "" # Return empty string on error

def write_file(filepath: Path, content: str):
    """Writes content to a text file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding='utf-8')
        log.debug(f"Successfully wrote file to {filepath}")
    except Exception as e:
        log.error(f"Error writing file {filepath}: {e}")


# --- Task Utilities ---
# (task_exists, find_task_by_id, get_next_task_id, get_next_subtask_id remain the same)
def task_exists(tasks: List[Dict[str, Any]], task_id: Union[str, int]) -> bool:
    """Checks if a task or subtask exists."""
    if not task_id or not tasks:
        return False

    task_id_str = str(task_id)
    if '.' in task_id_str:
        try:
            parent_id_str, sub_id_str = task_id_str.split('.')
            parent_id = int(parent_id_str)
            sub_id = int(sub_id_str)
            parent_task = next((t for t in tasks if t.get('id') == parent_id), None)
            if parent_task and 'subtasks' in parent_task and isinstance(parent_task['subtasks'], list):
                return any(st.get('id') == sub_id for st in parent_task['subtasks'])
        except (ValueError, TypeError):
            return False # Invalid format or structure
    else:
        try:
            numeric_id = int(task_id_str)
            return any(t.get('id') == numeric_id for t in tasks)
        except (ValueError, TypeError):
            return False # Invalid format or structure

    return False

def find_task_by_id(tasks: List[Dict[str, Any]], task_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    """Finds a task or subtask by its ID."""
    if not task_id or not tasks:
        return None

    task_id_str = str(task_id)
    if '.' in task_id_str:
        try:
            parent_id_str, sub_id_str = task_id_str.split('.')
            parent_id = int(parent_id_str)
            sub_id = int(sub_id_str)
            parent_task = next((t for t in tasks if t.get('id') == parent_id), None)
            if parent_task and 'subtasks' in parent_task and isinstance(parent_task['subtasks'], list):
                subtask = next((st for st in parent_task['subtasks'] if st.get('id') == sub_id), None)
                if subtask:
                    # Add parent info for context
                    subtask['parentTaskId'] = parent_id
                    subtask['isSubtask'] = True
                    # Make sure parent reference isn't circular for display/logging
                    subtask['parentTaskInfo'] = {'id': parent_task.get('id'), 'title': parent_task.get('title')}
                return subtask
        except (ValueError, TypeError):
            return None
    else:
        try:
            numeric_id = int(task_id_str)
            return next((t for t in tasks if t.get('id') == numeric_id), None)
        except (ValueError, TypeError):
             return None
    return None


def get_next_task_id(tasks: List[Dict[str, Any]]) -> int:
    """Determines the next available task ID."""
    if not tasks:
        return 1
    return max([t.get('id', 0) for t in tasks], default=0) + 1

def get_next_subtask_id(parent_task: Dict[str, Any]) -> int:
    """Determines the next available subtask ID within a parent task."""
    if not parent_task or 'subtasks' not in parent_task or not parent_task['subtasks'] or not isinstance(parent_task['subtasks'], list):
        return 1
    return max([st.get('id', 0) for st in parent_task['subtasks']], default=0) + 1


# --- Text & String Utilities ---
# (truncate, sanitize_filename remain the same)
def truncate(text: Optional[str], max_length: int) -> str:
    """Truncates text to a specified length."""
    if not text:
        return ""
    text = str(text) # Ensure it's a string
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be used as a filename."""
    name = str(name) # Ensure string
    name = re.sub(r'[^\w\-_\. ]', '_', name) # Allow letters, numbers, underscore, hyphen, dot, space
    name = re.sub(r'\s+', '_', name) # Replace spaces with underscores
    name = name.strip('_')
    return name if name else "untitled" # Ensure not empty


# --- Complexity Report Utilities ---
# (read_complexity_report, find_task_in_complexity_report remain the same)
def read_complexity_report(custom_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Reads the complexity report."""
    report_path = custom_path or config.COMPLEXITY_REPORT_PATH
    return read_json(report_path)

def find_task_in_complexity_report(report: Dict[str, Any], task_id: int) -> Optional[Dict[str, Any]]:
    """Finds a task analysis in the complexity report."""
    if not report or 'complexityAnalysis' not in report or not isinstance(report['complexityAnalysis'], list):
        return None
    return next((analysis for analysis in report['complexityAnalysis'] if analysis.get('taskId') == task_id), None)

# --- Dependency Cycle Detection ---
# (find_cycles remains the same)
def find_cycles(
    task_id: Union[str, int],
    dependency_map: Dict[str, List[str]], # Expects string keys/values now
    visited: Set[str],
    recursion_stack: Set[str]
) -> List[List[str]]:
    """
    Finds cycles in dependencies using DFS.
    Returns a list of cycles found, where each cycle is a list of task ID strings.
    """
    task_id_str = str(task_id)
    visited.add(task_id_str)
    recursion_stack.add(task_id_str)
    cycles_found = []

    dependencies = dependency_map.get(task_id_str, [])
    for dep_id_str in dependencies: # Iterate over string IDs
        if dep_id_str not in visited:
            new_cycles = find_cycles(dep_id_str, dependency_map, visited, recursion_stack)
            cycles_found.extend(new_cycles)
        elif dep_id_str in recursion_stack:
            # Cycle detected
            log.warning(f"Cycle detected involving {task_id_str} and {dep_id_str}")
            # Return the edge causing the cycle (node -> node already in stack)
            cycles_found.append([task_id_str, dep_id_str])

    recursion_stack.remove(task_id_str)
    return cycles_found