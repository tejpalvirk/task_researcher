import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
dotenv_path = Path('.') / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- LLM Configuration ---
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20240620") # Primary model
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

# --- STORM Configuration ---
STORM_RETRIEVER = os.getenv("STORM_RETRIEVER", "bing").lower()
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")
YDC_API_KEY = os.getenv("YDC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# Add other potential keys here

STORM_LLM_MODEL = os.getenv("STORM_LLM_MODEL", "claude-3-haiku-20240307") # Model for STORM internals
STORM_EMBEDDING_MODEL = os.getenv("STORM_EMBEDDING_MODEL") # e.g., "text-embedding-3-small"
STORM_SEARCH_TOP_K = int(os.getenv("STORM_SEARCH_TOP_K", 5))
STORM_MAX_TOKENS_CONV = int(os.getenv("STORM_MAX_TOKENS_CONV", 500))
STORM_MAX_TOKENS_ARTICLE = int(os.getenv("STORM_MAX_TOKENS_ARTICLE", 3000))
STORM_OUTPUT_DIR = Path(os.getenv("STORM_OUTPUT_DIR", "scripts/storm_research_output"))

# --- Task Researcher Configuration ---
DEFAULT_SUBTASKS = int(os.getenv("DEFAULT_SUBTASKS", 3))
DEFAULT_PRIORITY = os.getenv("DEFAULT_PRIORITY", "medium")
PROJECT_NAME = os.getenv("PROJECT_NAME", "Task Researcher Project")
PROJECT_VERSION = "0.1.0" # Should ideally pull from pyproject.toml if possible
TASKS_FILE_PATH = Path(os.getenv("TASKS_FILE_PATH", "tasks/tasks.json"))
COMPLEXITY_REPORT_PATH = Path(os.getenv("COMPLEXITY_REPORT_PATH", "scripts/task-complexity-report.json"))
TASK_FILES_DIR = Path(os.getenv("TASK_FILES_DIR", "tasks"))

# --- Input Files ---
FUNCTIONAL_SPEC_PATH = Path("scripts/functional_spec.md")
TECHNICAL_SPEC_PATH = Path("scripts/technical_spec.md")
PLAN_PATH = Path("scripts/plan.md")
BACKGROUND_PATH = Path("scripts/background.md")

# --- Logging ---
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

# --- Validation ---
def check_api_keys():
    """Checks if necessary API keys are set."""
    keys_needed = []
    # Basic check - litellm needs at least one primary key
    if not any([ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY]):
         keys_needed.append("At least one LLM API Key (ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY)")

    # Check STORM retriever key based on config
    retriever = STORM_RETRIEVER
    if retriever == "bing" and not BING_SEARCH_API_KEY:
        keys_needed.append("BING_SEARCH_API_KEY (for STORM)")
    elif retriever == "you" and not YDC_API_KEY:
         keys_needed.append("YDC_API_KEY (for STORM)")
    elif retriever == "tavily" and not TAVILY_API_KEY:
         keys_needed.append("TAVILY_API_KEY (for STORM)")
    elif retriever == "serper" and not SERPER_API_KEY:
          keys_needed.append("SERPER_API_KEY (for STORM)")
    # Add other retriever checks here...

    return keys_needed

# Ensure default directories exist
TASKS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
COMPLEXITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
TASK_FILES_DIR.mkdir(parents=True, exist_ok=True)
STORM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Create STORM output dir