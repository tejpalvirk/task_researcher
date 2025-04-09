# Task Researcher

A Python task management system designed for AI-driven development, featuring integrated, in-depth research capabilities using the `knowledge-storm` library. Break down complex projects, generate tasks, and leverage automated research to inform implementation details.

**This package provides both a command-line interface (CLI) and a Model Context Protocol (MCP) Server.**

## Core Features

*   **Parse Inputs**: Generate initial tasks from project specification files (`functional_spec.md`, `technical_spec.md`, `plan.md`, `background.md`).
*   **Expand Tasks**:
    *   Break down tasks into subtasks using AI (`claude`, `gemini`, etc. via `litellm`).
    *   **STORM-Powered Research (`--research` flag)**: For complex tasks, automatically identify research questions, group them into topics, run the `knowledge-storm` engine for each topic, and use the aggregated research to generate highly informed subtasks.
*   **Update Tasks**: Modify pending tasks based on new prompts or requirement changes.
*   **Analyze Complexity**: Assess task complexity using AI, generating a report with recommendations and tailored expansion prompts. (`--research-hint` flag available).
*   **Dependency Management**: Validate and automatically fix dependency issues (missing refs, self-deps, simple cycles).
*   **Generate Files**: Create individual `.txt` files for each task and subtask.
*   **Standalone Research (`research-topic`)**: Generate a detailed research report on any topic using `knowledge-storm`.

## Requirements

*   Python 3.10+
*   An API key for at least one supported LLM provider (e.g., Anthropic, Google Gemini, OpenAI) set in a `.env` file (used for task generation, complexity analysis, etc.).
*   `knowledge-storm` library (`pip install knowledge-storm`).
*   API key for a search engine supported by `knowledge-storm` (e.g., Bing Search, You.com, Tavily) set in `.env` (Required for `--research` in `expand` and the `research-topic` command).
*   (Optional) `mcp` library (`pip install mcp`) if running as an MCP server.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd task-researcher
    ```
2.  **Install dependencies (using Poetry recommended):**
    ```bash
    pip install poetry
    poetry install
    ```
3.  **Configure Environment:**
    *   Copy `.env.example` to `.env`.
    *   Fill in your primary LLM API key (e.g., `ANTHROPIC_API_KEY`).
    *   Set the `LLM_MODEL` for primary tasks (e.g., `"claude-3-5-sonnet-20240620"`).
    *   Set the `STORM_RETRIEVER` (e.g., `"bing"`) and its corresponding API key (`BING_SEARCH_API_KEY`).
    *   (Optional) Set `STORM_LLM_MODEL` if you want STORM to use a different (e.g., faster/cheaper) model internally.
    *   (Optional) Adjust other settings like `MAX_TOKENS`, `TEMPERATURE`, file paths, etc.

## Usage Command Line Interface (CLI)

Use the `task-researcher` command (if installed via Poetry scripts) or `python -m task_researcher`.

```bash
# Show help
task-researcher --help

# Generate initial tasks from spec files in ./scripts/
task-researcher parse-inputs --num-tasks 20

# Analyze complexity (using primary LLM's knowledge)
task-researcher analyze-complexity --research-hint

# View the complexity report
task-researcher complexity-report

# --- Expanding Tasks ---

# Expand task 5 normally (using primary LLM)
task-researcher expand --id 5 --num 4

# Expand task 7 using the STORM research workflow
# (Generates questions -> groups topics -> runs STORM -> generates subtasks w/ research)
task-researcher expand --id 7 --research

# Expand all eligible tasks using STORM research workflow
task-researcher expand --all --research

# Expand task 9, forcing overwrite, using STORM, with extra user context
task-researcher expand --id 9 --research --force --prompt "Ensure compatibility with React 19 features."

# --- Other Commands ---

# Update tasks from ID 4 onwards due to a change
task-researcher update --from 4 --prompt "Refactor authentication to use JWT instead of session cookies."

# Validate dependencies
task-researcher validate-deps

# Fix dependencies automatically
task-researcher fix-deps

# Generate individual .txt task files
task-researcher generate

# Generate a standalone research report on a topic using STORM
task-researcher research-topic "Comparison of Vector Databases for RAG" -o "reports/vector_db_comparison.md" --retriever tavily # Requires TAVILY_API_KEY
```
## MCP Server

This package can also run as an MCP server, exposing its functionality as tools and resources for MCP clients like Claude Desktop, Cursor, or custom applications.

### Running the Server:

```bash
poetry run task-researcher serve-mcp
# OR
poetry run python -m task_researcher.mcp.server
```
This starts the server using **stdio transport** by default.

### Connecting from Clients (Example: Claude Desktop):

1. Open Claude Desktop settings -> Developer -> Edit Config.
2. Add an entry to mcpServers in claude_desktop_config.json:
```json
{
  "mcpServers": {
    "task-researcher": {
      // Adjust command based on your environment (Poetry vs. global python)
      // Option 1: Using Poetry
      "command": "poetry", // Or the full path to your poetry executable
      "args": [
        "run",
        "task-researcher", // The script name from pyproject.toml
        "serve-mcp"
      ],
      // Make sure poetry run executes in the correct project directory
      "options": {
          "cwd": "/absolute/path/to/your/task-researcher/project"
      }
      // Option 2: Assuming task-researcher is installed globally or in PATH
      // "command": "task-researcher",
      // "args": ["serve-mcp"]
      // Option 3: Direct Python execution
      // "command": "python", // Or python3 or full path
      // "args": [
      //     "-m",
      //     "task_researcher.mcp.server",
      //      // Specify the working directory if needed relative paths are used in config
      //      // or ensure .env is found relative to where python is run
      // ],
      // "options": { // Ensure correct working directory if needed
      //     "cwd": "/absolute/path/to/your/task-researcher/project"
      // }
    }
    // Add other servers here...
  }
}
```
3. Replace /absolute/path/to/your/task-researcher/project with the actual path to the cloned repository.
4. Restart Claude Desktop.

### Exposed MCP Tools:

* `parse_inputs`: Generates initial tasks from configured spec files.
* `update_tasks`: Updates tasks from a given ID based on a prompt.
* `generate_task_files`: Creates individual task_XXX.txt files.
* `expand_task`: Expands a single task into subtasks (supports research=True for STORM).
* `expand_all_tasks`: Expands all eligible pending tasks (supports research=True).
* `analyze_complexity`: Analyzes task complexity and saves a report.
* `validate_dependencies`: Checks dependencies for issues.
* `fix_dependencies`: Attempts to automatically fix dependency issues.
* `research_topic`: Runs STORM to generate a research report on a topic.

### Exposed MCP Resources:

* `tasks://current`: The content of the main tasks.json file.
* `report://complexity`: The content of the task-complexity-report.json file.
* `research://{topic_name}`: The content of a generated STORM report for the given topic.
* `taskfile://{task_id}`: The content of a specific task_XXX.txt file.

(See `mcp/server.py` for precise tool/resource definitions and parameters).


## `--research` vs `--research-hint`

*   **`expand --research`**: Triggers the full **STORM-based workflow**: AI generates questions -> AI groups questions -> `knowledge-storm` runs web searches & synthesizes reports -> AI generates subtasks using the synthesized reports as context. **Requires `knowledge-storm` and a configured retriever API key.**
*   **`analyze-complexity --research-hint`**: Modifies the prompt for the *primary LLM* performing the complexity analysis, asking it to leverage its internal knowledge base more deeply, like a research assistant would. **Does not use `knowledge-storm` or perform live web searches.**
*   **`update --research-hint`**: Similar to `analyze-complexity`, hints the primary LLM to use its internal knowledge when processing the update prompt.

## Configuration (`.env`)

Refer to `.env.example`. Key settings include:

*   `LLM_MODEL`: Primary model for tasks, analysis, subtask generation (non-STORM).
*   `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.: Credentials for the primary LLM.
*   `STORM_RETRIEVER`: Search engine for STORM (`bing`, `you`, `tavily`, etc.).
*   `BING_SEARCH_API_KEY`, `YDC_API_KEY`, `TAVILY_API_KEY`, etc.: Key for the chosen STORM retriever.
*   `STORM_LLM_MODEL`: (Optional) Different model for STORM's internal processing.
*   `STORM_SEARCH_TOP_K`, `STORM_MAX_TOKENS_*`: Control STORM's depth.
*   File paths (`TASKS_FILE_PATH`, etc.).

## Task Structure (`tasks/tasks.json`)

The structure follows the Pydantic models defined in `task_master_py/models.py`, including `Task`, `Subtask`, and `TasksData`.

```json
{
  "meta": {
    "projectName": "Task Master Project",
    "projectVersion": "0.1.0",
    "sourceFiles": ["scripts/plan.md", "..."],
    "generatedAt": "...",
    "totalTasks": 15
  },
  "tasks": [
    {
      "id": 1,
      "title": "Setup Project Environment",
      "description": "Initialize project structure, install dependencies, configure linters.",
      "details": "...",
      "status": "pending",
      "dependencies": [],
      "priority": "high",
      "testStrategy": "Verify environment setup by running basic commands.",
      "subtasks": []
    },
    {
      "id": 2,
      "title": "Implement Core Data Models",
      "description": "Define Pydantic models for tasks and reports.",
      "details": "...",
      "status": "pending",
      "dependencies": [1], // Depends on task 1
      "priority": "high",
      "testStrategy": "Unit tests for model validation.",
      "subtasks": [
        {
            "id": 1, // Subtask ID is relative to parent
            "title": "Define Task Model",
            "description": "...",
            "details": "...",
            "status": "pending",
            "dependencies": [] // Dependencies relative to sibling subtasks (ints)
        },
        {
            "id": 2,
            "title": "Define Report Model",
            "description": "...",
            "details": "...",
            "status": "pending",
            "dependencies": [1] // Depends on subtask 1
        }
      ]
    },
    // ... more tasks
  ]
}
```