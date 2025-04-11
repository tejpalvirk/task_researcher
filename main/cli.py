import typer
import asyncio # Import asyncio
from .commands import app as typer_app # Import the Typer app instance
from mcp.server import run_server as run_mcp_server # Import the MCP server runner

# Option 1: Keep the main entry point running the Typer CLI
# Users would run the MCP server via `python -m task_researcher.mcp.server`

# Option 2: Add a command to the Typer CLI to start the MCP server
@typer_app.command("serve-mcp")
def serve_mcp_command():
    """Runs the Task Researcher MCP Server."""
    print("Starting Task Researcher MCP Server...")
    print("Note: This server uses stdio transport by default.")
    print("Press Ctrl+C to stop.")
    run_mcp_server() # Call the server runner function

# Main entry point still runs the Typer app
if __name__ == "__main__":
    typer_app()

# --- How to run ---
# CLI commands: poetry run task-researcher <command> ...
# MCP Server:   poetry run task-researcher serve-mcp
# Or directly:  poetry run python -m task_researcher.mcp.server