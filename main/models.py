from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Union, Literal, Dict, Any

from . import config # Import config for defaults

# --- Task & Subtask Models ---
class Subtask(BaseModel):
    model_config = ConfigDict(extra='allow') # Allow extra fields like parentTaskId

    id: int
    title: str = Field(..., description="Brief, descriptive title of the subtask.")
    description: Optional[str] = Field(None, description="Concise description of what the subtask involves.")
    details: Optional[str] = Field(None, description="In-depth implementation instructions for the subtask.")
    acceptanceCriteria: Optional[str] = Field(None, description="Criteria to verify subtask completion.") # Added field
    status: Literal["pending", "in-progress", "done", "deferred", "blocked"] = Field("pending", description="Current state of the subtask.")
    # Dependencies are IDs of *sibling* subtasks (int) or parent/other tasks (int/str)
    dependencies: List[Union[int, str]] = Field([], description="IDs of subtasks (int) or tasks (int/str) this depends on.")

    @field_validator('dependencies', mode='before')
    @classmethod
    def ensure_dependencies_list(cls, v):
        if v is None: return []
        if not isinstance(v, list): raise ValueError("Dependencies must be a list")
        return [str(item) if isinstance(item, str) else int(item) for item in v]

class Task(BaseModel):
    id: int
    title: str = Field(..., description="Brief, descriptive title of the task.")
    phase: Optional[str] = Field(None, description="The development phase this task belongs to (e.g., 'Setup', 'Backend API', 'UI').") # Added field
    description: Optional[str] = Field(None, description="Concise description of what the task involves.")
    details: Optional[str] = Field(None, description="In-depth implementation instructions.")
    status: Literal["pending", "in-progress", "done", "deferred", "blocked"] = Field("pending", description="Current state of the task.")
    # Dependencies are IDs of tasks (int) or subtasks (str like '1.2')
    dependencies: List[Union[int, str]] = Field([], description="IDs of tasks (int) or subtasks (str like '1.2') this depends on.")
    priority: Literal["high", "medium", "low"] = Field(config.DEFAULT_PRIORITY, description="Importance level of the task.")
    testStrategy: Optional[str] = Field(None, description="Verification approach for the task.")
    subtasks: List[Subtask] = Field([], description="List of smaller, specific subtasks.")

    @field_validator('dependencies', mode='before')
    @classmethod
    def ensure_dependencies_list(cls, v):
         if v is None: return []
         if not isinstance(v, list): raise ValueError("Dependencies must be a list")
         return [str(item) if isinstance(item, str) else int(item) for item in v]

class TaskFileMetadata(BaseModel):
    projectName: str = Field(default=config.PROJECT_NAME)
    projectVersion: str = Field(default=config.PROJECT_VERSION)
    sourceFiles: List[str] = Field([], description="List of input files used to generate tasks.")
    generatedAt: Optional[str] = None
    totalTasks: Optional[int] = None

class TasksData(BaseModel):
    meta: TaskFileMetadata = Field(default_factory=TaskFileMetadata)
    tasks: List[Task] = []

# --- Complexity Analysis Models ---
class ComplexityAnalysisItem(BaseModel):
    model_config = ConfigDict(extra='ignore')

    taskId: int
    taskTitle: str
    complexityScore: float = Field(..., ge=1, le=10, description="Complexity score from 1 to 10.")
    recommendedSubtasks: int = Field(..., ge=1, description="Recommended number of subtasks.")
    expansionPrompt: str = Field(..., description="AI-generated prompt for expanding this task.")
    reasoning: str = Field(..., description="Brief explanation for the complexity assessment.")

class ComplexityReport(BaseModel):
    meta: Dict[str, Any] = Field(..., description="Metadata about the report generation.")
    complexityAnalysis: List[ComplexityAnalysisItem] = Field(..., description="List of complexity analyses for each task.")


# --- Research Workflow Models ---
class ResearchQuestions(BaseModel):
    questions: List[str] = Field(..., description="List of specific research questions generated for a task.")

class ResearchTopics(BaseModel):
    # Dictionary where keys are topic names and values are lists of questions
    topics: Dict[str, List[str]] = Field(..., description="Research questions grouped into relevant topics.")