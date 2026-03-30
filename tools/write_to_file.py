from pathlib import Path

from atomic_agents.base.base_io_schema import BaseIOSchema
from atomic_agents.base.base_tool import BaseTool, BaseToolConfig
from pydantic import Field


class WriteToFileToolInputSchema(BaseIOSchema):
    """Input payload for the tool."""

    path: str = Field(..., description="write file path")
    content: str = Field(..., description="contents of the file to be written")


class WriteToFileToolOutputSchema(BaseIOSchema):
    """Output payload for the tool."""

    absolute_path: str = Field(..., description="absolute path of the written file")


class WriteToFileTool(
    BaseTool[WriteToFileToolInputSchema, WriteToFileToolOutputSchema]
):
    def __init__(self, config: BaseToolConfig = BaseToolConfig()):
        super().__init__(config)

    def run(self, params: WriteToFileToolInputSchema) -> WriteToFileToolOutputSchema:
        file_path = Path(params.path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(params.content)

        return WriteToFileToolOutputSchema(absolute_path=str(file_path.resolve()))