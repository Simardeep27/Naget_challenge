from __future__ import annotations

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


class NullProgressReporter:
    def __enter__(self) -> "NullProgressReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def update(self, message: str) -> None:
        return None

    def complete(self, message: str = "Done") -> None:
        return None


class CliProgressReporter:
    def __init__(self) -> None:
        self.console = Console(stderr=True)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self.task_id: int | None = None

    def __enter__(self) -> "CliProgressReporter":
        self.progress.start()
        self.task_id = self.progress.add_task("Starting...", total=None)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.stop()

    def update(self, message: str) -> None:
        if self.task_id is not None:
            self.progress.update(self.task_id, description=message)

    def complete(self, message: str = "Done") -> None:
        if self.task_id is not None:
            self.progress.update(self.task_id, description=message)
