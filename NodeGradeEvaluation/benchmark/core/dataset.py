from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union
import shutil
from enum import Enum, auto
import pandas as pd


class StepStatus(Enum):
    """Status of dataset preparation steps."""
    PENDING = auto()
    COMPLETED = auto()


class Dataset(ABC):
    """Base class for dataset implementations."""

    def __init__(self, name: str, base_path: Optional[Path] = None):
        self.name = name
        self.base_path = base_path or Path("datasets") / name
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.status_dir = self.base_path / ".status"
        self.status_dir.mkdir(exist_ok=True)
        self._cached_df: Optional[pd.DataFrame] = None

    @property
    def path(self) -> Path:
        """Returns path to processed dataset."""
        return self.base_path / "processed"

    def _get_status(self, step: str) -> StepStatus:
        """Get status of a preparation step."""
        if (self.status_dir / f"{step}.done").exists():
            return StepStatus.COMPLETED
        return StepStatus.PENDING

    def _mark_completed(self, step: str) -> None:
        """Mark a step as completed."""
        (self.status_dir / f"{step}.done").touch()

    @abstractmethod
    def download(self) -> None:
        """Download dataset from source."""

    @abstractmethod
    def extract(self) -> None:
        """Extract downloaded dataset."""

    @abstractmethod
    def transform(self) -> None:
        """Transform dataset into final format."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the processed dataset into a pandas DataFrame."""
        pass

    def get_data(self) -> pd.DataFrame:
        """Get dataset as DataFrame, using cache if available."""
        if self._cached_df is None:
            self._cached_df = self.load()
        return self._cached_df

    def cleanup(self) -> None:
        """Remove temporary files."""
        tmp_dir = self.base_path / "tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def setup(self) -> Path:
        """Main pipeline for dataset preparation."""
        if self._get_status("download") == StepStatus.PENDING:
            self.download()
            self._mark_completed("download")

        if self._get_status("extract") == StepStatus.PENDING:
            self.extract()
            self._mark_completed("extract")

        if self._get_status("transform") == StepStatus.PENDING:
            self.transform()
            self._mark_completed("transform")

        if self._get_status("cleanup") == StepStatus.PENDING:
            self.cleanup()
            self._mark_completed("cleanup")

        return self.path

    def is_ready(self) -> bool:
        """Check if all steps are completed."""
        steps = ["download", "extract", "transform", "cleanup"]
        return all(self._get_status(step) == StepStatus.COMPLETED for step in steps)

    def reset(self) -> None:
        """Reset all preparation steps."""
        if self.status_dir.exists():
            shutil.rmtree(self.status_dir)
        self.status_dir.mkdir()


class DatasetManager:
    """Manages dataset registration and retrieval."""

    def __init__(self, project_path: Path):
        self.datasets: Dict[str, Dataset] = {}
        self.project_path = project_path

    def register(self, dataset: Dataset) -> None:
        """Register a dataset."""
        self.datasets[dataset.name] = dataset

    def get(self, name: str) -> Dataset:
        """Get a registered dataset."""
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found")
        return self.datasets[name]
