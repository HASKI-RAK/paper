from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Generic, ParamSpec
from tqdm import tqdm

from core.dataset import Dataset

# Define Function as a type alias for a callable

# ParamSpec is used to capture all parameters of a function
P = ParamSpec('P')
R = TypeVar('R')


class Experiment(Generic[P, R]):
    """A single experiment within a project."""

    def __init__(self, name: str, function: Callable[P, R]):
        self.name = name
        self.inputs = []
        self.function = function
        self.datasets: Dict[str, Path] = {}
        self.parameters = {
            "function_name": function.__name__,
            "function_doc": function.__doc__,
            "inputs": [],
            "datasets": {}
        }

    def add_dataset(self, name: str, dataset: Dataset) -> None:
        """Add a dataset to the experiment."""
        path = dataset.setup()
        self.datasets[name] = path
        # Add dataset metadata
        self.parameters["datasets"][name] = {
            "path": str(path),
            "name": dataset.name
        }

    def add_input(self, *args: Any, description: Optional[str] = None, **kwargs: Any):
        """Adds input arguments to the experiment."""
        self.inputs.append((args, kwargs))
        # Store parameters with description
        param_dict = {
            "args": [str(arg) for arg in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "description": description
        }
        self.parameters["inputs"].append(param_dict)

    def run(self, input) -> list[R]:
        """Runs the experiment with the stored function"""
        args, kwargs = input
        return self.function(*args, **kwargs)  # type: ignore
