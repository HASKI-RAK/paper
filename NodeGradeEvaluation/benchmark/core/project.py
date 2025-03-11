from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Any
from datetime import datetime
import json
import os

from tqdm import tqdm

from core.dataset import DatasetManager
from core.decorators import ExperimentFunction
from core.experiment import Experiment
from core.metrics import MetricFunction, L1Metric


class Project(ABC):
    """Manages research projects and their configurations."""
    experiments: Dict[str, Experiment]

    def __init__(self, name: str, project_root: Path):
        self.name = name
        self.project_root = project_root
        self.base_path = project_root / 'projects' / name
        self.run_history_path = self.base_path / 'run_history'
        self.run_history_path.mkdir(parents=True, exist_ok=True)

        # Update dataset path to be in project root
        self.dataset_path = project_root / 'datasets'
        self.dataset_manager = DatasetManager(self.dataset_path)

        self.experiments = {}
        self.experiment_registry: Dict[str, Callable] = {}
        self._register_experiments()

    def save_run(self, experiment: Experiment, results: Any):
        """Saves the results and parameters of an experiment with a timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{experiment.name}-{timestamp}.json"
        file_path = self.run_history_path / filename

        run_data = {
            "metadata": {
                "timestamp": timestamp,
                "experiment_name": experiment.name
            },
            "parameters": experiment.parameters,
            "results": results
        }

        with open(file_path, 'w') as f:
            json.dump(run_data, f, indent=2)
        print(f"Run saved to {file_path}")

    def load_run(self, experiment_name: str) -> tuple[Any, dict]:
        """Loads the most recent run of an experiment."""
        runs = sorted(self.run_history_path.glob(
            f"{experiment_name}-*.json"), reverse=True)
        if runs:
            with open(runs[0], 'r') as f:
                data = json.load(f)
            print(f"Loaded run from {runs[0]}")
            return data["results"], data["parameters"]
        else:
            print(f"No previous runs found for experiment '{experiment_name}'")
            return None, {}

    def load_specific_run(self, run_name: str) -> tuple[Any, dict]:
        """Loads a specific run based on the provided run name."""
        file_path = self.run_history_path / f"{run_name}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded run from {file_path}")
            return data["results"], data["parameters"]
        else:
            print(f"Run '{run_name}' not found.")
            return None, {}

    def validate_run(self, experiment_name: str, metric: MetricFunction | None = None) -> tuple[float, dict]:
        """
        Validates a run by comparing stored results with a fresh execution.

        Args:
            experiment_name: Name of the experiment to validate
            metric: Optional metric function to compare results (defaults to L1Metric)

        Returns:
            tuple[float, dict]: Difference score and original parameters
        """
        # Load the original run
        stored_results, stored_params = self.load_run(experiment_name)
        if stored_results is None:
            raise ValueError(
                f"No stored run found for experiment '{experiment_name}'")

        # Setup experiment
        experiment = self.setup_experiment(experiment_name)

        # Restore inputs from stored parameters
        for stored_input in stored_params["inputs"]:
            # Convert arguments while preserving strings
            args = []
            for arg in stored_input["args"]:
                try:
                    # Try to evaluate as literal
                    args.append(eval(arg))
                except (NameError, SyntaxError):
                    # If eval fails, keep as string
                    args.append(arg)

            # Convert kwargs while preserving strings
            kwargs = {}
            for k, v in stored_input["kwargs"].items():
                try:
                    kwargs[k] = eval(v)
                except (NameError, SyntaxError):
                    kwargs[k] = v

            experiment.add_input(
                *args, description=stored_input.get("description"), **kwargs)

        # Restore datasets if any
        for dataset_name, dataset_path in stored_params["datasets"].items():
            dataset = self.dataset_manager.get(dataset_name)
            experiment.add_dataset(dataset_name, dataset)

        # Run experiment
        inp = experiment.parameters["inputs"][0]  # TODO: fix this
        new_results = experiment.run(inp)

        # Compare results using the metric
        if metric is None:
            metric = L1Metric()

        difference = metric(new_results, stored_results)

        print("\nValidation Results:")
        print(f"Original results: {stored_results}")
        print(f"New results: {new_results}")
        print(f"Difference (using {metric.__class__.__name__}): {difference}")

        return difference, stored_params

    def _register_experiments(self) -> None:
        """Automatically register all experiment functions."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, 'is_experiment'):
                if not hasattr(attr, 'experiment_name'):
                    raise ValueError(
                        f"Experiment function '{attr_name}' is missing 'experiment_name' attribute")
                self.experiment_registry[
                    attr.experiment_name  # type: ignore
                ] = attr

    def setup_experiment(self, experiment_name: str) -> Experiment:
        if experiment_name not in self.experiment_registry:
            raise ValueError(f"No experiment found with name '{
                             experiment_name}'")

        func = self.experiment_registry[experiment_name]
        exp = Experiment(experiment_name, func)

        # Add configured inputs
        for input_config in func.config.inputs:  # type: ignore
            exp.add_input(description=input_config.description,
                          *input_config.args, **input_config.kwargs)

        # Add configured datasets
        for dataset_name, version in func.config.datasets.items():  # type: ignore
            dataset = self.dataset_manager.get(dataset_name)
            exp.add_dataset(dataset_name, dataset)

        return exp

    def run(self, experiment_name: str) -> Any:
        """Runs an experiment by name or loads existing results."""
        if '-' in experiment_name:  # TODO: fix possible bug because of the - in the name
            # Specific run with timestamp provided
            print(f"Loading specific run '{experiment_name}'")
            return self.load_specific_run(experiment_name)
        else:
            # Run the experiment and save the results
            experiment = self.setup_experiment(experiment_name)
            results = []
            for inp in tqdm(experiment.inputs,
                            desc=f"Running {self.name}",
                            unit='input',
                            colour='green'):
                res = experiment.run(inp)
                results.append(res)
                self.save_run(experiment, res)
            return results

    @abstractmethod
    def output_metrics_and_results(self, results: Any) -> None:
        """Output detailed metrics and results for the experiment."""
        print("Metrics and results:")
        print(results)


class Publisher:
    """Publishes a project in a publication folder."""

    def __init__(self, project: Project, publication_path: Path):
        ...

    def publish(self): ...
