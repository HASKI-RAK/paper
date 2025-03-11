"""
This example shows an easy-to-understand project on how to use the framework to download a dataset,
setup experiments and run them, and evaluate the results.
Feel free to use this as a template for your own projects.
    Please read the README.md file for more information.
"""
import json
from typing import Dict

from core.dataset import Dataset
from core.decorators import experiment
from core.experiment_config import ExperimentConfig
from core.project import Project
from projects.example_project.custom_fibonacci import calculate_fibonacci


class ExampleProject(Project):
    """An example project that calculates Fibonacci numbers and evaluates the results.

    Create a custom project by inheriting from the Project class.
    Add your custom experiments using the @experiment decorator.
    """

    def __init__(self, name: str):
        super().__init__(name)
        # This is an example of how to register a dataset.
        # You can specify details (how it is downloaded, extracted, etc.)
        # in a custom dataset class
        self.dataset_manager.register(FibonacciDataset("fibonacci"))

    @experiment("fibonacci_basic",
                ExperimentConfig("fibonacci_basic")
                .add_input(10, description="Calculate 10th Fibonacci number")
                .add_input(20, description="Calculate 20th Fibonacci number"))
    def run_fibonacci_basic(self, n: int) -> int:
        """Calculate basic Fibonacci numbers."""
        # This is a custom function (your function)
        return calculate_fibonacci(n)

    @experiment("fibonacci_dataset",
                ExperimentConfig("fibonacci_dataset")
                .add_input(description="Calculate Fibonacci numbers from dataset"))
    def run_fibonacci_dataset(self) -> int:
        """Calculate Fibonacci numbers from dataset."""
        # This is how you can access a dataset
        dataset = self.dataset_manager.get("fibonacci")
        with open(dataset.path / "fibonacci.json", "r", encoding='utf-8') as f:
            numbers = json.load(f)
        return numbers

    def evaluate(self, results: list[int]) -> Dict[str, float]:
        """Evaluates the results of the project."""
        # Implement evaluation logic if needed
        return {'result': results[0]}  # Example

    def show_output(self, results: list[int]) -> None:
        """Shows the results of the project."""
        # This is printed at the end of the run
        print(f'The result is: {results[0]}')


# Example implementation
class FibonacciDataset(Dataset):
    """Example dataset that generates Fibonacci numbers."""
    # You can implement custom methods to download, extract, and transform the dataset.
    # After transformation, a file named .ready is created in the dataset directory.
    # This file is used to check whether to re-download and re-extract, etc. the dataset.

    def download(self) -> None:
        # Simulate download
        (self.base_path / "tmp").mkdir(exist_ok=True)

    def extract(self) -> None:
        # Simulate extraction
        (self.base_path / "processed").mkdir(exist_ok=True)

    def transform(self) -> None:
        # You can use this to perform transformations on the dataset to format it correctly
        # Generate example data
        numbers = [0, 1]
        for i in range(2, 100):
            numbers.append(numbers[i-1] + numbers[i-2])

        with open(self.path / "fibonacci.json", "w", encoding='utf-8') as f:
            json.dump(numbers, f)
