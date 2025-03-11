# Paper Evaluation Suite

A framework for managing research projects and running experiments.
Here used for the NodeGrade evaluation.

## Project Structure
```
paper-evaluation-suite/
├── core/                 # Core framework classes
├── datasets/             # Dataset directory
├── projects/            # Custom projects directory
│   └── example_project/ # Example project implementation
│       ├── __init__.py
│       ├── custom_fibonacci.py
│       ├── main.py
│       └── run_history/
└── main.py             # CLI entry point
```

## Adding a New Project

1. Create a new directory under `projects/` with your project name:
```sh
mkdir projects/my_project
```

2. Create the following files in your project directory:



main.py


- Contains your project class with experiment definitions
- Additional module files for your project's functionality

3. Define experiments using the `@experiment` decorator:

```python
from typing import Dict
from core.project import Project
from core.experiment import Experiment
from core.decorators import experiment
from core.experiment_config import ExperimentConfig

class MyProject(Project):
    def __init__(self, name: str):
        super().__init__(name)
    
    @experiment("my_experiment",
               ExperimentConfig("my_experiment")
               .add_input(10, description="Process with value 10")
               .add_input(20, description="Process with value 20"))
    def run_my_experiment(self, n: int) -> int:
        """This docstring explains the experiment's purpose."""
        return n * 2

    def evaluate(self, results: list[int]) -> Dict[str, float]:
        """Evaluates the results of the project."""
        return {'result': results[0]}
```

### Experiment Configuration

- Use `@experiment` decorator to define experiments
- Provide experiment name and optional configuration
- Add inputs with descriptions using `ExperimentConfig`
- All parameters are automatically tracked in run history


## Running Projects
Create a new environment and install the requirements:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Use the CLI interface in `main.py` to run projects:

```sh
python main.py <project_name> <experiment_name> [--main MAIN] [--validate]
```
To NodeGrade evaluation:
```sh
python main.py ng test
```

Example:
```sh
# Run an experiment
python main.py example_project test

# Load an experiment
python main.py example_project test-20240204 # Timestamp is found in the run_history folder

# Validate an existing run
python main.py example_project test --validate
```

Optional arguments:
- `--main`: Specify a different main module name (default: 'main')


## Example Project Output

```json
{
  "metadata": {
    "timestamp": "20241210183649",
    "experiment_name": "my_experiment"
  },
  "parameters": {
    "function_name": "run_my_experiment",
    "function_doc": "This docstring explains the experiment's purpose.",
    "inputs": [
      {
        "args": [10],
        "kwargs": {},
        "description": "Process with value 10"
      }
    ],
    "datasets": {}
  },
  "results": [20]
}
```

## Dataset Management
You can define a custm dataset using the `Dataset` class.
Implement its functions to download, extract, transform and clean the dataset.
Each step generates a .* file in the .ready directory inside the dataset folder.
If you want to redo a step, you can delete the corresponding file and run the experiment again.

## Example Project

The `example_project` demonstrates:
- Project class implementation in `main.py`
- Custom function implementation in `custom_fibonacci.py`
- Integration with the `Experiment` framework
- Run history tracking in `run_history.py`

## Key Components

- `Project`: Base class for all projects
- `Experiment`: Handles individual experiment runs
- `MetricFunction`: Protocol for implementing evaluation metrics
- `RunHistory`: Tracks the history of experiment runs

## Requirements

- Python 3.12
- Type hints support