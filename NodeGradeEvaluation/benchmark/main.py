import argparse
import importlib
from pathlib import Path
from typing import Type, Dict

from core.project import Project


def load_projects(
    main: str
) -> Dict[str, Type[Project]]:
    """Dynamically load all project classes from the projects directory."""
    projects = {}
    projects_dir = Path(__file__).parent / 'projects'

    for project_dir in projects_dir.iterdir():
        if project_dir.is_dir() and not project_dir.name.startswith('_'):
            try:
                # Import the project module
                module_path = f"projects.{project_dir.name}" + f".{main}"
                module = importlib.import_module(module_path)

                # Look for a class that inherits from Project
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Project) and attr != Project:
                        projects[project_dir.name] = attr
                        break
            except ImportError as e:
                print(f"Warning: Could not load project {
                      project_dir.name}: {e}")

    return projects


if __name__ == "__main__":
    # Define project root at the top level
    PROJECT_ROOT = Path(__file__).parent.absolute()
    
    parser = argparse.ArgumentParser(description='Run a specific project')
    parser.add_argument('project_name', type=str,
                        help='Name of the project to run')
    parser.add_argument('experiment_name', type=str,
                        help='Name of the experiment to run')
    parser.add_argument('--main', type=str, default='main',
                        help='Name of the main module inside your project to import')
    parser.add_argument('--validate', action='store_true',
                        help='Validate the experiment by comparing with previous run')
    parser.add_argument('--evaluate', action='store_true',
                        help='Output detailed metrics and results for the experiment')

    args = parser.parse_args()

    # Load available projects
    available_projects = load_projects(args.main)

    if args.project_name not in available_projects:
        raise ValueError(f"Project '{args.project_name}' not found. Available projects: {
                         list(available_projects.keys())}")

    # Create instance of specific project class with root path
    project_class = available_projects[args.project_name]
    project = project_class(args.project_name, project_root=PROJECT_ROOT)

    if args.validate:
        difference, params = project.validate_run(args.experiment_name)
        if difference == 0:
            print("\n✅ Validation successful: Results are identical")
        else:
            print(f"\n⚠️  Validation failed: Results differ by {difference}")
    else:
        result = project.run(args.experiment_name)
        if args.evaluate:
            if hasattr(project, 'output_metrics_and_results'):
                project.output_metrics_and_results(result)
            else:
                print("Project does not support detailed evaluation")
