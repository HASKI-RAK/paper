from functools import wraps
from typing import Callable, Optional, TypeVar, ParamSpec, Protocol

from core.experiment_config import ExperimentConfig

P = ParamSpec('P')
R = TypeVar('R')


class ExperimentFunction(Protocol[P, R]):
    """Protocol for experiment functions with metadata."""
    experiment_name: str
    __call__: Callable[P, R]


def experiment(name: str, config: Optional[ExperimentConfig] = None):
    """
    A decorator to mark a function as an experiment with a given name and optional configuration.
    Args:
        name (str): The name of the experiment.
        config (Optional[ExperimentConfig]): An optional configuration object for the experiment. 
                   The config contains input arguments and descriptions. Take a look at `ExperimentConfig.add_input`.
                    If not provided, a default ExperimentConfig with the given name will be used.
    Returns:
        Callable: A decorator that wraps the function, adding experiment metadata to it.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)
        wrapper.experiment_name = name  # type: ignore
        wrapper.is_experiment = True  # type: ignore
        wrapper.config = config or ExperimentConfig(name)  # type: ignore
        return wrapper
    return decorator
