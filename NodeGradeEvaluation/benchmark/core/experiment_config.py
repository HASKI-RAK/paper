from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InputConfig:
    """Configuration for experiment inputs"""
    args: List[Any]
    kwargs: Dict[str, Any]
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "args": [str(arg) for arg in self.args],
            "kwargs": {k: str(v) for k, v in self.kwargs.items()},
            "description": self.description
        }


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    name: str
    inputs: List[InputConfig] = field(default_factory=list)
    datasets: Dict[str, str] = field(default_factory=dict)

    def add_input(self, *args: Any, description: Optional[str] = None, **kwargs: Any) -> 'ExperimentConfig':
        self.inputs.append(InputConfig(
            args=list(args), kwargs=kwargs, description=description))
        return self

    def add_dataset(self, name: str, version: str) -> 'ExperimentConfig':
        self.datasets[name] = version
        return self
