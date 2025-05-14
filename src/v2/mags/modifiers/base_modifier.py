"""
Base class for all graph modifiers.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
from mags.graph.lkg import LiquidKnowledgeGraph # Use forward reference if circular dependency
from mags.hyperparameters.hyperparameters import Hyperparameters

class ModifierRunType(Enum):
    """
    Defines when a modifier is typically run, although LKG directly uses ModifierPhase.
    This enum can be used by modifier implementations for clarity or internal logic.
    """
    BEFORE_QUERY = "before_query"
    AFTER_QUERY = "after_query"
    DETACHED = "detached"


class BaseModifier(ABC):
    """
    Abstract base class for all modifiers that can alter the LiquidKnowledgeGraph.
    """
    def __init__(self, hyperparams: Hyperparameters, run_type: Optional[ModifierRunType] = None):
        """
        Initializes the base modifier.

        Args:
            hyperparams: The global hyperparameter settings.
            run_type: Indicates the typical phase this modifier runs in.
                      This is more for documentation/categorization as LKG.run_modifiers
                      uses its own ModifierPhase enum and registration.
        """
        self.hyperparams = hyperparams
        self.run_type = run_type
        self._component_name = self.__class__.__name__.lower().replace("modifier", "")

    def _get_param(self, key: str, default: Any = None) -> Any:
        """Helper to get component-specific hyperparameter."""
        return self.hyperparams.get_component_param(self._component_name, key, default)

    @abstractmethod
    def apply(self, graph: LiquidKnowledgeGraph, *args: Any, **kwargs: Any) -> None:
        """
        Applies the modification to the graph.
        The specific arguments required will depend on the modifier.

        Args:
            graph: The LiquidKnowledgeGraph instance to modify.
            *args: Positional arguments specific to the modifier.
            **kwargs: Keyword arguments specific to the modifier.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} run_type={self.run_type}>"