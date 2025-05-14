"""
Hyperparameters module for managing configuration across the MAGS system.
"""
from typing import Any, Dict, Optional

class Hyperparameters:
    """
    A flexible class to manage hyperparameters with default values.
    Allows different parts of the system to define and access their specific
    hyperparameters.
    """
    _instance: Optional['Hyperparameters'] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Hyperparameters, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, default_params: Optional[Dict[str, Any]] = None):
        if self._initialized:
            return
        self._params: Dict[str, Any] = {}
        if default_params:
            self._params.update(default_params)
        self._initialized = True

    def set_param(self, key: str, value: Any) -> None:
        """Sets a hyperparameter."""
        self._params[key] = value

    def get_param(self, key: str, default: Optional[Any] = None) -> Any:
        """Gets a hyperparameter, returning a default if not found."""
        return self._params.get(key, default)

    def register_defaults(self, component_name: str, defaults: Dict[str, Any]) -> None:
        """
        Registers default hyperparameters for a specific component.
        These are stored namespaced by the component name.
        Example: hyperparams.register_defaults("lkg", {"max_nodes": 1000})
        """
        for key, value in defaults.items():
            full_key = f"{component_name}.{key}"
            if full_key not in self._params:
                self._params[full_key] = value

    def get_component_param(self, component_name: str, key: str, default: Optional[Any] = None) -> Any:
        """
        Gets a hyperparameter for a specific component.
        Example: hyperparams.get_component_param("lkg", "max_nodes")
        """
        full_key = f"{component_name}.{key}"
        return self._params.get(full_key, default)

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """Updates multiple hyperparameters at once."""
        self._params.update(new_params)

    def __repr__(self) -> str:
        return f"<Hyperparameters params={self._params}>"

# Global instance (Singleton-like access, but can be instantiated for tests)
# This allows any module to import and use `global_hyperparameters`
# However, it's generally better to pass Hyperparameter instances explicitly.
# global_hyperparameters = Hyperparameters()