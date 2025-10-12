"""dgl_abm.plugins - Infrastructure for connecting plugins.

Function(s):
- register: Decorator to register a plugin
- load_internal_plugins: Load plugins from the internal plugins directory
- load_external_plugins: Load external plugins with the 'dgl_abm_' prefix
- register_all_plugins: Support function-based registration
"""

import importlib
import logging
import pkgutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PLUGINS = {}

logger = logging.getLogger(__name__)


def register(plugin_name: str) -> Callable[[Any], Any]:
    """Register a custom model type or additional functionalities.

    Args:
        plugin_name (str): Name of the plugin to register

    Returns:
        decorator (Callable): Decorator function to register the plugin

    Notes:
        Use this function as a decorator to be able to call a function
        in dgl-abm that is not part of the core package.
        E.g.,
        tools.py
        from dgl_abm.plugins import register

        @register("special_function")
        def custom_function(x):
            return x * 2

        This will register the function `custom_function` under the name
        "special_function". You can then call it via
        PLUGINS[plugins.tools]["special_function"](x) in, for example, step.py.
    """

    def decorator(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        namespace = function.__module__.split(".")[-2:]
        namespace_key = ".".join(namespace)
        if namespace_key not in PLUGINS:
            PLUGINS[namespace_key] = {}
        if plugin_name in PLUGINS[namespace_key]:
            message = f"Plugin '{plugin_name}' already in '{namespace_key}'"
            raise ValueError(message)
        PLUGINS[namespace_key][plugin_name] = function
        return function

    return decorator


def load_internal_plugins() -> None:
    """Import any plugins available in the 'dgl_abm/plugins/' directory."""
    try:
        plugin_path = Path(__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(plugin_path)]):
            try:
                importlib.import_module(f"dgl_abm.plugins.{module_name}")
            except ImportError:
                logger.exception("Error occurred loading plugin %s", module_name)
    except (OSError, ImportError):
        logger.exception("Error occurred scanning internal plugins")


def load_external_plugins() -> None:
    """Load any system-wide external plugins with the name prefix 'dgl_abm_'."""
    for _, name, _ in pkgutil.iter_modules():
        if name.startswith("dgl_abm_"):
            importlib.import_module(name)


def register_all_plugins() -> None:
    """Support function-based registration in addition to decorator pattern."""
    for module in sys.modules.values():
        if hasattr(module, "register_plugins"):
            module.register_plugins(PLUGINS)


# Only load plugins when this module is imported (not when run as main script)
if __name__ != "__main__":
    load_internal_plugins()
    load_external_plugins()
    register_all_plugins()
