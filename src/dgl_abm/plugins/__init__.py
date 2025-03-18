import importlib
import pkgutil
import sys
from pathlib import Path

PLUGINS = {}

def register(plugin_name):
    """Register a custom model type or additional functionalities."""
    def decorator(func):
        PLUGINS[plugin_name] = func
        return func
    return decorator

def load_internal_plugins():
    """Import any plugins available in the 'dgl_abm/plugins/' directory."""
    try:
        plugin_path = Path(__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(plugin_path)]):
            try:
                importlib.import_module(f"dgl_abm.plugins.{module_name}")
            except Exception as e:
                print(f"Error loading plugin {module_name}: {e}")
    except Exception as e:
        print(f"Error scanning internal plugins: {e}")
    
def load_external_plugins():
    """Load any system-wide external plugins with the name prefix 'dgl_abm_'."""
    for _, name, _ in pkgutil.iter_modules():
        if name.startswith("dgl_abm_"):
            importlib.import_module(name)

def register_all_plugins():
    """Support function-based registration in addition to decorator pattern."""
    for module in sys.modules.values():
        if hasattr(module, "register_plugins"):
            module.register_plugins(PLUGINS)

# Only load plugins when this module is imported (not when run as main script)
if __name__ != "__main__":
    load_internal_plugins()
    load_external_plugins()
    register_all_plugins()

