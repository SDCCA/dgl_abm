import types
import sys
from src.dgl_abm.plugins import PLUGINS, register, load_external_plugins, register_all_plugins

def test_register_decorator():
    PLUGINS.clear()
    module_name = "namespace"
    test_module = type(sys)(f"dgl_abm.plugins.{module_name}")
    sys.modules[f"dgl_abm.plugins.{module_name}"] = test_module

    def dummy_function():
        return "registered"
    
    dummy_function.__module__ = f"dgl_abm.plugins.{module_name}"
    decorated_function = register("mock_function")(dummy_function)
    decorated_function()

    assert "plugins." + module_name in PLUGINS
    assert "mock_function" in PLUGINS["plugins." + module_name]
    assert PLUGINS["plugins." + module_name]["mock_function"]() == "registered"

def test_register_all_plugins(monkeypatch):
    PLUGINS.clear()
    dummy_module = types.ModuleType("dummy_module")

    def register_plugins(plugins):
        plugins["mock_function"] = lambda: "registered"

    dummy_module.register_plugins = register_plugins
    sys.modules["dummy_module"] = dummy_module
    register_all_plugins()
    
    assert "mock_function" in PLUGINS
    assert PLUGINS["mock_function"]() == "registered"
    del sys.modules["dummy_module"]
