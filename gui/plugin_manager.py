"""
Plugin Manager for SmallLLM GUI
Handles loading, managing, and running plugins
"""

import json
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback


class PluginSpec:
    """Plugin specification."""

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        version: str,
        author: str,
        enabled: bool = True,
        module: Optional[str] = None,
        class_name: Optional[str] = None,
        icon: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.version = version
        self.author = author
        self.enabled = enabled
        self.module = module
        self.class_name = class_name
        self.icon = icon or "🧩"
        self.config = config or {}


class PluginManager:
    """Manages plugins for SmallLLM GUI."""

    # Map built-in plugin name -> "module:ClassName" import target
    BUILTIN_PLUGIN_CLASSES = {
        "code_completer": "code_completer:CodeCompleterPlugin",
        "vscode_integration": "vscode_integration:VSCodeIntegrationPlugin",
        "web_search": "web_search:WebSearchPlugin",
        "math_solver": "math_solver:MathSolverPlugin",
        "translator": "translator:TranslatorPlugin",
    }

    def __init__(self, plugins_dir: str = "plugins"):
        """
        Initialize plugin manager.

        Args:
            plugins_dir: Directory to look for third-party plugins
        """
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, PluginSpec] = {}
        self.loaded_plugins: Dict[str, Any] = {}

        # Built-in plugins
        self.builtin_plugins = {
            "code_completer": {
                "display_name": "Code Completer",
                "description": "Auto-complete code in responses",
                "version": "1.0",
                "author": "SmallLLM",
                "icon": "💻",
                "enabled": False
            },
            "vscode_integration": {
                "display_name": "VS Code Integration",
                "description": "Connect to VS Code for code analysis",
                "version": "1.0",
                "author": "SmallLLM",
                "icon": "📁",
                "enabled": False
            },
            "web_search": {
                "display_name": "Web Search",
                "description": "Search the web for up-to-date information",
                "version": "1.0",
                "author": "SmallLLM",
                "icon": "🌐",
                "enabled": False
            },
            "math_solver": {
                "display_name": "Math Solver",
                "description": "Solve mathematical problems",
                "version": "1.0",
                "author": "SmallLLM",
                "icon": "➕",
                "enabled": False
            },
            "translator": {
                "display_name": "Translator",
                "description": "Translate text between languages",
                "version": "1.0",
                "author": "SmallLLM",
                "icon": "🌍",
                "enabled": False
            }
        }

        # Load plugins
        self.load_plugins()

    def load_plugins(self):
        """Load all available plugins."""
        # Add built-in plugins
        for name, spec in self.builtin_plugins.items():
            self.plugins[name] = PluginSpec(
                name=name,
                display_name=spec["display_name"],
                description=spec["description"],
                version=spec["version"],
                author=spec["author"],
                enabled=spec.get("enabled", False),
                icon=spec.get("icon", "🧩")
            )

        # Load from plugins directory (only if it exists)
        if self.plugins_dir.is_dir():
            for plugin_dir in self.plugins_dir.iterdir():
                if plugin_dir.is_dir():
                    self.load_plugin_from_dir(plugin_dir)

    def load_plugin_from_dir(self, plugin_dir: Path):
        """Load a plugin from directory."""
        try:
            # Look for plugin.json
            plugin_json = plugin_dir / "plugin.json"
            if not plugin_json.exists():
                return

            with open(plugin_json, 'r', encoding='utf-8') as f:
                spec_data = json.load(f)

            plugin_spec = PluginSpec(
                name=spec_data.get("name", plugin_dir.name),
                display_name=spec_data.get("display_name", plugin_dir.name),
                description=spec_data.get("description", ""),
                version=spec_data.get("version", "1.0"),
                author=spec_data.get("author", "Unknown"),
                enabled=spec_data.get("enabled", True),
                module=spec_data.get("module"),
                class_name=spec_data.get("class_name"),
                icon=spec_data.get("icon", "🧩"),
                config=spec_data.get("config", {})
            )

            self.plugins[plugin_spec.name] = plugin_spec

        except Exception as e:
            print(f"Error loading plugin {plugin_dir.name}: {e}")

    def get_plugin_list(self) -> List[PluginSpec]:
        """Get list of all plugins."""
        return list(self.plugins.values())

    def get_enabled_plugins(self) -> List[PluginSpec]:
        """Get list of enabled plugins."""
        return [p for p in self.plugins.values() if p.enabled]

    def enable_plugin(self, name: str, enabled: bool = True):
        """Enable or disable a plugin."""
        if name in self.plugins:
            self.plugins[name].enabled = enabled
            return True
        return False

    def load_plugin(self, name: str) -> Optional[Any]:
        """Load and instantiate a plugin."""
        if name not in self.plugins:
            return None

        if name in self.loaded_plugins:
            return self.loaded_plugins[name]

        plugin_spec = self.plugins[name]
        instance = None
        try:
            instance = self._load_external_plugin(name, plugin_spec)
            if instance is None:
                instance = self._load_builtin_plugin(name, plugin_spec)
        except Exception as e:
            print(f"Error loading plugin {name}: {e}")
            traceback.print_exc()

        if instance is not None:
            self.loaded_plugins[name] = instance
        return instance

    def _load_external_plugin(self, name: str, plugin_spec: PluginSpec) -> Optional[Any]:
        """Load a plugin from a file declared in its plugin.json."""
        if not plugin_spec.module:
            return None

        module_path = self.plugins_dir / name / plugin_spec.module
        if not module_path.exists():
            return None

        spec = importlib.util.spec_from_file_location(f"{name}.plugin", str(module_path))
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.plugin"] = module
        spec.loader.exec_module(module)

        if plugin_spec.class_name and hasattr(module, plugin_spec.class_name):
            plugin_class = getattr(module, plugin_spec.class_name)
            return plugin_class(**plugin_spec.config)
        return None

    def _load_builtin_plugin(self, name: str, plugin_spec: PluginSpec) -> Optional[Any]:
        """Instantiate one of the bundled plugins."""
        target = self.BUILTIN_PLUGIN_CLASSES.get(name)
        if not target:
            return None

        module_name, class_name = target.split(":")
        module = importlib.import_module(f".plugins.{module_name}", package=__package__)
        plugin_class = getattr(module, class_name)
        return plugin_class(**plugin_spec.config)

    def unload_plugin(self, name: str):
        """Unload a plugin."""
        if name in self.loaded_plugins:
            del self.loaded_plugins[name]

    def apply_plugins(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        Apply all enabled plugins to text.

        Args:
            text: Input text
            context: Additional context for plugins

        Returns:
            Processed text
        """
        if context is None:
            context = {}

        result = text

        for plugin_spec in self.get_enabled_plugins():
            plugin = self.load_plugin(plugin_spec.name)
            if plugin and hasattr(plugin, 'process'):
                try:
                    result = plugin.process(result, context)
                except Exception as e:
                    print(f"Error in plugin {plugin_spec.name}: {e}")

        return result

    def pre_generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Apply pre-generation plugins.

        Args:
            prompt: User prompt
            context: Generation context

        Returns:
            Modified prompt
        """
        result = prompt

        for plugin_spec in self.get_enabled_plugins():
            plugin = self.load_plugin(plugin_spec.name)
            if plugin and hasattr(plugin, 'pre_generate'):
                try:
                    result = plugin.pre_generate(result, context)
                except Exception as e:
                    print(f"Error in plugin {plugin_spec.name} pre_generate: {e}")

        return result

    def post_generate(self, response: str, context: Dict[str, Any]) -> str:
        """
        Apply post-generation plugins.

        Args:
            response: AI response
            context: Generation context

        Returns:
            Modified response
        """
        result = response

        for plugin_spec in self.get_enabled_plugins():
            plugin = self.load_plugin(plugin_spec.name)
            if plugin and hasattr(plugin, 'post_generate'):
                try:
                    result = plugin.post_generate(result, context)
                except Exception as e:
                    print(f"Error in plugin {plugin_spec.name} post_generate: {e}")

        return result

    def get_plugin_actions(self, plugin_name: str) -> List[Dict[str, Any]]:
        """Get available actions for a plugin."""
        plugin = self.load_plugin(plugin_name)
        if plugin and hasattr(plugin, 'get_actions'):
            return plugin.get_actions()
        return []

    def execute_plugin_action(self, plugin_name: str, action: str, **kwargs) -> Any:
        """Execute a plugin action."""
        plugin = self.load_plugin(plugin_name)
        if plugin and hasattr(plugin, 'execute_action'):
            return plugin.execute_action(action, **kwargs)
        return None
