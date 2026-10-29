"""
SmallLLM GUI - Advanced Graphical Interface
"""

__version__ = "1.0.0"

from .main_window import MainWindow, main
from .plugin_manager import PluginManager
from .styles import apply_stylesheet

__all__ = ["MainWindow", "PluginManager", "apply_stylesheet", "main"]
