"""
Built-in plugins for SmallLLM GUI
"""

from .base import BasePlugin
from .code_completer import CodeCompleterPlugin
from .vscode_integration import VSCodeIntegrationPlugin
from .web_search import WebSearchPlugin
from .math_solver import MathSolverPlugin
from .translator import TranslatorPlugin

__all__ = [
    "BasePlugin",
    "CodeCompleterPlugin",
    "VSCodeIntegrationPlugin",
    "WebSearchPlugin",
    "MathSolverPlugin",
    "TranslatorPlugin"
]
