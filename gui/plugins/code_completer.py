"""
Code Completer Plugin for SmallLLM
Auto-complete code in AI responses
"""

import re
from typing import Dict, Any, List
from .base import BasePlugin


class CodeCompleterPlugin(BasePlugin):
    """Plugin for auto-completing code in responses."""

    def __init__(self, **config):
        super().__init__(**config)
        self.display_name = "Code Completer"
        self.description = "Auto-complete code snippets in AI responses"
        self.icon = "💻"
        self.version = "1.0"
        self.author = "SmallLLM"

        # Code snippets library
        self.snippets = {
            "python": {
                "import": "import {name}\n",
                "function": "def {name}({args}):\n    {body}\n",
                "class": "class {name}:\n    def __init__(self):\n        {body}\n",
                "print": "print({value})\n",
                "if": "if {condition}:\n    {body}\n",
                "for": "for {var} in {iterable}:\n    {body}\n",
                "while": "while {condition}:\n    {body}\n",
                "try": "try:\n    {body}\nexcept {error}:\n    {handler}\n",
            },
            "javascript": {
                "import": "import {name} from '{module}';\n",
                "function": "function {name}({args}) {\n  {body}\n}\n",
                "class": "class {name} {\n  constructor() {\n    {body}\n  }\n}\n",
                "console": "console.log({value});\n",
                "if": "if ({condition}) {\n  {body}\n}\n",
                "for": "for (let {var} of {iterable}) {\n  {body}\n}\n",
            },
            "html": {
                "div": "<div class=\"{classes}\">{content}</div>\n",
                "button": "<button onclick=\"{handler}\">{text}</button>\n",
                "form": "<form action=\"{action}\" method=\"{method}\">\n  {fields}\n</form>\n",
            },
            "css": {
                "class": ".{name} {\n  {properties}\n}\n",
                "id": "#{name} {\n  {properties}\n}\n",
            }
        }

    def post_generate(self, response: str, context: Dict[str, Any] = None) -> str:
        """Complete code snippets in the response."""
        if context is None:
            context = {}

        # Detect programming language from context
        language = context.get("language", self.detect_language(response))

        if language not in self.snippets:
            return response

        # Complete code snippets
        return self.complete_code(response, language)

    def detect_language(self, text: str) -> str:
        """Detect programming language from text."""
        # Simple detection based on keywords
        python_keywords = ["def ", "class ", "import ", "print(", "self."]
        js_keywords = ["function ", "const ", "let ", "console.", "=>"]
        html_keywords = ["<div", "<html", "<body", "<button"]
        css_keywords = [".class", "#id", "{", "color:"]

        text_lower = text.lower()

        if any(kw in text_lower for kw in python_keywords):
            return "python"
        elif any(kw in text_lower for kw in js_keywords):
            return "javascript"
        elif any(kw in text_lower for kw in html_keywords):
            return "html"
        elif any(kw in text_lower for kw in css_keywords):
            return "css"

        return "python"  # Default

    def complete_code(self, text: str, language: str) -> str:
        """Complete code snippets in text."""
        # This is a simple implementation
        # A real implementation would use a more sophisticated approach

        # Look for incomplete code blocks
        lines = text.split('\n')
        completed_lines = []

        for i, line in enumerate(lines):
            # Check for incomplete function definitions
            if language == "python" and line.strip().startswith("def ") and "(" in line and ")" not in line:
                # Try to complete the function
                completed = self.complete_python_function(line)
                completed_lines.append(completed)
            elif language == "javascript" and line.strip().startswith("function ") and "{" not in line:
                completed = self.complete_js_function(line)
                completed_lines.append(completed)
            else:
                completed_lines.append(line)

        return '\n'.join(completed_lines)

    def complete_python_function(self, line: str) -> str:
        """Complete a Python function definition."""
        # Extract function name and parameters
        match = re.match(r'def\s+(\w+)\s*\(', line)
        if match:
            # Add closing parenthesis and colon
            if ')' not in line:
                line += "):"
            elif ':' not in line:
                line += ":"
        return line

    def complete_js_function(self, line: str) -> str:
        """Complete a JavaScript function definition."""
        if ')' not in line:
            line += ")"
        if '{' not in line:
            line += " {"
        return line

    def get_actions(self) -> List[Dict[str, Any]]:
        """Get available actions."""
        return [
            {
                "name": "complete_code",
                "display_name": "Complete Code",
                "description": "Complete code snippets in the response",
                "icon": "💻"
            },
            {
                "name": "explain_code",
                "display_name": "Explain Code",
                "description": "Add explanations to code",
                "icon": "📚"
            },
            {
                "name": "format_code",
                "display_name": "Format Code",
                "description": "Format code in the response",
                "icon": "✨"
            }
        ]

    def execute_action(self, action: str, **kwargs) -> Any:
        """Execute a plugin action."""
        if action == "complete_code":
            text = kwargs.get("text", "")
            language = kwargs.get("language", "python")
            return self.complete_code(text, language)
        elif action == "explain_code":
            text = kwargs.get("text", "")
            return self.explain_code(text)
        elif action == "format_code":
            text = kwargs.get("text", "")
            return self.format_code(text)
        return None

    def explain_code(self, text: str) -> str:
        """Add explanations to code."""
        lines = text.split('\n')
        explained = []

        for line in lines:
            explained.append(line)
            # Add simple explanations
            if line.strip().startswith("def "):
                explained.append("    # Function definition")
            elif line.strip().startswith("class "):
                explained.append("    # Class definition")
            elif line.strip().startswith("import "):
                explained.append("    # Import statement")
            elif line.strip().startswith("if "):
                explained.append("    # Conditional statement")
            elif line.strip().startswith("for "):
                explained.append("    # Loop")

        return '\n'.join(explained)

    def format_code(self, text: str) -> str:
        """Format code (simple implementation)."""
        # Simple formatting - a real implementation would use a proper formatter
        lines = text.split('\n')
        formatted = []

        for line in lines:
            # Remove trailing whitespace
            formatted_line = line.rstrip()
            formatted.append(formatted_line)

        return '\n'.join(formatted)
