"""
VS Code Integration Plugin for SmallLLM
Connect to VS Code for code analysis and editing
"""

import json
import socket
from typing import Dict, Any, List, Optional
from .base import BasePlugin


class VSCodeIntegrationPlugin(BasePlugin):
    """Plugin for integrating with VS Code."""

    def __init__(self, **config):
        super().__init__(**config)
        self.display_name = "VS Code Integration"
        self.description = "Connect to VS Code for code analysis and editing"
        self.icon = "📁"
        self.version = "1.0"
        self.author = "SmallLLM"

        self.vscode_socket = None
        self.vscode_connected = False
        self.vscode_port = config.get("port", 5001)

    def connect_to_vscode(self) -> bool:
        """Try to connect to VS Code extension."""
        try:
            # Try to connect to local socket
            self.vscode_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.vscode_socket.connect(('localhost', self.vscode_port))
            self.vscode_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to VS Code: {e}")
            self.vscode_connected = False
            return False

    def send_to_vscode(self, command: str, data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Send a command to VS Code."""
        if not self.vscode_connected or not self.vscode_socket:
            if not self.connect_to_vscode():
                return None

        try:
            message = json.dumps({
                "command": command,
                "data": data or {}
            }).encode('utf-8')

            # Send length first (4 bytes)
            self.vscode_socket.send(len(message).to_bytes(4, 'big'))
            # Send message
            self.vscode_socket.send(message)

            # Receive response
            length_bytes = self.vscode_socket.recv(4)
            if not length_bytes:
                return None

            length = int.from_bytes(length_bytes, 'big')
            response = b''
            while len(response) < length:
                chunk = self.vscode_socket.recv(length - len(response))
                if not chunk:
                    break
                response += chunk

            return json.loads(response.decode('utf-8'))

        except Exception as e:
            print(f"Error communicating with VS Code: {e}")
            self.vscode_connected = False
            return None

    def open_in_vscode(self, file_path: str, line: int = 1, column: int = 1) -> bool:
        """Open a file in VS Code at specific position."""
        result = self.send_to_vscode("openFile", {
            "file": file_path,
            "line": line,
            "column": column
        })
        return result is not None

    def get_current_file(self) -> Optional[Dict[str, Any]]:
        """Get the currently open file in VS Code."""
        result = self.send_to_vscode("getCurrentFile")
        return result

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a file from VS Code."""
        result = self.send_to_vscode("getFileContent", {"file": file_path})
        if result and "content" in result:
            return result["content"]
        return None

    def save_file_content(self, file_path: str, content: str) -> bool:
        """Save content to a file in VS Code."""
        result = self.send_to_vscode("saveFileContent", {
            "file": file_path,
            "content": content
        })
        return result is not None

    def run_code(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Run code in VS Code."""
        result = self.send_to_vscode("runCode", {"file": file_path})
        return result

    def analyze_code(self, code: str, language: str = "python") -> Optional[Dict[str, Any]]:
        """Analyze code using VS Code's language server."""
        result = self.send_to_vscode("analyzeCode", {
            "code": code,
            "language": language
        })
        return result

    def pre_generate(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Enhance prompt with VS Code context."""
        if context is None:
            context = {}

        # If there's a current file in VS Code, add context
        current_file = self.get_current_file()
        if current_file:
            file_path = current_file.get("path", "")
            if file_path:
                # Add file context to prompt
                return f"File: {file_path}\n\n{prompt}"

        return prompt

    def post_generate(self, response: str, context: Dict[str, Any] = None) -> str:
        """Add VS Code integration to response."""
        if context is None:
            context = {}

        # Check if response contains code
        if self.contains_code(response):
            response += "\n\n---"
            response += "\n💡 **VS Code Integration Available**"
            response += "\n- Open this code in VS Code"
            response += "\n- Run the code directly"
            response += "\n- Analyze for errors"

        return response

    def contains_code(self, text: str) -> bool:
        """Check if text contains code."""
        code_indicators = [
            "def ", "class ", "import ", "print(", "# ",
            "function ", "const ", "let ", "//",
            "<html", "<div", "<script", "{",
            "public class", "void ", "System.out"
        ]

        return any(indicator in text for indicator in code_indicators)

    def get_actions(self) -> List[Dict[str, Any]]:
        """Get available actions."""
        return [
            {
                "name": "connect",
                "display_name": "Connect to VS Code",
                "description": "Connect to VS Code extension",
                "icon": "🔌"
            },
            {
                "name": "open_file",
                "display_name": "Open File in VS Code",
                "description": "Open a file in VS Code",
                "icon": "📁"
            },
            {
                "name": "get_current_file",
                "display_name": "Get Current File",
                "description": "Get the currently open file from VS Code",
                "icon": "📄"
            },
            {
                "name": "run_code",
                "display_name": "Run Code in VS Code",
                "description": "Run code in VS Code",
                "icon": "▶️"
            },
            {
                "name": "analyze_code",
                "display_name": "Analyze Code",
                "description": "Analyze code using VS Code's language server",
                "icon": "🔍"
            }
        ]

    def execute_action(self, action: str, **kwargs) -> Any:
        """Execute a plugin action."""
        if action == "connect":
            return self.connect_to_vscode()
        elif action == "open_file":
            file_path = kwargs.get("file_path", "")
            line = kwargs.get("line", 1)
            column = kwargs.get("column", 1)
            return self.open_in_vscode(file_path, line, column)
        elif action == "get_current_file":
            return self.get_current_file()
        elif action == "run_code":
            file_path = kwargs.get("file_path", "")
            return self.run_code(file_path)
        elif action == "analyze_code":
            code = kwargs.get("code", "")
            language = kwargs.get("language", "python")
            return self.analyze_code(code, language)
        return None

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "port": {
                "type": "int",
                "default": 5001,
                "description": "VS Code extension port"
            }
        }
