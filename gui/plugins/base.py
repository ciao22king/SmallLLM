"""
Base plugin class for SmallLLM
"""

from typing import Dict, Any, List


class BasePlugin:
    """Base class for all plugins."""
    
    def __init__(self, **config):
        """
        Initialize plugin.
        
        Args:
            **config: Plugin configuration
        """
        self.config = config
        self.name = self.__class__.__name__.replace("Plugin", "")
        self.display_name = self.name.replace("_", " ").title()
        self.description = ""
        self.version = "1.0"
        self.author = "SmallLLM"
        self.icon = "🧩"
    
    def process(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        Process text through the plugin.
        
        Args:
            text: Input text to process
            context: Additional context
        
        Returns:
            Processed text
        """
        if context is None:
            context = {}
        return text
    
    def pre_generate(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Modify the prompt before generation.
        
        Args:
            prompt: User prompt
            context: Generation context
        
        Returns:
            Modified prompt
        """
        return prompt
    
    def post_generate(self, response: str, context: Dict[str, Any] = None) -> str:
        """
        Modify the response after generation.
        
        Args:
            response: AI response
            context: Generation context
        
        Returns:
            Modified response
        """
        return response
    
    def get_actions(self) -> List[Dict[str, Any]]:
        """
        Get list of available actions for this plugin.
        
        Returns:
            List of action dictionaries with name, display_name, and description
        """
        return []
    
    def execute_action(self, action: str, **kwargs) -> Any:
        """
        Execute a plugin action.
        
        Args:
            action: Action name
            **kwargs: Action arguments
        
        Returns:
            Action result
        """
        return None
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get plugin configuration schema.
        
        Returns:
            Dictionary describing the configuration options
        """
        return {}
