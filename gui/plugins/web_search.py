"""
Web Search Plugin for SmallLLM
Search the web for up-to-date information
"""

import requests
from typing import Dict, Any, List
from .base import BasePlugin


class WebSearchPlugin(BasePlugin):
    """Plugin for web search functionality."""

    def __init__(self, **config):
        super().__init__(**config)
        self.display_name = "Web Search"
        self.description = "Search the web for up-to-date information"
        self.icon = "🌐"
        self.version = "1.0"
        self.author = "SmallLLM"

        # Configuration
        self.search_engine = config.get("search_engine", "google")
        self.api_key = config.get("api_key", "")
        self.max_results = config.get("max_results", 3)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search the web for a query."""
        if self.search_engine == "google":
            return self.google_search(query)
        elif self.search_engine == "bing":
            return self.bing_search(query)
        elif self.search_engine == "duckduckgo":
            return self.duckduckgo_search(query)
        else:
            return []

    def google_search(self, query: str) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        if not self.api_key:
            print("Google Custom Search API key not configured")
            return []

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": "017576662512468239146:4g8x4qyqc1a",  # Example CX
                "q": query,
                "num": self.max_results
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []

                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "Google"
                    })

                return results

        except Exception as e:
            print(f"Google search error: {e}")
            return []

    def bing_search(self, query: str) -> List[Dict[str, Any]]:
        """Search using Bing Search API."""
        if not self.api_key:
            print("Bing Search API key not configured")
            return []

        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {"q": query, "count": self.max_results}

            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []

                for item in data.get("webPages", {}).get("value", []):
                    results.append({
                        "title": item.get("name", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "Bing"
                    })

                return results

        except Exception as e:
            print(f"Bing search error: {e}")
            return []

    def duckduckgo_search(self, query: str) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (no API key required)."""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []

                # Abstract
                if "Abstract" in data and data["Abstract"]:
                    results.append({
                        "title": "Summary",
                        "link": data.get("AbstractURL", ""),
                        "snippet": data["Abstract"],
                        "source": "DuckDuckGo"
                    })

                # Related topics
                for topic in data.get("RelatedTopics", []):
                    if "Text" in topic:
                        results.append({
                            "title": topic.get("FirstURL", "").split('/')[-1] if topic.get("FirstURL") else "Info",
                            "link": topic.get("FirstURL", ""),
                            "snippet": topic["Text"],
                            "source": "DuckDuckGo"
                        })

                return results[:self.max_results]

        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []

    def pre_generate(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Add web search capability to prompt."""
        if context is None:
            context = {}

        # Check if prompt requests web search
        search_phrases = [
            "search the web",
            "look up",
            "find information",
            "what is the latest",
            "recent news",
            "current information",
            "search for",
            "google"
        ]

        prompt_lower = prompt.lower()
        for phrase in search_phrases:
            if phrase in prompt_lower:
                # Extract search query
                query = self.extract_search_query(prompt, phrase)
                if query:
                    # Perform search
                    results = self.search(query)
                    if results:
                        # Add search results to context
                        context["web_search_results"] = results
                        # Modify prompt to include search capability
                        return f"{prompt}\n\n[Web search enabled for: {query}]"

        return prompt

    def extract_search_query(self, prompt: str, phrase: str) -> str:
        """Extract search query from prompt."""
        prompt_lower = prompt.lower()
        idx = prompt_lower.find(phrase)
        if idx != -1:
            query = prompt[idx + len(phrase):].strip()
            # Clean up query
            query = query.replace("please", "").replace("for me", "").strip()
            if query and len(query) > 3:
                return query
        return ""

    def post_generate(self, response: str, context: Dict[str, Any] = None) -> str:
        """Add web search results to response if available."""
        if context is None:
            context = {}

        if "web_search_results" in context:
            results = context["web_search_results"]
            if results:
                response += "\n\n---"
                response += "\n🌐 **Web Search Results**"

                for i, result in enumerate(results, 1):
                    response += f"\n\n{i}. **{result.get('title', 'No title')}**"
                    if result.get('link'):
                        response += f"\n   Source: {result['link']}"
                    if result.get('snippet'):
                        response += f"\n   {result['snippet'][:200]}..."

        return response

    def get_actions(self) -> List[Dict[str, Any]]:
        """Get available actions."""
        return [
            {
                "name": "search",
                "display_name": "Web Search",
                "description": "Search the web for information",
                "icon": "🔍"
            },
            {
                "name": "configure",
                "display_name": "Configure Search",
                "description": "Configure search engine settings",
                "icon": "⚙️"
            }
        ]

    def execute_action(self, action: str, **kwargs) -> Any:
        """Execute a plugin action."""
        if action == "search":
            query = kwargs.get("query", "")
            return self.search(query)
        elif action == "configure":
            search_engine = kwargs.get("search_engine", "google")
            api_key = kwargs.get("api_key", "")
            max_results = kwargs.get("max_results", 3)

            self.search_engine = search_engine
            self.api_key = api_key
            self.max_results = max_results

            return {
                "status": "configured",
                "search_engine": search_engine,
                "max_results": max_results
            }
        return None

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "search_engine": {
                "type": "str",
                "default": "google",
                "options": ["google", "bing", "duckduckgo"],
                "description": "Search engine to use"
            },
            "api_key": {
                "type": "str",
                "default": "",
                "description": "API key for the search engine"
            },
            "max_results": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of search results"
            }
        }
