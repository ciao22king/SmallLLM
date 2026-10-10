"""
Translator Plugin for SmallLLM
Translate text between different languages
"""

import re
from typing import Dict, Any, List, Optional
from .base import BasePlugin


class TranslatorPlugin(BasePlugin):
    """Plugin for translating text between languages."""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.display_name = "Translator"
        self.description = "Translate text between different languages"
        self.icon = "🌍"
        self.version = "1.0"
        self.author = "SmallLLM"
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'it': 'Italian',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic',
        }
        
        # Simple translation dictionary (for demo purposes)
        # In a real implementation, you'd use a proper translation API
        self.translation_dict = self._build_translation_dict()
    
    def _build_translation_dict(self) -> Dict[str, Dict[str, str]]:
        """Build a simple translation dictionary for demo purposes."""
        translations = {}
        
        # English to Italian
        translations['en'] = {
            'it': {
                'hello': 'ciao',
                'goodbye': 'arrivederci',
                'thank you': 'grazie',
                'please': 'per favore',
                'yes': 'sì',
                'no': 'no',
                'good': 'buono',
                'bad': 'cattivo',
                'big': 'grande',
                'small': 'piccolo',
                'how are you': 'come stai',
                'what is your name': 'come ti chiami',
                'my name is': 'mi chiamo',
                'today': 'oggi',
                'tomorrow': 'domani',
                'yesterday': 'ieri',
            }
        }
        
        # Italian to English
        translations['it'] = {
            'en': {
                'ciao': 'hello',
                'arrivederci': 'goodbye',
                'grazie': 'thank you',
                'per favore': 'please',
                'sì': 'yes',
                'no': 'no',
                'buono': 'good',
                'cattivo': 'bad',
                'grande': 'big',
                'piccolo': 'small',
                'come stai': 'how are you',
                'come ti chiami': 'what is your name',
                'mi chiamo': 'my name is',
                'oggi': 'today',
                'domani': 'tomorrow',
                'ieri': 'yesterday',
            }
        }
        
        # Spanish to English
        translations['es'] = {
            'en': {
                'hola': 'hello',
                'adiós': 'goodbye',
                'gracias': 'thank you',
                'por favor': 'please',
                'sí': 'yes',
                'no': 'no',
                'bueno': 'good',
                'malo': 'bad',
                'grande': 'big',
                'pequeño': 'small',
            }
        }
        
        # English to Spanish
        translations['en']['es'] = {v: k for k, v in translations['es']['en'].items()}
        
        # French to English
        translations['fr'] = {
            'en': {
                'bonjour': 'hello',
                'au revoir': 'goodbye',
                'merci': 'thank you',
                's\'il vous plaît': 'please',
                'oui': 'yes',
                'non': 'no',
            }
        }
        
        # English to French
        translations['en']['fr'] = {v: k for k, v in translations['fr']['en'].items()}
        
        return translations
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source language to target language."""
        if source_lang == target_lang:
            return text
        
        # Check if languages are supported
        if source_lang not in self.translation_dict:
            return text
        if target_lang not in self.translation_dict[source_lang]:
            # Try reverse translation
            if target_lang in self.translation_dict and source_lang in self.translation_dict[target_lang]:
                # Use reverse dictionary
                reverse_trans = self.translation_dict[target_lang][source_lang]
                return self._translate_with_dict(text, reverse_trans)
            return text
        
        # Get translation dictionary
        trans_dict = self.translation_dict[source_lang][target_lang]
        return self._translate_with_dict(text, trans_dict)
    
    def _translate_with_dict(self, text: str, trans_dict: Dict[str, str]) -> str:
        """Translate text using a dictionary."""
        # Simple word-by-word translation
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            # Clean word
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            # Try to find translation
            if clean_word in trans_dict:
                translated_words.append(trans_dict[clean_word])
            else:
                # Keep original word
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text (simple implementation)."""
        # Count words that match each language
        text_lower = text.lower()
        
        best_lang = 'en'  # Default to English
        best_count = 0
        
        for lang, lang_dict in self.translation_dict.items():
            if 'en' in lang_dict:
                count = sum(1 for word in text_lower.split() 
                            if word in lang_dict['en'])
                if count > best_count:
                    best_count = count
                    best_lang = lang
        
        return best_lang
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages."""
        return [
            {"code": code, "name": name}
            for code, name in self.supported_languages.items()
        ]
    
    def pre_generate(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Detect translation requests in prompt."""
        if context is None:
            context = {}
        
        # Check for translation requests
        translation_patterns = [
            r'translate\s+(?:to\s+)?(\w+)\s*:',
            r'tradurre\s+(?:in\s+)?(\w+)\s*:',
            r'traducir\s+(?:a\s+)?(\w+)\s*:',
            r'traduire\s+(?:en\s+)?(\w+)\s*:',
        ]
        
        for pattern in translation_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                target_lang = match.group(1)
                context["translate_to"] = target_lang
                # Add hint to model
                return f"{prompt}\n\n[Translation mode: translate to {target_lang}]"
        
        return prompt
    
    def post_generate(self, response: str, context: Dict[str, Any] = None) -> str:
        """Translate response if translation was requested."""
        if context is None:
            context = {}
        
        if "translate_to" in context:
            target_lang = context["translate_to"]
            source_lang = self.detect_language(response)
            
            if source_lang != target_lang:
                translated = self.translate(response, source_lang, target_lang)
                if translated != response:
                    return f"{translated}\n\n[Translated from {source_lang} to {target_lang}]"
        
        return response
    
    def get_actions(self) -> List[Dict[str, Any]]:
        """Get available actions."""
        return [
            {
                "name": "translate",
                "display_name": "Translate Text",
                "description": "Translate text between languages",
                "icon": "🌍"
            },
            {
                "name": "detect_language",
                "display_name": "Detect Language",
                "description": "Detect the language of text",
                "icon": "💬"
            },
            {
                "name": "list_languages",
                "display_name": "List Languages",
                "description": "List supported languages",
                "icon": "📍"
            }
        ]
    
    def execute_action(self, action: str, **kwargs) -> Any:
        """Execute a plugin action."""
        if action == "translate":
            text = kwargs.get("text", "")
            source_lang = kwargs.get("source_lang", "en")
            target_lang = kwargs.get("target_lang", "it")
            
            if not text:
                return {"error": "No text provided", "success": False}
            
            # Auto-detect source language if not specified
            if source_lang == "auto":
                source_lang = self.detect_language(text)
            
            translated = self.translate(text, source_lang, target_lang)
            
            return {
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "translated": translated,
                "success": True
            }
        elif action == "detect_language":
            text = kwargs.get("text", "")
            if not text:
                return {"error": "No text provided", "success": False}
            
            lang = self.detect_language(text)
            return {
                "text": text,
                "language": lang,
                "language_name": self.supported_languages.get(lang, "Unknown"),
                "success": True
            }
        elif action == "list_languages":
            return {
                "languages": self.get_supported_languages(),
                "success": True
            }
        
        return None
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "default_source": {
                "type": "str",
                "default": "en",
                "description": "Default source language"
            },
            "default_target": {
                "type": "str",
                "default": "it",
                "description": "Default target language"
            },
            "auto_detect": {
                "type": "bool",
                "default": True,
                "description": "Automatically detect source language"
            }
        }
