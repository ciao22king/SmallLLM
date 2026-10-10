"""
SmallLLM - A tiny AI that runs on 2011-era PCs
"""

__version__ = "0.1.0"
__author__ = "ciao22king"

from .tokenizer import Tokenizer
from .model import SmallLLM
from .config import ModelConfig, get_config_by_name, TINY_CONFIG, SMALL_CONFIG, MEDIUM_CONFIG
from .cli import CLI
from .api import app, run_api

__all__ = [
    "Tokenizer", 
    "SmallLLM", 
    "ModelConfig", 
    "CLI", 
    "app", 
    "run_api",
    "get_config_by_name",
    "TINY_CONFIG",
    "SMALL_CONFIG", 
    "MEDIUM_CONFIG"
]
