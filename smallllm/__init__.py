"""
SmallLLM - A tiny AI that runs on 2011-era PCs
"""

__version__ = "0.1.0"
__author__ = "ciao22king"

from .tokenizer import Tokenizer
from .model import SmallLLM
from .config import ModelConfig
from .cli import CLI

__all__ = ["Tokenizer", "SmallLLM", "ModelConfig", "CLI"]
