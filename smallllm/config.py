"""
Configuration for SmallLLM
Handles model hyperparameters and lazy loading settings
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib


@dataclass
class ModelConfig:
    """
    Configuration for SmallLLM model.
    All parameters are designed to work on low-memory systems.
    """
    
    # Model architecture
    vocab_size: int = 2048
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 128
    
    # Training/Inference
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 4
    
    # Memory optimization
    use_8bit: bool = True
    lazy_load: bool = True
    cache_dir: str = ".smallllm_cache"
    
    # Generation
    temperature: float = 0.7
    top_k: int = 50
    max_new_tokens: int = 50
    
    # Special tokens
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    
    # File paths (lazy-loaded)
    _model_path: Optional[str] = None
    _weights_loaded: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(exist_ok=True)
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_dim // self.num_heads
    
    @property
    def num_params(self) -> int:
        """Estimate total number of parameters."""
        # Embedding layer
        params = self.vocab_size * self.embedding_dim
        
        # Transformer layers
        per_layer = (
            # Self-attention
            (self.hidden_dim * self.hidden_dim * 3) +  # Q, K, V
            (self.hidden_dim * self.hidden_dim) +      # Output projection
            # Feed-forward
            (self.hidden_dim * self.hidden_dim * 4) +   # Expand and project
            # Layer norms
            self.hidden_dim * 2
        )
        
        params += per_layer * self.num_layers
        
        # Output layer
        params += self.hidden_dim * self.vocab_size
        
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'use_8bit': self.use_8bit,
            'lazy_load': self.lazy_load,
            'cache_dir': self.cache_dir,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'max_new_tokens': self.max_new_tokens,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'sos_token': self.sos_token,
            'eos_token': self.eos_token,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_weights_filename(self, layer_idx: int, param_name: str) -> str:
        """Generate filename for a specific weight tensor."""
        hash_input = f"{self.vocab_size}_{self.embedding_dim}_{self.hidden_dim}_{self.num_layers}_{self.num_heads}"
        model_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{model_hash}_L{layer_idx}_{param_name}.npy"
    
    def get_weights_path(self, layer_idx: int, param_name: str) -> Path:
        """Get full path for a weight file."""
        return Path(self.cache_dir) / self.get_weights_filename(layer_idx, param_name)
    
    def mark_weights_loaded(self, loaded: bool = True) -> None:
        """Mark whether weights have been loaded."""
        self._weights_loaded = loaded
    
    def are_weights_loaded(self) -> bool:
        """Check if weights have been loaded."""
        return self._weights_loaded


# Predefined configurations
TINY_CONFIG = ModelConfig(
    vocab_size=1024,
    embedding_dim=32,
    hidden_dim=64,
    num_layers=2,
    num_heads=2,
    max_seq_len=64,
    use_8bit=True,
    lazy_load=True,
)

SMALL_CONFIG = ModelConfig(
    vocab_size=2048,
    embedding_dim=64,
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    max_seq_len=128,
    use_8bit=True,
    lazy_load=True,
)

MEDIUM_CONFIG = ModelConfig(
    vocab_size=4096,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=4,
    num_heads=8,
    max_seq_len=256,
    use_8bit=True,
    lazy_load=True,
)


def get_config_by_name(name: str) -> ModelConfig:
    """Get a predefined configuration by name."""
    configs = {
        'tiny': TINY_CONFIG,
        'small': SMALL_CONFIG,
        'medium': MEDIUM_CONFIG,
    }
    if name in configs:
        return configs[name]
    raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")
