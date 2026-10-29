"""
SmallLLM Model Implementation
Tiny transformer-based language model optimized for old hardware
"""

import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from .config import ModelConfig
from .tokenizer import Tokenizer


class LayerNorm:
    """Simple Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self._initialized = False

    def init_params(self) -> None:
        """Initialize parameters (lazy)."""
        if self._initialized:
            return
        self.gamma = np.ones(self.dim, dtype=np.float32)
        self.beta = np.zeros(self.dim, dtype=np.float32)
        self._initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        self.init_params()

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Reshape gamma and beta to match x's dimensions
        # x shape: (batch, seq_len, dim) or (batch, dim)
        if x.ndim == 3:
            gamma = self.gamma.reshape(1, 1, -1)
            beta = self.beta.reshape(1, 1, -1)
        else:
            gamma = self.gamma.reshape(1, -1)
            beta = self.beta.reshape(1, -1)

        return gamma * x_norm + beta

    def load_params(self, gamma: np.ndarray, beta: np.ndarray) -> None:
        """Load pre-trained parameters."""
        self.gamma = gamma
        self.beta = beta
        self._initialized = True

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get parameters for saving."""
        self.init_params()
        return {'gamma': self.gamma, 'beta': self.beta}


class Linear:
    """Linear layer with lazy loading support."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.weight: Optional[np.ndarray] = None
        self.bias_vec: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self._initialized = False

    def init_params(self, scale: float = 0.02) -> None:
        """Initialize parameters with Xavier initialization."""
        if self._initialized:
            return

        # Xavier initialization
        limit = math.sqrt(6 / (self.in_dim + self.out_dim))
        self.weight = np.random.uniform(-limit, limit, (self.in_dim, self.out_dim)).astype(np.float32)

        if self.bias:
            self.bias_vec = np.zeros(self.out_dim, dtype=np.float32)

        self._initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.init_params()

        weight = self.dequantized_weight()

        out = np.dot(x, weight)

        if self.bias and self.bias_vec is not None:
            out += self.bias_vec

        return out

    def dequantized_weight(self) -> np.ndarray:
        """Return float32 weights, dequantizing 8-bit weights on the fly."""
        if self.weight.dtype == np.uint8:
            scale = self.scale if self.scale is not None else 1.0
            return (self.weight.astype(np.float32) - 128.0) * scale
        return self.weight

    def load_params(self, weight: np.ndarray, bias: Optional[np.ndarray] = None,
                    scale: Optional[np.ndarray] = None) -> None:
        """Load pre-trained parameters."""
        self.weight = weight
        self.bias_vec = bias
        self.scale = scale
        self._initialized = True

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get parameters for saving."""
        self.init_params()
        result = {'weight': self.weight}
        if self.bias and self.bias_vec is not None:
            result['bias'] = self.bias_vec
        if self.scale is not None:
            result['scale'] = self.scale
        return result

    def quantize(self) -> None:
        """Quantize weights to 8-bit with a per-tensor symmetric scale."""
        if self.weight is None or self.weight.dtype == np.uint8:
            return
        max_abs = float(np.max(np.abs(self.weight)))
        if max_abs == 0.0:
            self.scale = np.float32(1.0)
            self.weight = np.full(self.weight.shape, 128, dtype=np.uint8)
            return
        # 127 levels map to [-max_abs, +max_abs], stored as 0..255 with 128 = 0.
        self.scale = np.float32(max_abs / 127.0)
        quantized = np.round(self.weight / self.scale) + 128.0
        self.weight = np.clip(quantized, 0, 255).astype(np.uint8)


class MultiHeadAttention:
    """Multi-head self-attention with lazy loading."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        # Projection layers (lazy-loaded)
        self.q_proj: Optional[Linear] = None
        self.k_proj: Optional[Linear] = None
        self.v_proj: Optional[Linear] = None
        self.out_proj: Optional[Linear] = None

        self._initialized = False

    def init_params(self) -> None:
        """Initialize all parameters."""
        if self._initialized:
            return

        self.q_proj = Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = Linear(self.hidden_dim, self.hidden_dim)

        self._initialized = True

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split into multiple heads."""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Merge heads back."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, heads, head_dim)
        return x.reshape(batch_size, seq_len, self.hidden_dim)

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass for self-attention."""
        self.init_params()

        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = np.einsum('bhik,bhjk->bhij', q, k) / scale

        # Apply mask (for causal attention)
        if mask is None:
            # Create causal mask
            mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float32) * (-1e9)
            mask = mask[np.newaxis, np.newaxis, :, :]

        scores += mask

        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply attention to values
        output = np.einsum('bhij,bhjk->bhik', attn_weights, v)

        # Merge heads
        output = self._merge_heads(output)

        # Output projection
        output = self.out_proj(output)

        return output

    def load_params(
        self,
        q_weight: np.ndarray,
        k_weight: np.ndarray,
        v_weight: np.ndarray,
        out_weight: np.ndarray,
        q_bias: Optional[np.ndarray] = None,
        k_bias: Optional[np.ndarray] = None,
        v_bias: Optional[np.ndarray] = None,
        out_bias: Optional[np.ndarray] = None,
        scales: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Load pre-trained parameters."""
        scales = scales or {}
        self.q_proj = Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = Linear(self.hidden_dim, self.hidden_dim)

        self.q_proj.load_params(q_weight, q_bias, scales.get('q_proj'))
        self.k_proj.load_params(k_weight, k_bias, scales.get('k_proj'))
        self.v_proj.load_params(v_weight, v_bias, scales.get('v_proj'))
        self.out_proj.load_params(out_weight, out_bias, scales.get('out_proj'))

        self._initialized = True

    def quantize(self) -> None:
        """Quantize all projection weights to 8-bit."""
        self.init_params()
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.quantize()

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get all parameters."""
        self.init_params()
        return {
            'q_proj': self.q_proj.get_params(),
            'k_proj': self.k_proj.get_params(),
            'v_proj': self.v_proj.get_params(),
            'out_proj': self.out_proj.get_params(),
        }


class FeedForward:
    """Simple feed-forward network."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.hidden_dim = config.hidden_dim

        # Layers (lazy-loaded)
        self.fc1: Optional[Linear] = None
        self.fc2: Optional[Linear] = None

        self._initialized = False

    def init_params(self) -> None:
        """Initialize parameters."""
        if self._initialized:
            return

        # Expand to 4x hidden_dim for better expressiveness
        self.fc1 = Linear(self.hidden_dim, self.hidden_dim * 4)
        self.fc2 = Linear(self.hidden_dim * 4, self.hidden_dim)

        self._initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.init_params()

        # GELU activation (simplified)
        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

        h = gelu(self.fc1(x))
        return self.fc2(h)

    def load_params(
        self,
        fc1_weight: np.ndarray,
        fc1_bias: Optional[np.ndarray],
        fc2_weight: np.ndarray,
        fc2_bias: Optional[np.ndarray],
        fc1_scale: Optional[np.ndarray] = None,
        fc2_scale: Optional[np.ndarray] = None,
    ) -> None:
        """Load pre-trained parameters."""
        self.fc1 = Linear(self.hidden_dim, self.hidden_dim * 4)
        self.fc2 = Linear(self.hidden_dim * 4, self.hidden_dim)

        self.fc1.load_params(fc1_weight, fc1_bias, fc1_scale)
        self.fc2.load_params(fc2_weight, fc2_bias, fc2_scale)

        self._initialized = True

    def quantize(self) -> None:
        """Quantize both linear weights to 8-bit."""
        self.init_params()
        self.fc1.quantize()
        self.fc2.quantize()

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get all parameters."""
        self.init_params()
        return {
            'fc1': self.fc1.get_params(),
            'fc2': self.fc2.get_params(),
        }


class TransformerBlock:
    """Single transformer block with lazy loading."""

    def __init__(self, config: ModelConfig):
        self.config = config

        # Components (lazy-loaded)
        self.attn: Optional[MultiHeadAttention] = None
        self.ffn: Optional[FeedForward] = None
        self.norm1: Optional[LayerNorm] = None
        self.norm2: Optional[LayerNorm] = None

        self._initialized = False

    def init_params(self) -> None:
        """Initialize all parameters."""
        if self._initialized:
            return

        self.attn = MultiHeadAttention(self.config)
        self.ffn = FeedForward(self.config)
        self.norm1 = LayerNorm(self.config.hidden_dim)
        self.norm2 = LayerNorm(self.config.hidden_dim)

        self._initialized = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.init_params()

        # Self-attention with residual
        attn_out = self.attn(self.norm1(x))
        x = x + attn_out

        # Feed-forward with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x

    def load_params(self, params: Dict[str, Any]) -> None:
        """Load pre-trained parameters."""
        self.attn = MultiHeadAttention(self.config)
        self.ffn = FeedForward(self.config)
        self.norm1 = LayerNorm(self.config.hidden_dim)
        self.norm2 = LayerNorm(self.config.hidden_dim)

        attn_params = params['attn']
        self.attn.load_params(
            attn_params['q_proj']['weight'],
            attn_params['k_proj']['weight'],
            attn_params['v_proj']['weight'],
            attn_params['out_proj']['weight'],
            attn_params['q_proj'].get('bias'),
            attn_params['k_proj'].get('bias'),
            attn_params['v_proj'].get('bias'),
            attn_params['out_proj'].get('bias'),
            scales={name: attn_params[name].get('scale')
                    for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj')},
        )

        ffn_params = params['ffn']
        self.ffn.load_params(
            ffn_params['fc1']['weight'],
            ffn_params['fc1'].get('bias'),
            ffn_params['fc2']['weight'],
            ffn_params['fc2'].get('bias'),
            ffn_params['fc1'].get('scale'),
            ffn_params['fc2'].get('scale'),
        )

        self.norm1.load_params(
            params['norm1']['gamma'],
            params['norm1']['beta'],
        )
        self.norm2.load_params(
            params['norm2']['gamma'],
            params['norm2']['beta'],
        )

        self._initialized = True

    def quantize(self) -> None:
        """Quantize the attention and feed-forward weights to 8-bit."""
        self.init_params()
        self.attn.quantize()
        self.ffn.quantize()

    def get_params(self) -> Dict[str, Any]:
        """Get all parameters."""
        self.init_params()
        return {
            'attn': self.attn.get_params(),
            'ffn': self.ffn.get_params(),
            'norm1': self.norm1.get_params(),
            'norm2': self.norm2.get_params(),
        }


class SmallLLM:
    """
    Main SmallLLM model class.
    Implements a tiny transformer-based language model
    that can run on 2011-era hardware.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Initialize SmallLLM model.

        Args:
            config: Model configuration (uses SMALL_CONFIG by default)
            tokenizer: Tokenizer instance (created if None)
        """
        self.config = config or ModelConfig()
        self.tokenizer = tokenizer or Tokenizer(vocab_size=self.config.vocab_size)

        # Model components (lazy-loaded)
        self.embedding: Optional[np.ndarray] = None
        self.position_embedding: Optional[np.ndarray] = None
        self.embedding_proj: Optional[Linear] = None
        self.layers: List[Optional[TransformerBlock]] = [None] * self.config.num_layers
        self.final_norm: Optional[LayerNorm] = None
        self.lm_head: Optional[Linear] = None

        # Quantization scales for the raw embedding arrays
        self.embedding_scale: Optional[np.ndarray] = None
        self.position_embedding_scale: Optional[np.ndarray] = None

        self._initialized = False
        self._weights_loaded = False

    def init_params(self) -> None:
        """Initialize all model parameters."""
        if self._initialized:
            return

        # Embedding layer
        self.embedding = np.random.normal(
            0, 0.02,
            (self.config.vocab_size, self.config.embedding_dim)
        ).astype(np.float32)

        # Position embedding
        self.position_embedding = np.zeros(
            (self.config.max_seq_len, self.config.embedding_dim)
        ).astype(np.float32)

        # Initialize position embeddings with sinusoidal pattern
        position = np.arange(self.config.max_seq_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.config.embedding_dim, 2)
            * -(math.log(10000.0) / self.config.embedding_dim))

        self.position_embedding[:, 0::2] = np.sin(position * div_term)
        self.position_embedding[:, 1::2] = np.cos(position * div_term)

        # Projection from embedding_dim to hidden_dim (if different)
        self.embedding_proj = None
        if self.config.embedding_dim != self.config.hidden_dim:
            self.embedding_proj = Linear(self.config.embedding_dim, self.config.hidden_dim)

        # Transformer layers
        for i in range(self.config.num_layers):
            self.layers[i] = TransformerBlock(self.config)

        # Final layer norm
        self.final_norm = LayerNorm(self.config.hidden_dim)

        # LM head (from hidden_dim to vocab_size)
        self.lm_head = Linear(self.config.hidden_dim, self.config.vocab_size)

        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure model is initialized."""
        if not self._initialized:
            self.init_params()

    @staticmethod
    def _dequantize(array: np.ndarray, scale: Optional[np.ndarray]) -> np.ndarray:
        """Dequantize a uint8 array that was produced by _quantize_array."""
        if array is not None and array.dtype == np.uint8:
            effective_scale = scale if scale is not None else np.float32(1.0)
            return (array.astype(np.float32) - 128.0) * effective_scale
        return array

    @staticmethod
    def _quantize_array(array: np.ndarray) -> Tuple[np.ndarray, np.float32]:
        """Quantize a float array to uint8 with a per-tensor symmetric scale."""
        if array is None or array.dtype == np.uint8:
            return array, np.float32(1.0)
        max_abs = float(np.max(np.abs(array)))
        if max_abs == 0.0:
            return np.full(array.shape, 128, dtype=np.uint8), np.float32(1.0)
        scale = np.float32(max_abs / 127.0)
        quantized = np.clip(np.round(array / scale) + 128.0, 0, 255).astype(np.uint8)
        return quantized, scale

    def _load_layer_weights(self, layer_idx: int) -> None:
        """Load weights for a specific layer on demand."""
        if self.layers[layer_idx] is None:
            self.layers[layer_idx] = TransformerBlock(self.config)

        if self.layers[layer_idx]._initialized:
            return

        # Try to load from cache
        cache_path = self.config.get_weights_path(layer_idx, 'block')
        if cache_path.exists():
            try:
                params = np.load(cache_path, allow_pickle=True).item()
                self.layers[layer_idx].load_params(params)
            except Exception:
                # If loading fails, initialize fresh
                self.layers[layer_idx].init_params()

    def forward(
        self,
        input_ids: np.ndarray,
        load_layers_on_demand: bool = True,
    ) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            load_layers_on_demand: Whether to load layers as needed

        Returns:
            Logits for next token prediction (batch_size, seq_len, vocab_size)
        """
        self._ensure_initialized()

        batch_size, seq_len = input_ids.shape

        # Ensure sequence length is within limits
        if seq_len > self.config.max_seq_len:
            input_ids = input_ids[:, :self.config.max_seq_len]
            seq_len = self.config.max_seq_len

        # Embedding lookup
        x = self._dequantize(self.embedding, self.embedding_scale)[input_ids]

        # Add position embeddings
        positions = np.arange(seq_len)[np.newaxis, :]
        pos_emb = self._dequantize(self.position_embedding, self.position_embedding_scale)
        x += pos_emb[positions]

        # Project to hidden_dim if needed
        if self.embedding_proj is not None:
            # Reshape for Linear layer: (batch, seq_len, embedding_dim) -> (batch * seq_len, embedding_dim)
            batch_size = x.shape[0]
            x_reshaped = x.reshape(-1, self.config.embedding_dim)
            x_projected = self.embedding_proj(x_reshaped)
            x = x_projected.reshape(batch_size, seq_len, self.config.hidden_dim)

        # Pass through transformer layers
        for i in range(self.config.num_layers):
            if load_layers_on_demand and not self._weights_loaded:
                self._load_layer_weights(i)

            if self.layers[i] is not None:
                x = self.layers[i](x)

        # Final layer norm
        if self.final_norm is not None:
            x = self.final_norm(x)

        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # Fallback if lm_head not initialized
            self.lm_head = Linear(self.config.hidden_dim, self.config.vocab_size)
            logits = self.lm_head(x)

        return logits

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated text
        """
        self._ensure_initialized()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            # An empty prompt would produce a zero-length sequence that cannot be
            # indexed; seed with the start token so generation still works.
            input_ids = [self.tokenizer.vocab.get(
                self.config.sos_token, self.tokenizer.vocab.get('[SOS]', 0))]
        input_ids = np.array([input_ids], dtype=np.int64)

        generated = []

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(input_ids, load_layers_on_demand=True)

            # Get last token's logits
            last_logits = logits[0, -1, :]

            # Apply temperature
            if temperature > 0:
                last_logits = last_logits / temperature

            # Softmax
            exp_logits = np.exp(last_logits - np.max(last_logits))
            probs = exp_logits / np.sum(exp_logits)

            # Sampling
            if do_sample:
                # Top-k sampling
                k = max(1, min(top_k, probs.shape[0]))
                top_k_indices = np.argsort(probs)[-k:]
                top_k_probs = probs[top_k_indices]
                top_k_probs = top_k_probs / np.sum(top_k_probs)

                next_token = np.random.choice(top_k_indices, p=top_k_probs)
            else:
                # Greedy decoding
                next_token = np.argmax(probs)

            # Append to input
            generated.append(next_token)
            input_ids = np.append(input_ids[0], [next_token])[np.newaxis, :]

            # Stop if EOS token
            eos_id = self.tokenizer.vocab.get(
                self.config.eos_token, self.tokenizer.vocab.get('[EOS]', -1))
            if next_token == eos_id:
                break

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated)

        return generated_text

    def chat(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> str:
        """
        Simple chat interface.

        Args:
            prompt: User input
            max_new_tokens: Maximum response length
            temperature: Sampling temperature

        Returns:
            Model response
        """
        # Add some context for better responses
        context = "Human: {prompt}\nAI:"
        return self.generate(context.format(prompt=prompt), max_new_tokens, temperature)

    def save_weights(self, path: str) -> None:
        """Save model weights to directory."""
        self._ensure_initialized()

        save_path = Path(path)
        save_path.mkdir(exist_ok=True)

        # Save config
        config_path = save_path / "config.json"
        self.config.save(str(config_path))

        # Save tokenizer
        tokenizer_path = save_path / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))

        # Save embedding
        np.save(save_path / "embedding.npy", self.embedding)
        np.save(save_path / "position_embedding.npy", self.position_embedding)

        # Save the quantization scales so quantized weights round-trip
        if self.embedding_scale is not None:
            np.save(save_path / "embedding_scale.npy", self.embedding_scale)
        if self.position_embedding_scale is not None:
            np.save(save_path / "position_embedding_scale.npy", self.position_embedding_scale)

        # Save the embedding projection when embedding_dim != hidden_dim
        if self.embedding_proj is not None:
            np.save(save_path / "embedding_proj.npy", self.embedding_proj.get_params())

        # Save layers
        for i, layer in enumerate(self.layers):
            if layer is not None:
                params = layer.get_params()
                layer_path = save_path / f"layer_{i}.npy"
                np.save(layer_path, params)

        # Save final norm
        if self.final_norm is not None:
            norm_path = save_path / "final_norm.npy"
            np.save(norm_path, self.final_norm.get_params())

        # Save LM head
        if self.lm_head is not None:
            head_path = save_path / "lm_head.npy"
            np.save(head_path, self.lm_head.get_params())

    @staticmethod
    def _load_npy_dict(path: Path) -> Optional[Dict[str, Any]]:
        """Load a saved parameter dict, or None if missing/corrupt."""
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=True).item()
        except Exception:
            return None

    @classmethod
    def load_weights(cls, path: str) -> 'SmallLLM':
        """Load model weights from directory."""
        load_path = Path(path)

        # Load config and tokenizer
        config = ModelConfig.load(str(load_path / "config.json"))
        tokenizer = Tokenizer.load(str(load_path / "tokenizer.json"))

        # Create model
        model = cls(config, tokenizer)
        model._initialized = True

        model._load_embeddings(load_path)
        model._load_projection(load_path, config)
        model._load_layers(load_path, config)
        model._load_final_norm(load_path, config)
        model._load_lm_head(load_path, config)

        model._weights_loaded = True
        model.config.mark_weights_loaded(True)

        return model

    def _load_embeddings(self, load_path: Path) -> None:
        """Load embedding and position-embedding arrays and their scales."""
        if (load_path / "embedding.npy").exists():
            self.embedding = np.load(load_path / "embedding.npy")
        if (load_path / "position_embedding.npy").exists():
            self.position_embedding = np.load(load_path / "position_embedding.npy")
        if (load_path / "embedding_scale.npy").exists():
            self.embedding_scale = np.load(load_path / "embedding_scale.npy")
        if (load_path / "position_embedding_scale.npy").exists():
            self.position_embedding_scale = np.load(load_path / "position_embedding_scale.npy")

    def _load_projection(self, load_path: Path, config: ModelConfig) -> None:
        """Rebuild the embedding projection when embedding_dim != hidden_dim."""
        if config.embedding_dim == config.hidden_dim:
            return
        self.embedding_proj = Linear(config.embedding_dim, config.hidden_dim)
        params = self._load_npy_dict(load_path / "embedding_proj.npy")
        if params is not None:
            try:
                self.embedding_proj.load_params(
                    params['weight'], params.get('bias'), params.get('scale'))
                return
            except Exception:
                pass
        self.embedding_proj.init_params()

    def _load_layers(self, load_path: Path, config: ModelConfig) -> None:
        """Load every transformer layer, falling back to fresh weights."""
        for i in range(config.num_layers):
            self.layers[i] = TransformerBlock(config)
            params = self._load_npy_dict(load_path / f"layer_{i}.npy")
            if params is not None:
                try:
                    self.layers[i].load_params(params)
                    continue
                except Exception:
                    pass
            self.layers[i].init_params()

    def _load_final_norm(self, load_path: Path, config: ModelConfig) -> None:
        """Load the final layer norm."""
        self.final_norm = LayerNorm(config.hidden_dim)
        params = self._load_npy_dict(load_path / "final_norm.npy")
        if params is not None:
            try:
                self.final_norm.load_params(params['gamma'], params['beta'])
                return
            except Exception:
                pass
        self.final_norm.init_params()

    def _load_lm_head(self, load_path: Path, config: ModelConfig) -> None:
        """Load the language-model head."""
        self.lm_head = Linear(config.hidden_dim, config.vocab_size)
        params = self._load_npy_dict(load_path / "lm_head.npy")
        if params is not None:
            try:
                self.lm_head.load_params(
                    params['weight'], params.get('bias'), params.get('scale'))
                return
            except Exception:
                pass
        self.lm_head.init_params()

    def quantize(self) -> None:
        """Quantize model weights to 8-bit."""
        self._ensure_initialized()

        if self.embedding is not None:
            self.embedding, self.embedding_scale = self._quantize_array(self.embedding)

        if self.position_embedding is not None:
            self.position_embedding, self.position_embedding_scale = \
                self._quantize_array(self.position_embedding)

        if self.embedding_proj is not None:
            self.embedding_proj.quantize()

        for layer in self.layers:
            if layer is not None:
                layer.quantize()

        if self.final_norm is not None:
            self.final_norm.init_params()

        if self.lm_head is not None:
            self.lm_head.quantize()

        self.config.use_8bit = True

    def get_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage in bytes."""
        memory = {}

        if self.embedding is not None:
            memory['embedding'] = self.embedding.nbytes

        if self.position_embedding is not None:
            memory['position_embedding'] = self.position_embedding.nbytes

        for i, layer in enumerate(self.layers):
            if layer is not None and layer._initialized:
                layer_memory = 0
                for param_dict in layer.get_params().values():
                    for value in param_dict.values():
                        if isinstance(value, np.ndarray):
                            layer_memory += value.nbytes
                memory[f'layer_{i}'] = layer_memory

        if self.final_norm is not None and self.final_norm._initialized:
            norm_memory = sum(v.nbytes for v in self.final_norm.get_params().values())
            memory['final_norm'] = norm_memory

        if self.lm_head is not None and self.lm_head._initialized:
            head_memory = sum(v.nbytes for v in self.lm_head.get_params().values())
            memory['lm_head'] = head_memory

        memory['total'] = sum(memory.values())

        return memory

    def train_step(
        self,
        input_ids: np.ndarray,
        target_ids: np.ndarray,
        learning_rate: float = 1e-4,
    ) -> float:
        """
        Simple training step (for demonstration).
        Not optimized for actual training on old hardware.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            target_ids: Target token IDs (batch_size, seq_len)
            learning_rate: Learning rate

        Returns:
            Loss value
        """
        self._ensure_initialized()

        # Forward pass
        logits = self.forward(input_ids)

        # Compute loss (cross-entropy)
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for softmax
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        # Softmax
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Gather probabilities for target tokens
        target_probs = probs[np.arange(len(targets_flat)), targets_flat]

        # Cross-entropy loss
        loss = -np.mean(np.log(target_probs + 1e-10))

        # Simple SGD update (very basic, not practical for real training)
        # In a real implementation, you'd want proper backpropagation
        # This is just for demonstration

        return float(loss)
