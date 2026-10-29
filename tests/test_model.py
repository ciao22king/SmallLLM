"""
Tests for SmallLLM Model
"""

import unittest
import tempfile
import os
import numpy as np
from smallllm.model import SmallLLM, Linear, LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock
from smallllm.config import ModelConfig


class TestLinear(unittest.TestCase):
    """Test cases for Linear layer."""

    def test_init(self):
        """Test initialization."""
        layer = Linear(10, 20)
        self.assertEqual(layer.in_dim, 10)
        self.assertEqual(layer.out_dim, 20)
        self.assertFalse(layer._initialized)

    def test_forward(self):
        """Test forward pass."""
        layer = Linear(3, 2)
        x = np.array([[1.0, 2.0, 3.0]])

        # Initialize and forward
        layer.init_params()
        output = layer(x)

        self.assertEqual(output.shape, (1, 2))

    def test_lazy_init(self):
        """Test lazy initialization."""
        layer = Linear(3, 2)
        self.assertFalse(layer._initialized)

        x = np.array([[1.0, 2.0, 3.0]])
        output = layer(x)

        self.assertTrue(layer._initialized)
        self.assertEqual(output.shape, (1, 2))


class TestLayerNorm(unittest.TestCase):
    """Test cases for LayerNorm."""

    def test_init(self):
        """Test initialization."""
        norm = LayerNorm(10)
        self.assertEqual(norm.dim, 10)
        self.assertFalse(norm._initialized)

    def test_forward(self):
        """Test forward pass."""
        norm = LayerNorm(3)
        x = np.array([[1.0, 2.0, 3.0]])

        output = norm(x)

        self.assertEqual(output.shape, (1, 3))
        self.assertTrue(norm._initialized)


class TestMultiHeadAttention(unittest.TestCase):
    """Test cases for MultiHeadAttention."""

    def setUp(self):
        """Set up test fixtures."""
        config = ModelConfig(
            vocab_size=100,
            embedding_dim=16,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
        )
        self.attn = MultiHeadAttention(config)

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.attn.num_heads, 2)
        self.assertEqual(self.attn.head_dim, 8)
        self.assertFalse(self.attn._initialized)

    def test_forward(self):
        """Test forward pass."""
        x = np.random.randn(1, 10, 16).astype(np.float32)

        output = self.attn(x)

        self.assertEqual(output.shape, (1, 10, 16))
        self.assertTrue(self.attn._initialized)


class TestFeedForward(unittest.TestCase):
    """Test cases for FeedForward."""

    def setUp(self):
        """Set up test fixtures."""
        config = ModelConfig(
            vocab_size=100,
            embedding_dim=16,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
        )
        self.ffn = FeedForward(config)

    def test_forward(self):
        """Test forward pass."""
        x = np.random.randn(1, 10, 16).astype(np.float32)

        output = self.ffn(x)

        self.assertEqual(output.shape, (1, 10, 16))


class TestTransformerBlock(unittest.TestCase):
    """Test cases for TransformerBlock."""

    def setUp(self):
        """Set up test fixtures."""
        config = ModelConfig(
            vocab_size=100,
            embedding_dim=16,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
        )
        self.block = TransformerBlock(config)

    def test_forward(self):
        """Test forward pass."""
        x = np.random.randn(1, 10, 16).astype(np.float32)

        output = self.block(x)

        self.assertEqual(output.shape, (1, 10, 16))


class TestSmallLLM(unittest.TestCase):
    """Test cases for SmallLLM model."""

    def setUp(self):
        """Set up test fixtures."""
        config = ModelConfig(
            vocab_size=100,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            max_seq_len=32,
        )
        self.model = SmallLLM(config)

    def test_init(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.config)
        self.assertIsNotNone(self.model.tokenizer)
        self.assertFalse(self.model._initialized)

    def test_forward(self):
        """Test forward pass."""
        # Initialize model
        self.model._ensure_initialized()

        # Create input
        input_ids = np.array([[1, 2, 3, 4]])

        # Forward pass
        output = self.model.forward(input_ids)

        self.assertEqual(output.shape, (1, 4, 100))

    def test_generate(self):
        """Test text generation."""
        prompt = "hello world"

        result = self.model.generate(prompt, max_new_tokens=10)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_chat(self):
        """Test chat interface."""
        prompt = "hello"

        response = self.model.chat(prompt, max_new_tokens=10)

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_save_load(self):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")

            # Save
            self.model._ensure_initialized()
            self.model.save_weights(path)
            self.assertTrue(os.path.exists(path))

            # Load
            loaded_model = SmallLLM.load_weights(path)

            self.assertIsNotNone(loaded_model)
            self.assertEqual(loaded_model.config.vocab_size, 100)

    def test_get_memory_usage(self):
        """Test memory usage estimation."""
        self.model._ensure_initialized()

        memory = self.model.get_memory_usage()

        self.assertIn('total', memory)
        self.assertGreater(memory['total'], 0)

    def test_lazy_loading(self):
        """Test that parameters are loaded lazily."""
        # Model should not be initialized yet
        self.assertFalse(self.model._initialized)

        # Generate should trigger initialization
        self.model.generate("test", max_new_tokens=5)

        # Now it should be initialized
        self.assertTrue(self.model._initialized)


class TestSmallLLMWithDefaultConfig(unittest.TestCase):
    """Test SmallLLM with default configuration."""

    def test_default_model(self):
        """Test model with default configuration."""
        model = SmallLLM()

        self.assertEqual(model.config.vocab_size, 2048)
        self.assertEqual(model.config.embedding_dim, 64)
        self.assertEqual(model.config.hidden_dim, 128)
        self.assertEqual(model.config.num_layers, 2)

    def test_generation_with_default(self):
        """Test generation with default model."""
        model = SmallLLM()

        result = model.generate("The quick brown fox", max_new_tokens=10)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_generate_empty_prompt(self):
        """An empty prompt must not crash generation."""
        model = SmallLLM()
        result = model.generate("", max_new_tokens=3)
        self.assertIsInstance(result, str)

    def test_generate_extreme_sampling_params(self):
        """top_k of 0 or a negative value must be handled, not crash."""
        model = SmallLLM()
        for top_k in (0, -5, 1):
            result = model.generate("hello", max_new_tokens=2, top_k=top_k)
            self.assertIsInstance(result, str)

    def test_quantized_save_load_roundtrip(self):
        """Quantized weights must survive a save/load cycle and still run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")

            model = SmallLLM()
            model.init_params()
            model.quantize()
            model.save_weights(path)

            loaded = SmallLLM.load_weights(path)
            self.assertIsNotNone(loaded.embedding_scale)

            out = loaded.generate("hello", max_new_tokens=3)
            self.assertIsInstance(out, str)

    def test_projection_save_load_roundtrip(self):
        """A model with embedding_dim != hidden_dim round-trips its projection."""
        config = ModelConfig(
            vocab_size=100,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            max_seq_len=32,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")

            model = SmallLLM(config)
            model.init_params()
            self.assertIsNotNone(model.embedding_proj)
            model.save_weights(path)

            loaded = SmallLLM.load_weights(path)
            self.assertIsNotNone(loaded.embedding_proj)
            out = loaded.generate("hi", max_new_tokens=2)
            self.assertIsInstance(out, str)


if __name__ == '__main__':
    unittest.main()
