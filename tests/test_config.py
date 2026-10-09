"""
Tests for ModelConfig
"""

import unittest
import tempfile
import os
from smallllm.config import ModelConfig, get_config_by_name, TINY_CONFIG, SMALL_CONFIG


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        self.assertEqual(config.vocab_size, 2048)
        self.assertEqual(config.embedding_dim, 64)
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.max_seq_len, 128)
    
    def test_head_dim(self):
        """Test head dimension calculation."""
        config = ModelConfig(hidden_dim=128, num_heads=4)
        self.assertEqual(config.head_dim, 32)
        
        config = ModelConfig(hidden_dim=256, num_heads=8)
        self.assertEqual(config.head_dim, 32)
    
    def test_num_params(self):
        """Test parameter count estimation."""
        config = ModelConfig(
            vocab_size=1024,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
        )
        
        params = config.num_params
        self.assertGreater(params, 0)
        self.assertIsInstance(params, int)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ModelConfig()
        data = config.to_dict()
        
        self.assertIn('vocab_size', data)
        self.assertIn('embedding_dim', data)
        self.assertIn('hidden_dim', data)
        self.assertEqual(data['vocab_size'], 2048)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'vocab_size': 1024,
            'embedding_dim': 32,
            'hidden_dim': 64,
            'num_layers': 2,
            'num_heads': 2,
        }
        
        config = ModelConfig.from_dict(data)
        
        self.assertEqual(config.vocab_size, 1024)
        self.assertEqual(config.embedding_dim, 32)
        self.assertEqual(config.hidden_dim, 64)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.num_heads, 2)
    
    def test_save_load(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            
            # Create and save
            config = ModelConfig(vocab_size=1024, embedding_dim=32)
            config.save(path)
            self.assertTrue(os.path.exists(path))
            
            # Load
            loaded_config = ModelConfig.load(path)
            
            self.assertEqual(loaded_config.vocab_size, 1024)
            self.assertEqual(loaded_config.embedding_dim, 32)
    
    def test_get_config_by_name(self):
        """Test getting predefined configurations."""
        # Test known configs
        tiny = get_config_by_name('tiny')
        self.assertEqual(tiny.vocab_size, 1024)
        self.assertEqual(tiny.embedding_dim, 32)
        
        small = get_config_by_name('small')
        self.assertEqual(small.vocab_size, 2048)
        self.assertEqual(small.embedding_dim, 64)
        
        medium = get_config_by_name('medium')
        self.assertEqual(medium.vocab_size, 4096)
        self.assertEqual(medium.embedding_dim, 128)
    
    def test_invalid_config_name(self):
        """Test handling of invalid configuration name."""
        with self.assertRaises(ValueError):
            get_config_by_name('invalid')
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ModelConfig(hidden_dim=64, num_heads=4)
        self.assertEqual(config.num_heads, 4)
        
        # Invalid: hidden_dim not divisible by num_heads
        with self.assertRaises(ValueError):
            ModelConfig(hidden_dim=64, num_heads=5)
        
        # Invalid: zero embedding_dim
        with self.assertRaises(ValueError):
            ModelConfig(embedding_dim=0)
        
        # Invalid: negative num_layers
        with self.assertRaises(ValueError):
            ModelConfig(num_layers=-1)
    
    def test_weights_filename(self):
        """Test weight filename generation."""
        config = ModelConfig()
        
        filename = config.get_weights_filename(0, 'embedding')
        self.assertIn('embedding', filename)
        self.assertIn('L0', filename)
        
        filename2 = config.get_weights_filename(1, 'attention')
        self.assertIn('attention', filename2)
        self.assertIn('L1', filename2)


if __name__ == '__main__':
    unittest.main()
