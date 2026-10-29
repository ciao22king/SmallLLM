"""
Tests for Tokenizer
"""

import unittest
import tempfile
import os
from smallllm.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    """Test cases for Tokenizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = Tokenizer(vocab_size=2048)

    def test_encode_decode_simple(self):
        """Test encoding and decoding simple text."""
        text = "hello world"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        # Should be able to encode and decode
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(encoded), 0)

    def test_encode_decode_roundtrip(self):
        """Test that encode then decode returns similar text."""
        text = "The quick brown fox jumps over the lazy dog."
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        # For a simple tokenizer, we expect the decoded text to contain the original
        self.assertIn("the", decoded.lower())
        self.assertIn("quick", decoded.lower())
        self.assertIn("fox", decoded.lower())

    def test_vocab_size(self):
        """Test vocabulary size."""
        vocab_size = self.tokenizer.get_vocab_size()
        self.assertGreater(vocab_size, 0)
        self.assertLessEqual(vocab_size, 2048)

    def test_special_tokens(self):
        """Test that special tokens are in vocabulary."""
        vocab = self.tokenizer.vocab

        self.assertIn('<PAD>', vocab)
        self.assertIn('<UNK>', vocab)
        self.assertIn('<SOS>', vocab)
        self.assertIn('<EOS>', vocab)

    def test_unknown_token(self):
        """Test handling of unknown tokens."""
        # Create a tokenizer with very small vocab
        small_tokenizer = Tokenizer(vocab_size=100)

        # Try to encode text with words likely not in small vocab
        text = "supercalifragilisticexpialidocious"
        encoded = small_tokenizer.encode(text)

        # Should still produce some tokens (likely character-level)
        self.assertGreater(len(encoded), 0)

    def test_batch_encode_decode(self):
        """Test batch encoding and decoding."""
        texts = ["hello", "world", "test"]
        encoded = self.tokenizer.encode_batch(texts)
        decoded = self.tokenizer.decode_batch(encoded)

        self.assertEqual(len(encoded), len(texts))
        self.assertEqual(len(decoded), len(texts))

    def test_save_load(self):
        """Test saving and loading tokenizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tokenizer.json")

            # Save
            self.tokenizer.save(path)
            self.assertTrue(os.path.exists(path))

            # Load
            loaded_tokenizer = Tokenizer.load(path)

            # Test that loaded tokenizer works
            text = "test text"
            encoded_orig = self.tokenizer.encode(text)
            encoded_loaded = loaded_tokenizer.encode(text)

            # Should produce same encoding
            self.assertEqual(encoded_orig, encoded_loaded)

    def test_case_insensitive(self):
        """Test that tokenizer is case-insensitive."""
        text1 = "Hello World"
        text2 = "hello world"

        encoded1 = self.tokenizer.encode(text1)
        encoded2 = self.tokenizer.encode(text2)

        # Should be the same (tokenizer converts to lowercase)
        self.assertEqual(encoded1, encoded2)

    def test_decode_after_load(self):
        """Loaded tokenizer must decode back to the original text.

        JSON turns integer keys into strings, so the reverse vocab has to be
        rebuilt on load or decode silently produces '<UNK_n>' markers.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tokenizer.json")
            self.tokenizer.save(path)
            loaded = Tokenizer.load(path)

            text = "hello world"
            encoded = loaded.encode(text)
            self.assertEqual(loaded.decode(encoded), text)

    def test_encode_decode_roundtrip_lossless(self):
        """Encoding then decoding preserves the input text exactly."""
        for text in ("hello world", "the quick brown fox", "ciao come stai"):
            encoded = self.tokenizer.encode(text)
            self.assertEqual(self.tokenizer.decode(encoded), text)


if __name__ == '__main__':
    unittest.main()
