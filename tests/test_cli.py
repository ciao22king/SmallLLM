"""
Tests for CLI
"""

import unittest
import tempfile
import os
import sys
from io import StringIO
from smallllm.cli import CLI


class TestCLI(unittest.TestCase):
    """Test cases for CLI."""

    def setUp(self):
        """Set up test fixtures."""
        self.cli = CLI()

    def test_parser_creation(self):
        """Test parser creation."""
        self.assertIsNotNone(self.cli.parser)

    def test_info_command(self):
        """Test info command."""
        args = ['info', '--config', 'tiny']
        result = self.cli.run(args)

        self.assertEqual(result, 0)

    def test_info_command_output(self):
        """Test info command output."""
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            args = ['info', '--config', 'tiny']
            result = self.cli.run(args)

            output = captured_output.getvalue()

            self.assertEqual(result, 0)
            self.assertIn('SmallLLM', output)
            self.assertIn('Configuration', output)
        finally:
            sys.stdout = old_stdout

    def test_memory_command(self):
        """Test memory command."""
        args = ['memory', '--config', 'tiny']
        result = self.cli.run(args)

        self.assertEqual(result, 0)

    def test_tokenize_command(self):
        """Test tokenize command."""
        args = ['tokenize', '--text', 'hello world']
        result = self.cli.run(args)

        self.assertEqual(result, 0)

    def test_tokenize_with_decode(self):
        """Test tokenize with decode option."""
        args = ['tokenize', '--text', 'hello', '--decode']
        result = self.cli.run(args)

        self.assertEqual(result, 0)

    def test_config_command(self):
        """Test config command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            args = [
                'config',
                '--vocab-size', '1024',
                '--embedding-dim', '32',
                '--hidden-dim', '64',
                '--num-layers', '2',
                '--num-heads', '2',
                '--save-path', config_path,
            ]
            result = self.cli.run(args)

            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(config_path))

    def test_generate_command(self):
        """Test generate command."""
        args = ['generate', '--prompt', 'hello', '--max-tokens', '10']
        result = self.cli.run(args)

        self.assertEqual(result, 0)

    def test_chat_command(self):
        """Test chat command."""
        args = ['chat', '--prompt', 'hello', '--max-tokens', '10']
        result = self.cli.run(args)

        self.assertEqual(result, 0)

    def test_invalid_command(self):
        """Test handling of invalid command."""
        args = ['invalid_command']
        try:
            result = self.cli.run(args)
            # Should raise SystemExit or return non-zero
            self.assertNotEqual(result, 0)
        except SystemExit as e:
            # argparse exits with code 2 for invalid commands
            self.assertEqual(e.code, 2)

    def test_no_command(self):
        """Test handling of no command."""
        args = []

        # Capture stdout to prevent help from printing
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = self.cli.run(args)
            self.assertEqual(result, 0)
        finally:
            sys.stdout = old_stdout

    def test_save_command(self):
        """Test save command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            args = ['save', '--path', save_path, '--config', 'tiny']
            result = self.cli.run(args)

            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(save_path))

    def test_load_command(self):
        """Test load command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")

            # First save a model
            self.cli.run(['save', '--path', save_path, '--config', 'tiny'])

            # Then load it
            args = ['load', '--path', save_path]
            result = self.cli.run(args)

            self.assertEqual(result, 0)

    def test_train_command_with_text(self):
        """Test train command with text."""
        args = ['train', '--text', 'hello world', '--epochs', '1']
        result = self.cli.run(args)

        self.assertEqual(result, 0)

    def test_train_command_with_file(self):
        """Test train command with file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_file = os.path.join(tmpdir, "text.txt")
            with open(text_file, 'w') as f:
                f.write("hello world test")

            args = ['train', '--file', text_file, '--epochs', '1']
            result = self.cli.run(args)

            self.assertEqual(result, 0)

    def test_quantize_command_fresh(self):
        """Test quantize command without a saved model."""
        result = self.cli.run(['quantize', '--config', 'tiny'])
        self.assertEqual(result, 0)

    def test_quantize_command_from_saved(self):
        """Test quantize command against a saved model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            self.cli.run(['save', '--path', save_path, '--config', 'tiny'])

            out_path = os.path.join(tmpdir, "quantized")
            result = self.cli.run(['quantize', '--path', save_path, '--save-path', out_path])

            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(os.path.join(out_path, "config.json")))

    def test_interactive_accepts_config(self):
        """Interactive mode must accept --config (regression: it had no args)."""
        args = self.cli.parser.parse_args(['interactive', '--config', 'tiny'])
        self.assertEqual(args.config, 'tiny')


class TestCLIModelLoading(unittest.TestCase):
    """Test CLI model loading functionality."""

    def test_get_model_tiny(self):
        """Test getting tiny model."""
        cli = CLI()
        model = cli._get_model('tiny')

        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, 1024)

    def test_get_model_small(self):
        """Test getting small model."""
        cli = CLI()
        model = cli._get_model('small')

        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, 2048)

    def test_get_model_medium(self):
        """Test getting medium model."""
        cli = CLI()
        model = cli._get_model('medium')

        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, 4096)

    def test_model_caching(self):
        """Test that model is cached."""
        cli = CLI()

        model1 = cli._get_model('tiny')
        model2 = cli._get_model('tiny')

        # Should return the same instance
        self.assertIs(model1, model2)


if __name__ == '__main__':
    unittest.main()
