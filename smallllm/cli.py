"""
Command Line Interface for SmallLLM
Complete CLI with multiple commands and lazy loading support
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List
import time
import textwrap
import numpy as np

try:
    from .model import SmallLLM
    from .tokenizer import Tokenizer
    from .config import ModelConfig, get_config_by_name
    HAS_CORE = True
except ImportError:
    HAS_CORE = False


class CLI:
    """
    Command Line Interface for SmallLLM.
    Supports multiple commands with lazy loading of parameters.
    """

    def __init__(self):
        self.parser = self._create_parser()
        self.model: Optional[SmallLLM] = None
        self.config: Optional[ModelConfig] = None
        self.config_name: Optional[str] = None
        self.tokenizer: Optional[Tokenizer] = None
        self._weights_loaded = False

        # Map command names to their handler methods
        self.COMMANDS = {
            'generate': self._cmd_generate,
            'chat': self._cmd_chat,
            'train': self._cmd_train,
            'info': self._cmd_info,
            'save': self._cmd_save,
            'load': self._cmd_load,
            'tokenize': self._cmd_tokenize,
            'memory': self._cmd_memory,
            'config': self._cmd_config,
            'quantize': self._cmd_quantize,
            'interactive': self._cmd_interactive,
        }

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all commands."""
        parser = argparse.ArgumentParser(
            prog='smallllm',
            description='SmallLLM - Tiny AI for old hardware (2011-era PCs)',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent('''
                Examples:
                  smallllm generate --prompt "Hello, how are you?"
                  smallllm chat --prompt "What is the meaning of life?"
                  smallllm train --text "This is a sample text."
                  smallllm info
                  smallllm save --path ./my_model
                  smallllm load --path ./my_model
            ''')
        )

        subparsers = parser.add_subparsers(
            dest='command',
            title='Commands',
            description='Available commands'
        )

        # Generate command
        gen_parser = subparsers.add_parser(
            'generate',
            help='Generate text from a prompt',
            description='Generate text from a given prompt'
        )
        gen_parser.add_argument(
            '--prompt', '-p',
            type=str,
            required=True,
            help='Input prompt for text generation'
        )
        gen_parser.add_argument(
            '--max-tokens', '-m',
            type=int,
            default=50,
            help='Maximum number of tokens to generate (default: 50)'
        )
        gen_parser.add_argument(
            '--temperature', '-t',
            type=float,
            default=0.7,
            help='Sampling temperature (default: 0.7)'
        )
        gen_parser.add_argument(
            '--top-k', '-k',
            type=int,
            default=50,
            help='Number of top tokens to consider (default: 50)'
        )
        gen_parser.add_argument(
            '--no-sample',
            action='store_true',
            help='Use greedy decoding instead of sampling'
        )
        gen_parser.add_argument(
            '--config', '-c',
            type=str,
            default='small',
            help='Model configuration: tiny, small, medium (default: small)'
        )
        gen_parser.add_argument(
            '--model-path',
            type=str,
            default=None,
            help='Path to load pre-trained model weights'
        )

        # Chat command
        chat_parser = subparsers.add_parser(
            'chat',
            help='Chat with the AI',
            description='Simple chat interface with the AI'
        )
        chat_parser.add_argument(
            '--prompt', '-p',
            type=str,
            required=True,
            help='Your message to the AI'
        )
        chat_parser.add_argument(
            '--max-tokens', '-m',
            type=int,
            default=50,
            help='Maximum number of tokens in response (default: 50)'
        )
        chat_parser.add_argument(
            '--temperature', '-t',
            type=float,
            default=0.7,
            help='Sampling temperature (default: 0.7)'
        )
        chat_parser.add_argument(
            '--config', '-c',
            type=str,
            default='small',
            help='Model configuration: tiny, small, medium (default: small)'
        )
        chat_parser.add_argument(
            '--model-path',
            type=str,
            default=None,
            help='Path to load pre-trained model weights'
        )

        # Train command
        train_parser = subparsers.add_parser(
            'train',
            help='Train the model (basic implementation)',
            description='Basic training on provided text'
        )
        train_parser.add_argument(
            '--text',
            type=str,
            default=None,
            help='Text to train on (or use --file)'
        )
        train_parser.add_argument(
            '--file', '-f',
            type=str,
            default=None,
            help='File containing training text'
        )
        train_parser.add_argument(
            '--epochs', '-e',
            type=int,
            default=1,
            help='Number of training epochs (default: 1)'
        )
        train_parser.add_argument(
            '--batch-size', '-b',
            type=int,
            default=4,
            help='Batch size for training (default: 4)'
        )
        train_parser.add_argument(
            '--learning-rate', '-lr',
            type=float,
            default=1e-4,
            help='Learning rate (default: 1e-4)'
        )
        train_parser.add_argument(
            '--config', '-c',
            type=str,
            default='tiny',
            help='Model configuration: tiny, small, medium (default: tiny)'
        )
        train_parser.add_argument(
            '--save-path', '-s',
            type=str,
            default=None,
            help='Path to save trained model'
        )

        # Info command
        info_parser = subparsers.add_parser(
            'info',
            help='Show model information',
            description='Display information about the model'
        )
        info_parser.add_argument(
            '--config', '-c',
            type=str,
            default='small',
            help='Model configuration to inspect: tiny, small, medium (default: small)'
        )
        info_parser.add_argument(
            '--model-path',
            type=str,
            default=None,
            help='Path to loaded model for memory info'
        )

        # Save command
        save_parser = subparsers.add_parser(
            'save',
            help='Save model weights',
            description='Save model weights to a directory'
        )
        save_parser.add_argument(
            '--path', '-p',
            type=str,
            required=True,
            help='Directory to save model weights'
        )
        save_parser.add_argument(
            '--config', '-c',
            type=str,
            default='small',
            help='Model configuration: tiny, small, medium (default: small)'
        )

        # Load command
        load_parser = subparsers.add_parser(
            'load',
            help='Load model weights',
            description='Load model weights from a directory'
        )
        load_parser.add_argument(
            '--path', '-p',
            type=str,
            required=True,
            help='Directory containing model weights'
        )

        # Quantize command
        quantize_parser = subparsers.add_parser(
            'quantize',
            help='Quantize a model to 8-bit',
            description='Quantize a model to 8-bit and optionally save it'
        )
        quantize_parser.add_argument(
            '--path', '-p',
            type=str,
            default=None,
            help='Directory of a saved model to quantize (default: a fresh in-memory model)'
        )
        quantize_parser.add_argument(
            '--save-path', '-s',
            type=str,
            default=None,
            help='Directory to save the quantized model (defaults to --path)'
        )
        quantize_parser.add_argument(
            '--config', '-c',
            type=str,
            default='small',
            help='Model configuration when no --path is given: tiny, small, medium'
        )

        # Tokenize command
        tokenize_parser = subparsers.add_parser(
            'tokenize',
            help='Tokenize text',
            description='Tokenize input text and show tokens'
        )
        tokenize_parser.add_argument(
            '--text', '-t',
            type=str,
            required=True,
            help='Text to tokenize'
        )
        tokenize_parser.add_argument(
            '--decode', '-d',
            action='store_true',
            help='Decode token IDs back to text'
        )
        tokenize_parser.add_argument(
            '--vocab-size', '-v',
            type=int,
            default=2048,
            help='Vocabulary size (default: 2048)'
        )

        # Memory command
        mem_parser = subparsers.add_parser(
            'memory',
            help='Show memory usage',
            description='Display estimated memory usage'
        )
        mem_parser.add_argument(
            '--config', '-c',
            type=str,
            default='small',
            help='Model configuration: tiny, small, medium (default: small)'
        )
        mem_parser.add_argument(
            '--model-path',
            type=str,
            default=None,
            help='Path to loaded model for actual memory usage'
        )

        # Config command
        config_parser = subparsers.add_parser(
            'config',
            help='Create a custom configuration',
            description='Create and save a custom model configuration'
        )
        config_parser.add_argument(
            '--vocab-size',
            type=int,
            default=2048,
            help='Vocabulary size (default: 2048)'
        )
        config_parser.add_argument(
            '--embedding-dim',
            type=int,
            default=64,
            help='Embedding dimension (default: 64)'
        )
        config_parser.add_argument(
            '--hidden-dim',
            type=int,
            default=128,
            help='Hidden dimension (default: 128)'
        )
        config_parser.add_argument(
            '--num-layers',
            type=int,
            default=3,
            help='Number of transformer layers (default: 3)'
        )
        config_parser.add_argument(
            '--num-heads',
            type=int,
            default=4,
            help='Number of attention heads (default: 4)'
        )
        config_parser.add_argument(
            '--max-seq-len',
            type=int,
            default=128,
            help='Maximum sequence length (default: 128)'
        )
        config_parser.add_argument(
            '--save-path', '-s',
            type=str,
            required=True,
            help='Path to save configuration file'
        )

        # Interactive mode
        interactive_parser = subparsers.add_parser(
            'interactive',
            help='Start interactive chat mode',
            description='Start an interactive chat session with the AI'
        )
        interactive_parser.add_argument(
            '--config', '-c',
            type=str,
            default='small',
            help='Model configuration: tiny, small, medium (default: small)'
        )
        interactive_parser.add_argument(
            '--model-path',
            type=str,
            default=None,
            help='Path to load pre-trained model weights'
        )

        return parser

    def _get_model(
        self,
        config_name: str = 'small',
        model_path: Optional[str] = None,
        force_reload: bool = False,
    ) -> SmallLLM:
        """
        Get or create model instance with lazy loading.

        Args:
            config_name: Name of configuration to use
            model_path: Path to pre-trained weights (optional)
            force_reload: Force reload of weights

        Returns:
            SmallLLM model instance
        """
        # Reuse the cached model only when it matches the requested config
        # and no explicit weights path was given.
        if self.model is not None and not force_reload and model_path is None \
                and self.config_name == config_name:
            return self.model

        # Get configuration
        config = get_config_by_name(config_name)

        # Create tokenizer
        tokenizer = Tokenizer(vocab_size=config.vocab_size)

        # Create model
        model = SmallLLM(config, tokenizer)

        # Load weights if specified
        if model_path and os.path.exists(model_path):
            try:
                model = SmallLLM.load_weights(model_path)
                self._weights_loaded = True
                print(f"Loaded model weights from: {model_path}")
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")
                print("Using randomly initialized weights instead.")

        self.model = model
        self.config = model.config
        self.config_name = config_name
        self.tokenizer = model.tokenizer

        return model

    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI with given arguments.

        Args:
            args: Command line arguments (uses sys.argv if None)

        Returns:
            Exit code
        """
        if args is None:
            args = sys.argv[1:]

        parsed = self.parser.parse_args(args)

        if not hasattr(parsed, 'command') or parsed.command is None:
            self.parser.print_help()
            return 0

        handler = self.COMMANDS.get(parsed.command)
        if handler is None:
            print(f"Unknown command: {parsed.command}")
            return 1

        try:
            return handler(parsed)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _cmd_generate(self, args: argparse.Namespace) -> int:
        """Handle generate command."""
        print(f"\nGenerating text with prompt: '{args.prompt}'")
        print(f"Configuration: {args.config}")
        print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}, Top-k: {args.top_k}")

        # Get model with lazy loading
        model = self._get_model(args.config, args.model_path)

        # Generate text
        start_time = time.time()
        result = model.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=not args.no_sample,
        )
        elapsed = time.time() - start_time

        print("\nGenerated text:")
        print("-" * 60)
        print(result)
        print("-" * 60)
        print(f"\nTime taken: {elapsed:.2f} seconds")

        return 0

    def _cmd_chat(self, args: argparse.Namespace) -> int:
        """Handle chat command."""
        print(f"\nYou: {args.prompt}")

        # Get model with lazy loading
        model = self._get_model(args.config, args.model_path)

        # Generate response
        start_time = time.time()
        response = model.chat(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        elapsed = time.time() - start_time

        print(f"\nAI: {response}")
        print(f"\nTime taken: {elapsed:.2f} seconds")

        return 0

    def _cmd_train(self, args: argparse.Namespace) -> int:
        """Handle train command."""
        print(f"\nTraining model with configuration: {args.config}")

        # Get training text
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            print("Error: Please provide --text or --file")
            return 1

        print(f"Training text length: {len(text)} characters")

        # Get model
        model = self._get_model(args.config)
        tokenizer = model.tokenizer

        # Tokenize text
        token_ids = tokenizer.encode(text)
        print(f"Tokenized to {len(token_ids)} tokens")

        # Prepare training data (very basic)
        # In a real implementation, you'd want proper batching, masking, etc.
        batch_size = args.batch_size
        num_batches = max(1, len(token_ids) // batch_size)

        print(f"Training for {args.epochs} epochs, {num_batches} batches per epoch")

        # Simple training loop
        for epoch in range(args.epochs):
            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(token_ids))

                # Create input and target (shifted by 1)
                input_batch = np.array([token_ids[start:end]])
                target_batch = np.array([token_ids[start+1:end+1]])

                # Ensure same length
                if len(target_batch[0]) < len(input_batch[0]):
                    target_batch = np.array([token_ids[start:end-1] + [0]])

                # Training step
                loss = model.train_step(
                    input_batch,
                    target_batch,
                    learning_rate=args.learning_rate,
                )

                epoch_loss += loss

                if (batch_idx + 1) % 10 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"  Epoch {epoch + 1}, Batch {batch_idx + 1}/{num_batches}: Loss = {avg_loss:.4f}")

            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")

        # Save model if requested
        if args.save_path:
            model.save_weights(args.save_path)
            print(f"Model saved to: {args.save_path}")

        return 0

    def _cmd_info(self, args: argparse.Namespace) -> int:
        """Handle info command."""
        config = get_config_by_name(args.config)

        print("\n" + "=" * 60)
        print("SmallLLM - Model Information")
        print("=" * 60)

        print(f"\nConfiguration: {args.config}")
        print("\nArchitecture:")
        print(f"  Vocabulary size: {config.vocab_size}")
        print(f"  Embedding dimension: {config.embedding_dim}")
        print(f"  Hidden dimension: {config.hidden_dim}")
        print(f"  Number of layers: {config.num_layers}")
        print(f"  Number of heads: {config.num_heads}")
        print(f"  Head dimension: {config.head_dim}")
        print(f"  Max sequence length: {config.max_seq_len}")

        print("\nTraining/Inference:")
        print(f"  Dropout: {config.dropout}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Batch size: {config.batch_size}")

        print("\nMemory Optimization:")
        print(f"  Use 8-bit: {config.use_8bit}")
        print(f"  Lazy loading: {config.lazy_load}")
        print(f"  Cache directory: {config.cache_dir}")

        print("\nGeneration:")
        print(f"  Temperature: {config.temperature}")
        print(f"  Top-k: {config.top_k}")
        print(f"  Max new tokens: {config.max_new_tokens}")

        print("\nParameter Count:")
        print(f"  Estimated parameters: {config.num_params:,}")

        # If model path provided, show actual memory usage
        if args.model_path and os.path.exists(args.model_path):
            print(f"\nActual Memory Usage (from {args.model_path}):")
            try:
                model = self._get_model(args.config, args.model_path, force_reload=True)
                memory = model.get_memory_usage()
                for key, value in memory.items():
                    print(f"  {key}: {value / 1024 / 1024:.2f} MB")
            except Exception as e:
                print(f"  Could not load model: {e}")

        print("\n" + "=" * 60)

        return 0

    def _cmd_save(self, args: argparse.Namespace) -> int:
        """Handle save command."""
        print(f"\nSaving model with configuration: {args.config}")

        # Get model
        model = self._get_model(args.config)

        # Save weights
        model.save_weights(args.path)

        print(f"Model saved to: {args.path}")

        # Show saved files
        save_path = Path(args.path)
        if save_path.exists():
            print("\nSaved files:")
            for file in save_path.iterdir():
                size = file.stat().st_size
                print(f"  {file.name}: {size / 1024:.2f} KB")

        return 0

    def _cmd_load(self, args: argparse.Namespace) -> int:
        """Handle load command."""
        print(f"\nLoading model from: {args.path}")

        if not os.path.exists(args.path):
            print(f"Error: Path does not exist: {args.path}")
            return 1

        # Try to load model
        try:
            model = SmallLLM.load_weights(args.path)
            self.model = model
            self._weights_loaded = True

            print("Model loaded successfully!")
            print(f"Configuration: {model.config.to_dict()}")
            print(f"Vocabulary size: {model.tokenizer.get_vocab_size()}")

            return 0
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _cmd_quantize(self, args: argparse.Namespace) -> int:
        """Handle quantize command."""
        if args.path:
            print(f"\nLoading model from: {args.path}")
            if not os.path.exists(args.path):
                print(f"Error: Path does not exist: {args.path}")
                return 1
            model = SmallLLM.load_weights(args.path)
        else:
            print(f"\nQuantizing a fresh '{args.config}' model")
            model = self._get_model(args.config)

        model.init_params()
        before = model.get_memory_usage()['total']
        model.quantize()
        after = model.get_memory_usage()['total']

        print("Model quantized to 8-bit.")
        print(f"  Memory before: {before / 1024:.2f} KB")
        print(f"  Memory after:  {after / 1024:.2f} KB")

        save_path = args.save_path or args.path
        if save_path:
            model.save_weights(save_path)
            print(f"Quantized model saved to: {save_path}")

        return 0

    def _cmd_tokenize(self, args: argparse.Namespace) -> int:
        """Handle tokenize command."""
        tokenizer = Tokenizer(vocab_size=args.vocab_size)

        print("\nTokenizing text:")
        print(f"  Input: '{args.text}'")

        # Encode
        token_ids = tokenizer.encode(args.text)
        print(f"\nEncoded tokens: {token_ids}")
        print(f"Number of tokens: {len(token_ids)}")

        # Show token details
        print("\nToken details:")
        for i, token_id in enumerate(token_ids):
            token = tokenizer.reverse_vocab.get(token_id, f"<UNK_{token_id}>")
            print(f"  Position {i}: ID={token_id}, Token='{token}'")

        # Decode if requested
        if args.decode:
            decoded = tokenizer.decode(token_ids)
            print(f"\nDecoded: '{decoded}'")

        return 0

    def _cmd_memory(self, args: argparse.Namespace) -> int:
        """Handle memory command."""
        config = get_config_by_name(args.config)

        print(f"\nEstimated Memory Usage for '{args.config}' configuration:")
        print("=" * 60)

        # Assuming 32-bit floats
        embedding_size = config.vocab_size * config.embedding_dim * 4
        position_embedding_size = config.max_seq_len * config.embedding_dim * 4

        # Per layer
        per_layer_attn = (
            (config.hidden_dim * config.hidden_dim * 3) +  # Q, K, V
            (config.hidden_dim * config.hidden_dim) +      # Output
            (config.hidden_dim * 2)  # Layer norm
        ) * 4  # 32-bit

        per_layer_ffn = (
            (config.hidden_dim * config.hidden_dim * 4) +  # Expand
            (config.hidden_dim * config.hidden_dim * 4) +  # Project
            (config.hidden_dim * 2)  # Layer norm
        ) * 4  # 32-bit

        per_layer_total = per_layer_attn + per_layer_ffn
        layers_total = per_layer_total * config.num_layers

        # LM head
        lm_head_size = config.hidden_dim * config.vocab_size * 4

        # Total
        total = embedding_size + position_embedding_size + layers_total + lm_head_size

        print(f"Embedding layer: {embedding_size / 1024 / 1024:.2f} MB")
        print(f"Position embedding: {position_embedding_size / 1024 / 1024:.2f} MB")
        print(f"Transformer layers ({config.num_layers}x): {layers_total / 1024 / 1024:.2f} MB")
        print(f"  - Attention per layer: {per_layer_attn / 1024 / 1024:.2f} MB")
        print(f"  - Feed-forward per layer: {per_layer_ffn / 1024 / 1024:.2f} MB")
        print(f"LM head: {lm_head_size / 1024 / 1024:.2f} MB")
        print(f"\nTotal estimated: {total / 1024 / 1024:.2f} MB")

        if config.use_8bit:
            print(f"\nWith 8-bit quantization: {total / 1024 / 1024 / 4:.2f} MB")

        # If model path provided, show actual usage
        if args.model_path and os.path.exists(args.model_path):
            print(f"\nActual memory usage (from {args.model_path}):")
            try:
                model = self._get_model(args.config, args.model_path, force_reload=True)
                memory = model.get_memory_usage()
                for key, value in memory.items():
                    print(f"  {key}: {value / 1024 / 1024:.2f} MB")
            except Exception as e:
                print(f"  Could not load model: {e}")

        return 0

    def _cmd_config(self, args: argparse.Namespace) -> int:
        """Handle config command."""
        print("\nCreating custom configuration...")

        # Create configuration
        config = ModelConfig(
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.max_seq_len,
        )

        print("\nConfiguration:")
        print(f"  Vocabulary size: {config.vocab_size}")
        print(f"  Embedding dimension: {config.embedding_dim}")
        print(f"  Hidden dimension: {config.hidden_dim}")
        print(f"  Number of layers: {config.num_layers}")
        print(f"  Number of heads: {config.num_heads}")
        print(f"  Max sequence length: {config.max_seq_len}")
        print(f"  Estimated parameters: {config.num_params:,}")

        # Save configuration
        config.save(args.save_path)
        print(f"\nConfiguration saved to: {args.save_path}")

        return 0

    def _interactive_meta_command(self, command: str, config_name: str = 'small') -> bool:
        """Handle interactive-only commands. Returns True if handled."""
        if command in ('help', '?'):
            print("\nAvailable commands:")
            print("  Any text - Send message to AI")
            print("  quit/exit/bye - Exit chat mode")
            print("  help/? - Show this help")
            print("  info - Show model information")
            print("  memory - Show memory usage")
            print("  clear - Clear screen")
            return True

        if command == 'info':
            self._cmd_info(argparse.Namespace(
                config=config_name, model_path=None, command='info'))
            return True

        if command == 'memory':
            self._cmd_memory(argparse.Namespace(
                config=config_name, model_path=None, command='memory'))
            return True

        if command == 'clear':
            print("\n" * 50)
            return True

        return False

    def _cmd_interactive(self, args: argparse.Namespace) -> int:
        """Handle interactive chat mode."""
        print("\n" + "=" * 60)
        print("SmallLLM - Interactive Chat Mode")
        print("=" * 60)
        print("Type your messages below. Type 'quit', 'exit', or 'bye' to end.")
        print("Type 'help' for commands.")
        print()

        config_name = getattr(args, 'config', 'small')
        model_path = getattr(args, 'model_path', None)
        model = self._get_model(config_name, model_path)

        while True:
            try:
                prompt = input("You: ").strip()

                if not prompt:
                    continue

                if prompt.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break

                if self._interactive_meta_command(prompt.lower(), config_name):
                    continue

                # Generate response
                start_time = time.time()
                response = model.chat(prompt, max_new_tokens=50, temperature=0.7)
                elapsed = time.time() - start_time

                print(f"\nAI: {response}")
                print(f"(Response time: {elapsed:.2f}s)")
                print()

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

        return 0


def main():
    """Main entry point for CLI."""
    cli = CLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
