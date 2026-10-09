# SmallLLM

**A Tiny AI That Runs on 2011-Era PCs**

SmallLLM is a lightweight transformer-based language model designed to run on very old hardware (2011-era PCs with limited RAM and CPU). It implements lazy loading of parameters to minimize memory usage and provides a complete CLI interface.

## Features

- **Tiny Transformer Architecture**: 2-4 layer transformer with configurable dimensions
- **Lazy Loading**: Parameters are loaded on-demand to save memory
- **8-bit Quantization**: Optional 8-bit weight quantization for even lower memory usage
- **Complete CLI**: Full command-line interface with multiple commands
- **Tokenization**: Built-in lightweight tokenizer
- **Generation**: Text generation with temperature and top-k sampling
- **Chat Mode**: Interactive chat interface
- **Training Support**: Basic training capabilities (for demonstration)

## Requirements

- Python 3.6 or later
- NumPy 1.16.0 or later

```bash
pip install numpy
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ciao22king/SmallLLM.git
cd SmallLLM

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install smallllm
```

## Quick Start

### Generate Text

```bash
smallllm generate --prompt "Hello, how are you?"
```

### Chat with AI

```bash
smallllm chat --prompt "What is the meaning of life?"
```

### Start Interactive Mode

```bash
smallllm interactive
```

### Tokenize Text

```bash
smallllm tokenize --text "This is a test sentence."
```

### Show Model Information

```bash
smallllm info
```

### Check Memory Usage

```bash
smallllm memory
```

## Configuration Options

SmallLLM provides three predefined configurations:

| Config | Vocab Size | Embedding | Hidden | Layers | Heads | Max Seq | Est. Params |
|--------|------------|-----------|--------|--------|-------|---------|-------------|
| tiny   | 1024       | 32        | 64     | 2      | 2     | 64      | ~200K       |
| small  | 2048       | 64        | 128    | 3      | 4     | 128     | ~1.5M       |
| medium | 4096       | 128       | 256    | 4      | 8     | 256     | ~12M        |

### Using Different Configurations

```bash
# Use tiny configuration (lowest memory)
smallllm generate --prompt "Hello" --config tiny

# Use medium configuration (better quality, more memory)
smallllm generate --prompt "Hello" --config medium
```

## CLI Commands

### generate
Generate text from a prompt.

```bash
smallllm generate --prompt "Once upon a time" --max-tokens 100 --temperature 0.9
```

Options:
- `--prompt, -p`: Input prompt (required)
- `--max-tokens, -m`: Maximum tokens to generate (default: 50)
- `--temperature, -t`: Sampling temperature (default: 0.7)
- `--top-k, -k`: Number of top tokens to consider (default: 50)
- `--no-sample`: Use greedy decoding instead of sampling
- `--config, -c`: Model configuration (tiny, small, medium)
- `--model-path`: Path to pre-trained model weights

### chat
Simple chat with the AI.

```bash
smallllm chat --prompt "Tell me a joke"
```

Options: Same as generate, plus model configuration.

### interactive
Start an interactive chat session.

```bash
smallllm interactive
```

In interactive mode, you can:
- Type messages to chat with the AI
- Type `help` or `?` for commands
- Type `info` to show model information
- Type `memory` to show memory usage
- Type `clear` to clear the screen
- Type `quit`, `exit`, or `bye` to end

### train
Basic training on text (for demonstration).

```bash
# Train on a text string
smallllm train --text "This is a sample text for training." --epochs 1

# Train on a file
smallllm train --file data.txt --epochs 3 --batch-size 4
```

Options:
- `--text`: Text to train on
- `--file, -f`: File containing training text
- `--epochs, -e`: Number of epochs (default: 1)
- `--batch-size, -b`: Batch size (default: 4)
- `--learning-rate, -lr`: Learning rate (default: 1e-4)
- `--config, -c`: Model configuration (default: tiny)
- `--save-path, -s`: Path to save trained model

### info
Show model information.

```bash
smallllm info --config small
```

Options:
- `--config, -c`: Configuration to inspect
- `--model-path`: Path to loaded model for actual memory info

### save
Save model weights.

```bash
smallllm save --path ./my_model --config small
```

Options:
- `--path, -p`: Directory to save weights (required)
- `--config, -c`: Model configuration

### load
Load model weights.

```bash
smallllm load --path ./my_model
```

Options:
- `--path, -p`: Directory containing weights (required)

### tokenize
Tokenize text.

```bash
smallllm tokenize --text "Hello world" --decode
```

Options:
- `--text, -t`: Text to tokenize (required)
- `--decode, -d`: Decode token IDs back to text
- `--vocab-size, -v`: Vocabulary size (default: 2048)

### memory
Show memory usage.

```bash
smallllm memory --config small
```

Options:
- `--config, -c`: Configuration to analyze
- `--model-path`: Path to loaded model for actual usage

### config
Create custom configuration.

```bash
smallllm config --vocab-size 2048 --hidden-dim 128 --num-layers 3 --save-path custom_config.json
```

Options:
- All model configuration parameters
- `--save-path, -s`: Path to save configuration (required)

## Python API

```python
from smallllm import SmallLLM, Tokenizer, ModelConfig

# Create model with default configuration
model = SmallLLM()

# Generate text
result = model.generate("Hello, how are you?", max_new_tokens=50)
print(result)

# Chat
response = model.chat("What is your name?")
print(response)

# Use custom configuration
config = ModelConfig(
    vocab_size=2048,
    embedding_dim=64,
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
)
model = SmallLLM(config)

# Tokenize
tokenizer = Tokenizer(vocab_size=2048)
token_ids = tokenizer.encode("Hello world")
text = tokenizer.decode(token_ids)
```

## Memory Optimization

SmallLLM implements several memory optimization techniques:

1. **Lazy Loading**: Parameters are initialized only when needed
2. **On-Demand Layer Loading**: Transformer layers are loaded as they're used
3. **8-bit Quantization**: Weights can be quantized to 8-bit (use `--use-8bit` in config)
4. **Compact Tokenizer**: Minimal vocabulary and simple tokenization

### Estimated Memory Usage

| Configuration | 32-bit Weights | 8-bit Weights |
|---------------|----------------|---------------|
| tiny          | ~0.8 MB        | ~0.2 MB       |
| small         | ~6 MB          | ~1.5 MB       |
| medium        | ~48 MB         | ~12 MB        |

*Note: These are estimates for model weights only. Actual usage includes Python overhead.*

## Training on Old Hardware

For training on very old hardware:

1. Use the `tiny` configuration
2. Set `batch_size=1` or `batch_size=2`
3. Use a small learning rate (e.g., 1e-5)
4. Train on very short sequences (max_seq_len=32 or 64)

```bash
smallllm train --text "Short text for training." --config tiny --batch-size 1 --epochs 1
```

## Performance Tips

1. **For Generation**: Use `tiny` or `small` configurations
2. **For Better Quality**: Use `small` or `medium` configurations
3. **For Minimum Memory**: Use `tiny` with 8-bit quantization
4. **For Speed**: Reduce `max_new_tokens` and use greedy decoding (`--no-sample`)

## Limitations

SmallLLM is designed for old hardware, which means:

- **Quality**: Output quality is limited by model size
- **Speed**: Generation is slower than modern LLMs
- **Training**: Training capabilities are basic (for demonstration only)
- **Features**: Limited feature set compared to full-scale LLMs

However, it provides a working AI that can run on hardware from 2011!

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

SmallLLM is inspired by the transformer architecture and the need for AI that can run on limited hardware. Special thanks to the open-source AI community for making this possible.

---

**Enjoy your tiny AI on old hardware!**
