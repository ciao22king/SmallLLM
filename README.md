# SmallLLM

**Your Local AI that Runs on 2011-era PCs**

SmallLLM is a lightweight language model designed to run on old hardware (2011-era PCs with limited RAM and CPU). It includes a complete ChatGPT-like GUI, a REST API, and a full CLI.

## 🌟 Features

- **🤖 Tiny Transformer Model**: 2-4 layer transformer with customizable configurations
- **💾 Lazy Loading**: Parameters loaded on-demand to save memory
- **⚡ Complete CLI**: 10+ commands for terminal usage
- **💻 GUI Application**: Modern ChatGPT-like interface with PyQt6
- **🌐 REST API**: Compatible with various services
- **🔒 100% Local**: All data stays on your computer
- **📦 8-bit Quantization**: To further reduce memory usage
- **🌍 Plugin System**: Extend functionality with plugins (Code Completer, Web Search, Math Solver, Translator, VS Code Integration)

## 📦 Installation

### Requirements

- Python 3.6 or higher
- NumPy 1.16.0 or higher
- PyQt6 or PySide6 (for GUI)
- Flask 2.0.0 or higher (for API)

```bash
pip install numpy Flask PyQt6
```

### Package Installation

```bash
# Clone the repository
git clone https://github.com/ciao22king/SmallLLM.git
cd SmallLLM

# Install in development mode
pip install -e .

# Or install requirements manually
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Run the GUI

```bash
# Run the GUI application
python -m gui.main_window

# Or after installation
smallllm-gui
```

### 2. Use the CLI

```bash
# Generate text
python -m smallllm generate --prompt "Hello, how are you?" --max-tokens 20

# Chat mode
python -m smallllm chat --prompt "What is the meaning of life?"

# Interactive mode
python -m smallllm interactive

# Show model info
python -m smallllm info
```

### 3. Run the API Server

```bash
# Start the API server
python -m smallllm.api

# Or use the entry point
smallllm-api
```

The API will be available at `http://localhost:5000`

#### API Endpoints

- `POST /api/generate` - Generate text
- `POST /api/chat` - Chat with the AI
- `POST /api/tokenize` - Tokenize text
- `POST /api/detokenize` - Detokenize text
- `GET /api/info` - Get model information
- `GET /api/health` - Health check
- `GET /api/models` - List available configurations
- `GET /api/config` / `POST /api/config` - Get or set the default configuration

OpenAI/Byok-compatible aliases are also exposed: `POST /v1/chat/completions`,
`POST /v1/completions`, `GET /v1/models` and `GET /v1/models/<name>`.

Example request:
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 20}'
```

## ⚙️ Configuration

SmallLLM provides three predefined configurations:

| Config | Vocab Size | Embedding Dim | Hidden Dim | Layers | Heads | Max Seq Len | Estimated Params |
|--------|------------|---------------|------------|-------|-------------|--------------|
| tiny | 1024 | 32 | 64 | 2 | 2 | 64 | ~100K |
| small | 2048 | 64 | 128 | 3 | 4 | 128 | ~500K |
| medium | 4096 | 128 | 256 | 4 | 8 | 256 | ~2M |

### Using a Specific Configuration

```bash
# In CLI
python -m smallllm generate --prompt "Hello" --config small

# In GUI
Select from the Settings tab
```

## 💻 GUI Features

- **ChatGPT-like Interface**: Modern chat interface with message bubbles
- **Dark Theme**: Professional dark theme
- **Multiple Tabs**: Chat, Settings, Info, Plugins
- **Plugin Management**: Enable/disable plugins easily
- **Model Configuration**: Change model settings on the fly
- **Memory Monitoring**: View memory usage in real-time
- **Model Actions**: Load, save, quantize models

### Available Plugins

1. **Code Completer** - Auto-complete code snippets
2. **VS Code Integration** - Connect to VS Code for code analysis
3. **Web Search** - Search the web for information
4. **Math Solver** - Solve mathematical expressions
5. **Translator** - Translate text between languages

## 💾 Lazy Loading

SmallLLM uses lazy loading to optimize memory usage:

- **Parameters are loaded on-demand** during generation
- **Only necessary layers** are loaded into memory
- **8-bit quantization** reduces memory usage by ~75%
- **Perfect for old hardware** with limited RAM

### Memory Usage Estimates

| Config | Without Quantization | With 8-bit Quantization |
|--------|---------------------|------------------------|
| tiny | ~10-20 MB | ~5-10 MB |
| small | ~40-60 MB | ~15-25 MB |
| medium | ~150-200 MB | ~50-75 MB |

## 📦 Model Weights

### Saving Model Weights

```bash
python -m smallllm save --path ./my_model
```

This creates a directory with:
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration
- `embedding.npy` - Embedding weights
- `position_embedding.npy` - Position embedding
- `layer_*.npy` - Transformer layer weights
- `final_norm.npy` - Final layer norm
- `lm_head.npy` - Language model head

### Loading Model Weights

```bash
python -m smallllm load --path ./my_model
```

### Training (Basic)

```bash
python -m smallllm train --text "Your training text here" --epochs 1 --config tiny
```

## ✨ Quantization

Reduce model size and memory usage with 8-bit quantization:

```bash
# Quantize the current model
python -m smallllm quantize

# Or in GUI
Click "Quantize (8-bit)" in Settings tab
```

## 💋 CLI Commands

| Command | Description |
|---------|-------------|
| `generate` | Generate text from a prompt |
| `chat` | Simple chat with the AI |
| `interactive` | Start interactive chat session |
| `train` | Train the model (basic) |
| `info` | Show model information |
| `save` | Save model weights |
| `load` | Load model weights |
| `quantize` | Quantize a model to 8-bit |
| `tokenize` | Tokenize text |
| `memory` | Show memory usage |
| `config` | Create custom configuration |

### Example Commands

```bash
# Generate with specific settings
python -m smallllm generate -p "Tell me a story" -m 50 -t 0.9 -k 40

# Chat with the AI
python -m smallllm chat -p "What is Python?" -m 30

# Show model info
python -m smallllm info --config medium

# Tokenize text
python -m smallllm tokenize -t "Hello world"

# Create custom config
python -m smallllm config --vocab-size 2048 --hidden-dim 128 --num-layers 3 -s my_config.json
```

## 💀 Requirements

See `requirements.txt` for full list of dependencies.

### Core Requirements

```
numpy>=1.16.0
Flask>=2.0.0
```

### GUI Requirements (Optional)

```
PyQt6>=6.4.0
# OR
PySide6>=6.4.0
```

### Plugin Requirements (Optional)

```
sympy>=1.12.0  # For Math Solver
requests>=2.28.0  # For Web Search
```

## 🔧 Troubleshooting

### PyQt6 Import Error

If you get `libGL.so.1` error on Linux:
```bash
sudo apt-get install libgl1-mesa-glx
```

### No Module Named Error

Make sure you have installed all requirements:
```bash
pip install -r requirements.txt
```

### Model Not Generating Good Text

The model generates random text without pre-trained weights. You need to:
1. Train the model on your data
2. Or load pre-trained weights

## 🔗 License

MIT License - See `LICENSE` file for details.

## 💁 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🐍 Author

**ciao22king** - [GitHub Profile](https://github.com/ciao22king)
