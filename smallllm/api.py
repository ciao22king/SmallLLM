"""
REST API for SmallLLM
Lightweight API that can be used with sites like Byok
"""

import time
from typing import Optional
from flask import Flask, request, jsonify

try:
    from .model import SmallLLM
    from .tokenizer import Tokenizer
    from .config import get_config_by_name
    HAS_CORE = True
except ImportError:
    HAS_CORE = False


app = Flask(__name__)

# Global model instance (lazy loaded)
MODEL_INSTANCE: Optional[SmallLLM] = None
MODEL_CONFIG: str = "small"
# Cache of models built for explicitly requested configs
_MODEL_CACHE: dict = {}


def get_model(config_name: str = None) -> SmallLLM:
    """Get or create a model instance for the requested configuration.

    An explicit ``config_name`` builds (and caches) a model for that config
    without changing the server default in ``MODEL_CONFIG``.
    """
    global MODEL_INSTANCE

    if config_name is None:
        if MODEL_INSTANCE is None:
            MODEL_INSTANCE = SmallLLM(get_config_by_name(MODEL_CONFIG))
        return MODEL_INSTANCE

    # Explicit config: keep a per-config cache, leave the default untouched.
    if _MODEL_CACHE.get(config_name) is None:
        _MODEL_CACHE[config_name] = SmallLLM(get_config_by_name(config_name))
    return _MODEL_CACHE[config_name]


@app.route('/')
def index():
    """API info endpoint."""
    return jsonify({
        "name": "SmallLLM API",
        "version": "0.1.0",
        "description": "Tiny AI API for old hardware (2011-era PCs)",
        "endpoints": {
            "/api/generate": "POST - Generate text",
            "/api/chat": "POST - Chat with AI",
            "/api/tokenize": "POST - Tokenize text",
            "/api/detokenize": "POST - Detokenize text",
            "/api/info": "GET - Model information",
            "/api/config": "GET/POST - Model configuration",
            "/api/health": "GET - Health check",
        },
        "configurations": ["tiny", "small", "medium"],
        "default_config": MODEL_CONFIG,
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        get_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": MODEL_INSTANCE is not None,
            "config": MODEL_CONFIG,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/info', methods=['GET'])
def info():
    """Get model information."""
    try:
        model = get_model()
        config = model.config

        return jsonify({
            "model": "SmallLLM",
            "config": MODEL_CONFIG,
            "vocab_size": config.vocab_size,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "max_seq_len": config.max_seq_len,
            "estimated_params": config.num_params,
            "memory_usage": model.get_memory_usage(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """Get or set model configuration."""
    global MODEL_CONFIG, MODEL_INSTANCE

    if request.method == 'GET':
        return jsonify({"current_config": MODEL_CONFIG, "available": ["tiny", "small", "medium"]})

    elif request.method == 'POST':
        data = request.get_json(silent=True) or {}
        new_config = data.get('config')

        if new_config not in ["tiny", "small", "medium"]:
            return jsonify({"error": f"Invalid config: {new_config}. Use: tiny, small, medium"}), 400

        MODEL_CONFIG = new_config
        MODEL_INSTANCE = None  # Force reload on next request
        _MODEL_CACHE.clear()

        return jsonify({"status": "ok", "config": MODEL_CONFIG})


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Generate text from a prompt.

    Request body:
    {
        "prompt": "Your text here",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_k": 50,
        "config": "small"
    }
    """
    try:
        data = request.get_json(silent=True) or {}

        prompt = _extract_prompt(data)
        max_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.7)
        top_k = data.get('top_k', 50)
        config_name = data.get('config', MODEL_CONFIG)
        do_sample = data.get('do_sample', True)

        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        model = get_model(config_name)

        start_time = time.time()
        result = model.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
        )
        elapsed = time.time() - start_time

        return jsonify({
            "generated_text": result,
            "config": config_name,
            "time_seconds": round(elapsed, 4),
            "tokens_generated": max_tokens,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _extract_prompt(data: dict) -> str:
    """Return the user prompt from either a plain ``prompt`` or OpenAI-style
    ``messages`` payload (the /v1 endpoints advertise OpenAI compatibility)."""
    prompt = data.get('prompt', '')
    if isinstance(prompt, list):
        # OpenAI's completions API also accepts a list of prompts; use the first.
        prompt = prompt[0] if prompt else ''
    if prompt:
        return prompt

    messages = data.get('messages') or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get('role') == 'user':
            return message.get('content', '') or ''
    return ''


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat with the AI.

    Request body (either form is accepted):
    {
        "prompt": "Your message",
        "max_tokens": 50,
        "temperature": 0.7,
        "config": "small"
    }
    {
        "messages": [{"role": "user", "content": "Your message"}],
        "max_tokens": 50,
        "config": "small"
    }
    """
    try:
        data = request.get_json(silent=True) or {}

        prompt = _extract_prompt(data)
        max_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.7)
        config_name = data.get('config', MODEL_CONFIG)

        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        model = get_model(config_name)

        start_time = time.time()
        result = model.chat(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.time() - start_time

        return jsonify({
            "response": result,
            "config": config_name,
            "time_seconds": round(elapsed, 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tokenize', methods=['POST'])
def tokenize():
    """
    Tokenize text.

    Request body:
    {
        "text": "Text to tokenize",
        "vocab_size": 2048
    }
    """
    try:
        data = request.get_json(silent=True) or {}

        text = data.get('text', '')
        vocab_size = data.get('vocab_size', 2048)

        if not text:
            return jsonify({"error": "text is required"}), 400

        tokenizer = Tokenizer(vocab_size=vocab_size)
        token_ids = tokenizer.encode(text)

        return jsonify({
            "tokens": token_ids,
            "num_tokens": len(token_ids),
            "vocab_size": tokenizer.get_vocab_size(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/detokenize', methods=['POST'])
def detokenize():
    """
    Detokenize text.

    Request body:
    {
        "token_ids": [1, 2, 3, 4],
        "vocab_size": 2048
    }
    """
    try:
        data = request.get_json(silent=True) or {}

        token_ids = data.get('token_ids', [])
        vocab_size = data.get('vocab_size', 2048)

        if not token_ids:
            return jsonify({"error": "token_ids is required"}), 400

        tokenizer = Tokenizer(vocab_size=vocab_size)
        text = tokenizer.decode(token_ids)

        return jsonify({
            "text": text,
            "num_tokens": len(token_ids),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/batch-generate', methods=['POST'])
def batch_generate():
    """
    Generate multiple responses from prompts.

    Request body:
    {
        "prompts": ["prompt 1", "prompt 2"],
        "max_tokens": 50,
        "temperature": 0.7,
        "config": "small"
    }
    """
    try:
        data = request.get_json(silent=True) or {}

        prompts = data.get('prompts', [])
        max_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.7)
        config_name = data.get('config', MODEL_CONFIG)

        if not prompts:
            return jsonify({"error": "prompts is required"}), 400

        model = get_model(config_name)

        results = []
        for prompt in prompts:
            start_time = time.time()
            result = model.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.time() - start_time

            results.append({
                "prompt": prompt,
                "generated_text": result,
                "time_seconds": round(elapsed, 4),
            })

        return jsonify({
            "results": results,
            "config": config_name,
            "num_prompts": len(prompts),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available model configurations."""
    configs = {
        "tiny": {
            "vocab_size": 1024,
            "embedding_dim": 32,
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 2,
            "max_seq_len": 64,
            "estimated_memory_mb": 0.76,
            "recommended": "For very old hardware (2011-era)",
        },
        "small": {
            "vocab_size": 2048,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_layers": 3,
            "num_heads": 4,
            "max_seq_len": 128,
            "estimated_memory_mb": 3.79,
            "recommended": "Default configuration",
        },
        "medium": {
            "vocab_size": 4096,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "max_seq_len": 256,
            "estimated_memory_mb": 28.0,
            "recommended": "For better quality on capable hardware",
        },
    }

    return jsonify({
        "models": configs,
        "current": MODEL_CONFIG,
    })


@app.route('/api/memory', methods=['GET'])
def memory_info():
    """Get memory usage information."""
    try:
        model = get_model()
        memory = model.get_memory_usage()

        return jsonify({
            "config": MODEL_CONFIG,
            "memory_usage": memory,
            "total_bytes": memory.get('total', 0),
            "total_mb": round(memory.get('total', 0) / (1024 * 1024), 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stream-generate', methods=['POST'])
def stream_generate():
    """
    Generate text with streaming (for Byok-style APIs).

    This endpoint returns a simple streaming response.
    For true streaming, use the /api/generate endpoint.

    Request body:
    {
        "prompt": "Your text here",
        "max_tokens": 50,
        "temperature": 0.7,
        "config": "small"
    }
    """
    try:
        data = request.get_json(silent=True) or {}

        prompt = _extract_prompt(data)
        max_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.7)
        config_name = data.get('config', MODEL_CONFIG)

        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        model = get_model(config_name)

        start_time = time.time()
        result = model.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.time() - start_time

        # Return in a format compatible with Byok and similar services
        return jsonify({
            "id": f"smallllm-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": f"smallllm:{config_name}",
            "choices": [{
                "text": result,
                "index": 0,
                "finish_reason": "length" if len(result) >= max_tokens else "stop",
            }],
            "usage": {
                "prompt_tokens": len(model.tokenizer.encode(prompt)),
                "completion_tokens": max_tokens,
                "total_tokens": len(model.tokenizer.encode(prompt)) + max_tokens,
            },
            "time_seconds": round(elapsed, 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# For Byok compatibility - alias endpoints
@app.route('/v1/chat/completions', methods=['POST'])
def byok_chat_completions():
    """Byok/OpenAI-compatible chat completions endpoint."""
    try:
        data = request.get_json(silent=True) or {}

        prompt = _extract_prompt(data)
        max_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.7)
        config_name = data.get('config', MODEL_CONFIG)

        if not prompt:
            return jsonify({"error": {"message": "prompt is required",
                                      "type": "invalid_request_error"}}), 400

        model = get_model(config_name)
        result = model.chat(prompt=prompt, max_new_tokens=max_tokens,
                            temperature=temperature)

        return jsonify({
            "id": f"smallllm-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"smallllm:{config_name}",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(model.tokenizer.encode(prompt)),
                "completion_tokens": len(model.tokenizer.encode(result)),
                "total_tokens": (len(model.tokenizer.encode(prompt))
                                 + len(model.tokenizer.encode(result))),
            },
        })
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


@app.route('/v1/completions', methods=['POST'])
def byok_completions():
    """Byok/OpenAI-compatible completions endpoint."""
    try:
        data = request.get_json(silent=True) or {}

        prompt = _extract_prompt(data)
        max_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.7)
        config_name = data.get('config', MODEL_CONFIG)

        if not prompt:
            return jsonify({"error": {"message": "prompt is required",
                                      "type": "invalid_request_error"}}), 400

        model = get_model(config_name)
        result = model.generate(prompt=prompt, max_new_tokens=max_tokens,
                                temperature=temperature)

        return jsonify({
            "id": f"smallllm-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": f"smallllm:{config_name}",
            "choices": [{
                "text": result,
                "index": 0,
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": len(model.tokenizer.encode(prompt)),
                "completion_tokens": len(model.tokenizer.encode(result)),
                "total_tokens": (len(model.tokenizer.encode(prompt))
                                 + len(model.tokenizer.encode(result))),
            },
        })
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


@app.route('/v1/models', methods=['GET'])
def byok_models():
    """Byok-compatible models endpoint."""
    return list_models()


@app.route('/v1/models/<model_name>', methods=['GET'])
def byok_model_info(model_name: str):
    """Byok-compatible model info endpoint."""
    configs = {
        "tiny": {
            "id": "smallllm:tiny",
            "object": "model",
            "created": 1700000000,
            "owned_by": "smallllm",
            "permission": [{"object": "model_permission", "allow_create_engine": False}],
            "root": "smallllm:tiny",
            "parent": None,
        },
        "small": {
            "id": "smallllm:small",
            "object": "model",
            "created": 1700000000,
            "owned_by": "smallllm",
            "permission": [{"object": "model_permission", "allow_create_engine": False}],
            "root": "smallllm:small",
            "parent": None,
        },
        "medium": {
            "id": "smallllm:medium",
            "object": "model",
            "created": 1700000000,
            "owned_by": "smallllm",
            "permission": [{"object": "model_permission", "allow_create_engine": False}],
            "root": "smallllm:medium",
            "parent": None,
        },
    }

    if model_name in configs:
        return jsonify(configs[model_name])
    else:
        return jsonify({"error": f"Model {model_name} not found"}), 404


def run_api(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Run the SmallLLM API server."""
    print(f"Starting SmallLLM API server on {host}:{port}")
    print(f"Using configuration: {MODEL_CONFIG}")
    print("\nAvailable endpoints:")
    print("  GET  /                    - API info")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/info            - Model information")
    print("  GET  /api/models          - List available models")
    print("  GET  /api/memory          - Memory usage")
    print("  POST /api/generate        - Generate text")
    print("  POST /api/chat            - Chat with AI")
    print("  POST /api/tokenize        - Tokenize text")
    print("  POST /api/detokenize      - Detokenize text")
    print("  POST /api/batch-generate  - Batch generation")
    print("  POST /api/stream-generate - Streaming generation")
    print("\nByok-compatible endpoints:")
    print("  POST /v1/chat/completions - Chat completions")
    print("  POST /v1/completions      - Completions")
    print("  GET  /v1/models           - List models")
    print("  GET  /v1/models/<name>    - Model info")
    print("\nPress Ctrl+C to stop")

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_api()
