#!/usr/bin/env python3
"""
Example of using SmallLLM API with Byok and similar services
"""

import requests
import json

# Start the API server first:
# python -m smallllm.api
# or: smallllm-api

BASE_URL = "http://localhost:5000"


def test_basic_api():
    """Test basic API endpoints."""
    print("Testing SmallLLM API...")
    
    # Health check
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Health: {response.json()}")
    
    # Model info
    response = requests.get(f"{BASE_URL}/api/info")
    print(f"Model Info: {response.json()}")
    
    # List models
    response = requests.get(f"{BASE_URL}/api/models")
    print(f"Available Models: {response.json()}")


def test_generation():
    """Test text generation."""
    print("\nTesting text generation...")
    
    data = {
        "prompt": "Ciao come stai?",
        "max_tokens": 20,
        "temperature": 0.7,
        "config": "tiny"
    }
    
    response = requests.post(f"{BASE_URL}/api/generate", json=data)
    result = response.json()
    print(f"Generated: {result}")


def test_chat():
    """Test chat endpoint."""
    print("\nTesting chat...")
    
    data = {
        "prompt": "Raccontami una barzelletta",
        "max_tokens": 30,
        "temperature": 0.9
    }
    
    response = requests.post(f"{BASE_URL}/api/chat", json=data)
    result = response.json()
    print(f"AI Response: {result['response']}")


def test_byok_compatibility():
    """Test Byok-compatible endpoints."""
    print("\nTesting Byok compatibility...")
    
    # List models (Byok style)
    response = requests.get(f"{BASE_URL}/v1/models")
    print(f"Byok Models: {response.json()}")
    
    # Chat completions (Byok style)
    data = {
        "model": "smallllm:tiny",
        "messages": [
            {"role": "user", "content": "Ciao!"}
        ],
        "max_tokens": 20,
        "temperature": 0.7
    }
    
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=data)
    result = response.json()
    print(f"Byok Chat Response: {result}")


def test_tokenization():
    """Test tokenization endpoints."""
    print("\nTesting tokenization...")
    
    # Tokenize
    data = {"text": "Ciao mondo!", "vocab_size": 2048}
    response = requests.post(f"{BASE_URL}/api/tokenize", json=data)
    print(f"Tokens: {response.json()}")
    
    # Detokenize
    token_ids = response.json()["tokens"]
    data = {"token_ids": token_ids, "vocab_size": 2048}
    response = requests.post(f"{BASE_URL}/api/detokenize", json=data)
    print(f"Detokenized: {response.json()}")


def test_batch_generation():
    """Test batch generation."""
    print("\nTesting batch generation...")
    
    data = {
        "prompts": [
            "Ciao!",
            "Come va?",
            "Che ore sono?"
        ],
        "max_tokens": 15,
        "temperature": 0.7,
        "config": "tiny"
    }
    
    response = requests.post(f"{BASE_URL}/api/batch-generate", json=data)
    result = response.json()
    print(f"Batch Results: {len(result['results'])} responses")
    for i, r in enumerate(result['results']):
        print(f"  {i+1}. {r['prompt']} -> {r['generated_text'][:50]}...")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SmallLLM API Example")
    print("=" * 60)
    
    try:
        test_basic_api()
        test_generation()
        test_chat()
        test_byok_compatibility()
        test_tokenization()
        test_batch_generation()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API server.")
        print("Make sure to start the server first:")
        print("  python -m smallllm.api")
        print("or")
        print("  smallllm-api")


if __name__ == '__main__':
    main()
