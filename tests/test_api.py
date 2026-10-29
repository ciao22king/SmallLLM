"""
Tests for the SmallLLM REST API
"""

import unittest

from smallllm import api
from smallllm.api import app


class TestAPI(unittest.TestCase):
    """Test cases for the Flask API endpoints."""

    def setUp(self):
        self.client = app.test_client()
        api.MODEL_INSTANCE = None
        api.MODEL_CONFIG = "small"
        api._MODEL_CACHE.clear()

    def test_index(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['name'], "SmallLLM API")

    def test_health(self):
        resp = self.client.get('/api/health')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['status'], "healthy")

    def test_config_get(self):
        resp = self.client.get('/api/config')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['current_config'], "small")

    def test_config_post_updates_default(self):
        resp = self.client.post('/api/config', json={"config": "tiny"})
        self.assertEqual(resp.status_code, 200)

        # The default must actually change (regression: config_endpoint used to
        # mutate a local copy because the global statement was missing).
        self.assertEqual(api.MODEL_CONFIG, "tiny")
        self.assertEqual(self.client.get('/api/config').get_json()['current_config'], "tiny")

    def test_config_post_invalid(self):
        resp = self.client.post('/api/config', json={"config": "huge"})
        self.assertEqual(resp.status_code, 400)

    def test_generate_requires_prompt(self):
        resp = self.client.post('/api/generate', json={"max_tokens": 1})
        self.assertEqual(resp.status_code, 400)

    def test_generate_empty_body(self):
        # get_json(silent=True) must not blow up on a non-JSON body.
        resp = self.client.post('/api/generate', data="not json",
                                content_type="text/plain")
        self.assertEqual(resp.status_code, 400)

    def test_generate(self):
        resp = self.client.post('/api/generate', json={"prompt": "hello", "max_tokens": 2})
        self.assertEqual(resp.status_code, 200)
        self.assertIn('generated_text', resp.get_json())

    def test_explicit_config_does_not_change_default(self):
        # A per-request config must not leak into the server default.
        self.client.post('/api/generate', json={"prompt": "hi", "max_tokens": 1, "config": "tiny"})
        self.assertEqual(api.MODEL_CONFIG, "small")

    def test_tokenize_detokenize_roundtrip(self):
        resp = self.client.post('/api/tokenize', json={"text": "hello world"})
        self.assertEqual(resp.status_code, 200)
        tokens = resp.get_json()['tokens']

        resp = self.client.post('/api/detokenize', json={"token_ids": tokens})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['text'], "hello world")

    def test_models_list(self):
        resp = self.client.get('/api/models')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('tiny', resp.get_json()['models'])

    def test_byok_model_info(self):
        self.assertEqual(self.client.get('/v1/models/small').status_code, 200)
        self.assertEqual(self.client.get('/v1/models/nope').status_code, 404)

    def test_chat_accepts_openai_messages(self):
        resp = self.client.post('/api/chat', json={
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn('response', resp.get_json())

    def test_byok_chat_completions_shape(self):
        """The /v1 endpoint must return an OpenAI-shaped chat completion."""
        resp = self.client.post('/v1/chat/completions', json={
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
        })
        self.assertEqual(resp.status_code, 200)

        body = resp.get_json()
        self.assertEqual(body['object'], "chat.completion")
        choice = body['choices'][0]
        self.assertEqual(choice['message']['role'], "assistant")
        self.assertIn('content', choice['message'])
        self.assertIn('usage', body)

    def test_byok_chat_completions_requires_prompt(self):
        resp = self.client.post('/v1/chat/completions', json={"max_tokens": 1})
        self.assertEqual(resp.status_code, 400)
        self.assertIn('error', resp.get_json())

    def test_byok_completions_shape(self):
        resp = self.client.post('/v1/completions', json={"prompt": "hello", "max_tokens": 2})
        self.assertEqual(resp.status_code, 200)

        body = resp.get_json()
        self.assertEqual(body['object'], "text_completion")
        self.assertIn('text', body['choices'][0])

    def test_generate_accepts_list_prompt(self):
        """OpenAI also allows a list of prompts; the first one must be used."""
        resp = self.client.post('/api/generate', json={"prompt": ["hello"], "max_tokens": 2})
        self.assertEqual(resp.status_code, 200)
        self.assertIn('generated_text', resp.get_json())

    def test_byok_completions_accepts_list_prompt(self):
        resp = self.client.post('/v1/completions', json={"prompt": ["hello"], "max_tokens": 2})
        self.assertEqual(resp.status_code, 200)

    def test_chat_uses_last_user_message(self):
        """With several messages the last user turn is the prompt."""
        resp = self.client.post('/v1/chat/completions', json={
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "the real question"},
            ],
            "max_tokens": 2,
        })
        self.assertEqual(resp.status_code, 200)


if __name__ == '__main__':
    unittest.main()
