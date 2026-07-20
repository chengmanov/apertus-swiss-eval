"""Minimal OpenAI-compatible chat client (llama.cpp llama-server, vLLM, ...)."""

import requests


class ChatClient:
    def __init__(self, base_url: str = "http://localhost:8080", model: str = "default",
                 temperature: float = 0.0, max_tokens: int = 256, timeout: int = 300):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def ask(self, system: str, user: str) -> str:
        resp = requests.post(self.url, json={
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
