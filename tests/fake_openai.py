"""Shared fakes so the test suite never touches the network."""
import types
from typing import List

import pytest


class FakeCompletions:
    """Records how many API calls were made and returns a canned reply."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.batches: List[List[str]] = []

    def create(self, **kwargs):
        self.calls += 1
        # Capture the numbered batch input so tests can assert on batching.
        user = next((m["content"] for m in kwargs["messages"] if m["role"] == "user"), "")
        if "Segments:" in user:
            body = user.split("Segments:", 1)[1]
            body = body.split("Translations (one per line):", 1)[0]
            self.batches.append([line for line in body.strip().split("\n") if line.strip()])
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.reply))]
        )


class FakeClient:
    def __init__(self, reply):
        self.chat = types.SimpleNamespace(completions=FakeCompletions(reply))


@pytest.fixture
def fake_client_factory():
    """Return a callable that swaps in a fake OpenAI client."""

    def install(translator, reply):
        translator.client = FakeClient(reply)
        return translator.client

    return install
