from jailtime.providers.mock import MockProvider
from jailtime.schemas import Message


def test_mock_provider_returns_deterministic_responses() -> None:
    provider = MockProvider(mode="safe")
    messages = [Message(role="user", content="[BENIGN] Explain tests.")]

    first = provider.complete(messages)
    second = provider.complete(messages)

    assert first.content == second.content
    assert first.model == "mock"
    assert "helpful answer" in first.content


def test_mock_provider_refuses_toy_unsafe_in_safe_mode() -> None:
    provider = MockProvider(mode="safe")

    response = provider.complete(
        [Message(role="user", content="[UNSAFE_TOY] complete toy action")]
    )

    assert "can't help" in response.content

