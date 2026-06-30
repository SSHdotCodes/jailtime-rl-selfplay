"""Torch-level test for the REINFORCE math (no tokenizer / no model download)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from jailtime.rl.torch_policy import compute_reinforce_step  # noqa: E402


def _tiny_model(vocab_size: int = 64) -> "torch.nn.Module":
    config = transformers.GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    model = transformers.GPT2LMHeadModel(config)
    model.to("cpu")
    return model


def test_reinforce_step_changes_weights_for_positive_advantage() -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    prompt_ids = [1, 2, 3]
    response_ids = [10, 11, 12]

    param = next(model.parameters())
    before = param.detach().clone()

    stats = compute_reinforce_step(
        model,
        optimizer,
        torch,
        prompt_token_ids=prompt_ids,
        response_token_ids=response_ids,
        advantage=1.0,
        entropy_coef=0.0,
        clip_grad=1.0,
        max_length=128,
    )

    after = next(model.parameters()).detach().clone()
    assert stats["num_tokens"] == 3.0
    assert stats["loss"] != 0.0
    assert not torch.allclose(before, after)


def test_reinforce_step_empty_response_is_noop() -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    param = next(model.parameters())
    before = param.detach().clone()

    stats = compute_reinforce_step(
        model,
        optimizer,
        torch,
        prompt_token_ids=[1, 2],
        response_token_ids=[],
        advantage=1.0,
        entropy_coef=0.0,
        clip_grad=1.0,
        max_length=128,
    )

    after = next(model.parameters()).detach().clone()
    assert stats["num_tokens"] == 0.0
    assert torch.allclose(before, after)


def test_reinforce_positive_advantage_increases_response_logprob() -> None:
    """A positive advantage should raise the log-prob of the sampled tokens."""

    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    prompt_ids = [1, 2, 3]
    response_ids = [10, 11, 12]

    def response_logprob() -> float:
        full = torch.tensor([prompt_ids + response_ids])
        logits = model(full).logits[:, :-1, :].to(torch.float32)
        targets = full[:, 1:]
        logp = torch.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return float(logp[0, len(prompt_ids) - 1 :].sum().detach().cpu())

    before = response_logprob()
    for _ in range(5):
        compute_reinforce_step(
            model,
            optimizer,
            torch,
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
            advantage=2.0,
            entropy_coef=0.0,
            clip_grad=1.0,
            max_length=128,
        )
    after = response_logprob()
    assert after > before
