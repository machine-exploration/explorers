"""Toy settings where the answer is known, trained in seconds on a CPU.

`multitask_lookup` is a quanta-style task. There are `n_tasks` tasks; task t maps a symbol x to
f_t(x), a fixed random permutation. Training draws tasks with Zipf frequencies p_t ~ t^-alpha,
so each task is one "quantum" and the quanta hypothesis predicts frequent tasks are learned first.
"""

from dataclasses import dataclass

import numpy as np

from explorers.core.data import Examples
from explorers.core.state import Step, Trajectory, snapshot


@dataclass(frozen=True)
class MultitaskLookup:
    n_tasks: int = 16
    n_symbols: int = 16
    alpha: float = 1.3
    seed: int = 0

    @property
    def vocab_size(self) -> int:
        return self.n_tasks + self.n_symbols

    @property
    def maps(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return np.stack([rng.permutation(self.n_symbols) for _ in range(self.n_tasks)])

    @property
    def frequencies(self) -> np.ndarray:
        p = (np.arange(1, self.n_tasks + 1)) ** -self.alpha
        return p / p.sum()

    def _rows(self, tasks: np.ndarray, xs: np.ndarray) -> np.ndarray:
        s0 = self.n_tasks
        return np.stack([tasks, s0 + xs, s0 + self.maps[tasks, xs]], axis=1).astype(np.int64)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        tasks = rng.choice(self.n_tasks, size=n, p=self.frequencies)
        return self._rows(tasks, rng.integers(0, self.n_symbols, size=n))

    def examples(self) -> Examples:
        """Every (task, symbol) pair once; only the answer position counts towards the loss."""
        tasks, xs = np.meshgrid(np.arange(self.n_tasks), np.arange(self.n_symbols), indexing="ij")
        tasks, xs = tasks.ravel(), xs.ravel()
        tokens = self._rows(tasks, xs)
        mask = np.zeros_like(tokens, dtype=bool)
        mask[:, 2] = True
        return Examples(tokens=tokens, loss_mask=mask, name=f"multitask_lookup_{self.n_tasks}x{self.n_symbols}",
                        meta={"task": tasks, "task_frequency": self.frequencies[tasks]})


def tiny_gpt(vocab_size: int, d: int = 64, layers: int = 2, heads: int = 4, ctx: int = 8, seed: int = 0):
    import torch
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=vocab_size, hidden_size=d, num_hidden_layers=layers,
                        num_attention_heads=heads, intermediate_size=4 * d, max_position_embeddings=ctx)
    return GPTNeoXForCausalLM(cfg)


def train(task: MultitaskLookup, steps: int = 1500, every: int = 50, batch_size: int = 64,
          lr: float = 3e-3, seed: int = 0, run: str | None = None, model=None) -> Trajectory:
    """Train a tiny transformer on `task`, keeping a State every `every` steps and a Step
    (with its gradient) at each of those points. Loss counts only the answer token."""
    import torch

    run = run or f"toy:{task.n_tasks}x{task.n_symbols}:a{task.alpha}:s{seed}"
    model = model or tiny_gpt(task.vocab_size, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 1)
    states = [snapshot(model, run, 0)]
    step_records = []
    for t in range(1, steps + 1):
        batch = torch.as_tensor(task.sample(batch_size, rng))
        logits = model(input_ids=batch).logits
        loss = torch.nn.functional.cross_entropy(logits[:, 1], batch[:, 2])
        opt.zero_grad()
        loss.backward()
        keep = t % every == 0
        if keep:
            grad = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
        opt.step()
        if keep:
            states.append(snapshot(model, run, t))
            step_records.append(Step(run=run, step=t, before=states[-2], after=states[-1], grad=grad))
    return Trajectory(run=run, states=states, steps=step_records,
                      coords={"alpha": task.alpha, "seed": seed, "lr": lr})
