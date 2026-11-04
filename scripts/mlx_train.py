"""Small, self-contained MLX trainer for Apple Silicon."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

BYTE_VOCAB = 259
END, USER, ASSISTANT = 256, 257, 258


@dataclass
class Config:
    d: int
    layers: int
    heads: int
    kv_heads: int
    ctx: int
    batch: int
    lr: float
    minutes: float
    tokens: int

    @property
    def head_dim(self) -> int:
        return self.d // self.heads

    @property
    def d_ff(self) -> int:
        return -(-int(8 * self.d / 3) // 64) * 64


PRESETS = {
    "smoke": Config(128, 2, 4, 2, 128, 16, 2e-3, 3, 2_000_000),
    "hour": Config(256, 8, 4, 2, 512, 32, 1e-3, 60, 60_000_000),
    "night": Config(512, 12, 8, 4, 1024, 4, 6e-4, 480, 150_000_000),
}


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.d, cfg.d, bias=False)
        self.k = nn.Linear(cfg.d, cfg.kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d, cfg.kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, base=10000)

    def __call__(self, x, cache=None):
        b, t, _ = x.shape
        q = self.q(x).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(b, t, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(b, t, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache[0].shape[2]
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5,
            mask="causal" if t > 1 else None,
        )
        return self.o(y.transpose(0, 2, 1, 3).reshape(b, t, -1)), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d)
        self.gate = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d, bias=False)

    def __call__(self, x, cache=None):
        h, cache = self.attn(self.norm1(x), cache)
        x = x + h
        normed = self.norm2(x)
        return x + self.down(nn.silu(self.gate(normed)) * self.up(normed)), cache


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(BYTE_VOCAB, cfg.d)
        self.blocks = [Block(cfg) for _ in range(cfg.layers)]
        self.norm = nn.RMSNorm(cfg.d)

    def __call__(self, ids, caches=None):
        h = self.embed(ids)
        new = []
        for i, block in enumerate(self.blocks):
            h, cache = block(h, None if caches is None else caches[i])
            new.append(cache)
        logits = self.embed.as_linear(self.norm(h))
        return logits if caches is None else (logits, new)


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


def decode(ids: list[int]) -> str:
    return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


def load_tokens(path: Path, target: int) -> np.ndarray:
    if path.exists():
        return np.load(path, mmap_mode="r")
    from datasets import load_dataset

    print(f"Downloading TinyStories until {target:,} tokens are available...")
    chunks, count = [], 0
    for row in load_dataset("roneneldan/TinyStories", split="train", streaming=True):
        chunk = np.asarray(encode(row["text"]) + [END], dtype=np.uint16)
        chunks.append(chunk)
        count += len(chunk)
        if count >= target:
            break
        if len(chunks) % 1000 == 0:
            print(f"  {count:,}/{target:,} tokens")
    tokens = np.concatenate(chunks)[:target]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tokens)
    return tokens


def batches(tokens: np.ndarray, cfg: Config, rng: np.random.Generator):
    starts = rng.integers(0, len(tokens) - cfg.ctx - 1, size=cfg.batch)
    rows = np.stack([tokens[s : s + cfg.ctx + 1] for s in starts]).astype(np.int32)
    return mx.array(rows[:, :-1]), mx.array(rows[:, 1:])


def loss_fn(model, x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))


def sample(model, ids: list[int], cfg: Config, count: int = 160) -> str:
    ids = ids[-cfg.ctx:]
    logits, cache = model(mx.array([ids]), [None] * cfg.layers)
    out = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        if token == END:
            break
        out.append(token)
        logits, cache = model(mx.array([[token]]), cache)
    return decode(out)


def get_base_dir() -> Path:
    """Return the project cache directory used for MLX experiment artifacts."""
    return Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train nanochat with MLX on Apple Silicon")
    ap.add_argument("--preset", choices=PRESETS, default="hour")
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data", type=Path)
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    if args.minutes is not None:
        cfg.minutes = args.minutes
    base_dir = get_base_dir()
    out = args.out or base_dir / "mlx_checkpoints" / args.preset
    data_path = args.data or base_dir / "mlx_data" / f"{args.preset}.npy"
    tokens = load_tokens(data_path, cfg.tokens)
    model = Model(cfg)
    opt = optim.AdamW(learning_rate=cfg.lr, betas=[0.9, 0.95], weight_decay=0.1)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(42)
    step, started = 0, time.time()
    deadline = started + cfg.minutes * 60
    max_steps = args.iterations or math.inf

    print(f"MLX nanochat: {cfg.d}d, {cfg.layers} layers, context {cfg.ctx}")
    print(f"data: {len(tokens):,} byte tokens | output: {out}")
    while time.time() < deadline and step < max_steps:
        x, y = batches(tokens, cfg, rng)
        loss, grads = loss_and_grad(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        step += 1
        if step == 1 or step % 50 == 0:
            print(f"step {step:6d} | loss {loss.item():.4f} | {(time.time() - started) / 60:.1f} min")

    out.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps({**asdict(cfg), "vocab": BYTE_VOCAB}, indent=2))
    (out / "summary.json").write_text(json.dumps({"steps": step, "minutes": (time.time() - started) / 60}, indent=2))
    print(f"saved {out / 'model.safetensors'}")
    print(f"sample: {sample(model, [END] + encode('Once upon a time'), cfg)}")


if __name__ == "__main__":
    main()
