"""Minimal chat/continuation client for native MLX checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from mlx_train import ASSISTANT, END, USER, Config, Model, decode, encode


def generate(model, cfg: Config, ids: list[int], count: int = 200) -> str:
    logits, cache = model(mx.array([ids[-cfg.ctx:]]), [None] * cfg.layers)
    out = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        if token == END:
            break
        out.append(token)
        logits, cache = model(mx.array([[token]]), cache)
    return decode(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--prompt")
    args = ap.parse_args()
    data = json.loads((args.run / "config.json").read_text())
    cfg = Config(**{k: data[k] for k in Config.__dataclass_fields__})
    model = Model(cfg)
    model.load_weights(str(args.run / "model.safetensors"))
    model.eval()

    def answer(text: str) -> str:
        return generate(model, cfg, [END, USER] + encode(text) + [ASSISTANT])

    if args.prompt:
        print(answer(args.prompt))
        return
    print(f"MLX nanochat {args.run} (byte-level model)")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.strip():
            print(answer(text))


if __name__ == "__main__":
    main()
