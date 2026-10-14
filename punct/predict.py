import argparse
from pathlib import Path
import torch
from .data import to_example, restore
from .engine import Decoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/main/best")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    torch.set_num_threads(8)
    source = a.text if a.text is not None else a.input.read_text(encoding="utf-8")
    # Preserve paragraph breaks; run windows across each paragraph and batch them.
    lines = source.splitlines() or [source]
    texts = [to_example(line)[0] for line in lines]
    decoder = Decoder.load(a.model, a.device)
    ys, _ = decoder.fixed(texts, a.batch_size)
    result = "\n".join(restore(t, y.tolist()) for t, y in zip(texts, ys))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
