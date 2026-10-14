"""BERT input alignment and batched, bounded-memory sliding-window decoding."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import unicodedata
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def encode_chars(tokenizer, text: str) -> list[int]:
    # One source character -> one token, also for digits/Latin/unknown characters.
    # Normalization affects model input only; restore() keeps source characters.
    tokens = []
    for c in text:
        normalized = unicodedata.normalize("NFKC", c).lower()
        tokens.append(normalized if len(normalized) == 1 else c)
    return tokenizer.convert_tokens_to_ids(tokens)


def collate(items, tokenizer):
    n = max(len(x[0]) for x in items) + 2
    ids = torch.full((len(items), n), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (x, y) in enumerate(items):
        if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
            raise ValueError("This exercise requires BERT CLS/SEP tokens")
        wrapped = [tokenizer.cls_token_id, *x, tokenizer.sep_token_id]
        ids[i, :len(wrapped)] = torch.tensor(wrapped)
        mask[i, :len(wrapped)] = 1
        if y is not None:
            labels[i, 1:1 + len(y)] = torch.tensor(y)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


@dataclass(frozen=True)
class Window:
    doc: int
    left: int
    right: int
    core_left: int
    core_right: int


def windows(length: int, doc: int = 0, window: int = 400, core: int = 300):
    if not 0 < core <= window <= 510:
        raise ValueError("Require 0 < core <= window <= 510")
    context = (window - core) // 2
    for a in range(0, length, core):
        b = min(a + core, length)
        yield Window(doc, max(0, a - context), min(length, b + window - core - context), a, b)


class Decoder:
    def __init__(self, model, tokenizer, device="cuda", amp=True):
        self.model, self.tokenizer = model.to(device).eval(), tokenizer
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")

    @classmethod
    def load(cls, path, device=None, amp=True):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(AutoModelForTokenClassification.from_pretrained(path), AutoTokenizer.from_pretrained(path), device, amp)

    @torch.inference_mode()
    def predict_batch(self, ids):
        batch = collate([(x, None) for x in ids], self.tokenizer)
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            pred = self.model(**inputs).logits.argmax(-1).cpu().numpy()
        return [p[1:1 + len(x)].copy() for p, x in zip(pred, ids)]

    def fixed(self, texts, batch_size=16, window=400, core=300):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        encoded = [encode_chars(self.tokenizer, t) for t in texts]
        outputs = [np.full(len(t), -1, dtype=np.int64) for t in texts]
        iterator = (w for i, t in enumerate(texts) for w in windows(len(t), i, window, core))
        forward_calls, window_count = 0, 0
        while group := list(islice(iterator, batch_size)):
            prediction = self.predict_batch([encoded[w.doc][w.left:w.right] for w in group])
            forward_calls += 1
            window_count += len(group)
            for w, y in zip(group, prediction):
                outputs[w.doc][w.core_left:w.core_right] = y[w.core_left - w.left:w.core_right - w.left]
        if any(np.any(x < 0) for x in outputs):
            raise RuntimeError("Uncovered original character")
        return outputs, {"windows": window_count, "forward_calls": forward_calls}

    def sequential(self, texts, window=400, core=300):
        if not 0 < core <= window <= 510:
            raise ValueError("Require 0 < core <= window <= 510")
        outputs, calls = [], 0
        for text in texts:
            encoded = encode_chars(self.tokenizer, text)
            out, start = [], 0
            while start < len(text):
                ids = encoded[start:start + window]
                y = self.predict_batch([ids])[0]
                calls += 1
                if start + len(ids) == len(text):
                    take = len(ids)
                else:
                    terminal = np.flatnonzero(np.isin(y[:core], [2, 3, 4]))
                    # Do not allow pathologically tiny strides.
                    terminal = terminal[terminal >= core // 2 - 1]
                    take = int(terminal[-1] + 1) if len(terminal) else core
                out.extend(y[:take].tolist())
                start += take
            outputs.append(np.asarray(out, dtype=np.int64))
        return outputs, {"windows": calls, "forward_calls": calls}
