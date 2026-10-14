"""Fine-tune all BERT parameters; select checkpoint by held-out dev F1."""
import argparse
import collections
import functools
import json
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from .data import LABELS, LABEL2ID, read_jsonl, dump_json
from .engine import Decoder, encode_chars, collate
from .metrics import confusion, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--model", default="google-bert/bert-base-chinese")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", default="runs/main")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-chars", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if not 1 <= a.max_chars <= 510 or a.epochs < 1 or a.batch_size < 1:
        p.error("Invalid training dimensions")
    out = Path(a.output)
    if (out / "best" / "config.json").exists():
        raise FileExistsError(f"Existing completed checkpoint: {out}; choose a new --output")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForTokenClassification.from_pretrained(a.model, revision=a.revision, num_labels=len(LABELS),
              id2label=dict(enumerate(LABELS)), label2id=LABEL2ID).to(a.device)
    if a.max_chars + 2 > model.config.max_position_embeddings:
        raise ValueError("Window exceeds model positional embedding limit")
    train_docs = read_jsonl(Path(a.data) / "train.jsonl")
    dev = read_jsonl(Path(a.data) / "dev.jsonl")
    items = []
    counts = collections.Counter()
    for row in train_docs:
        ids = encode_chars(tokenizer, row["text"])
        counts.update(row["labels"])
        for start in range(0, len(ids), a.max_chars):
            items.append((ids[start:start+a.max_chars], row["labels"][start:start+a.max_chars]))
    loader = DataLoader(items, batch_size=a.batch_size, shuffle=True, num_workers=0,
                        collate_fn=functools.partial(collate, tokenizer=tokenizer), pin_memory=a.device.startswith("cuda"))
    # Mild inverse-frequency weighting, calculated from TRAIN ONLY and capped.
    freq = np.array([counts[i] for i in range(len(LABELS))], dtype=float)
    weights = np.sqrt(freq.sum() / np.maximum(freq, 1))
    weights = np.minimum(weights / weights[0], 8.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=a.device), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total_steps = a.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
    amp = a.device.startswith("cuda")
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    import transformers
    env = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "numpy": np.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name() if amp else "cpu",
           "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "model_commit": getattr(model.config, "_commit_hash", None),
           "parameters": sum(x.numel() for x in model.parameters()), "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
           "training_chunks": len(items), "optimizer_steps_planned": total_steps, "class_weights": weights.tolist(), "args": vars(a)}
    env["training_precision"] = str(amp_dtype) if amp else "torch.float32"
    source_manifest = Path(a.model) / "source_manifest.json"
    if source_manifest.exists():
        env["pretrained_source"] = json.loads(source_manifest.read_text())
    dump_json(out / "environment.json", env)
    tokenizer.save_pretrained(out / "best")
    best, step, updates, skipped, history, started = -1.0, 0, 0, 0, [], time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        total_loss, epoch_started = 0.0, time.perf_counter()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(a.device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda" if amp else "cpu", dtype=amp_dtype, enabled=amp):
                logits = model(**batch).logits
                loss = loss_fn(logits.float().reshape(-1, len(LABELS)), labels.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
                updates += 1
            else:
                skipped += 1
            total_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "total_steps": total_steps, "loss": round(loss.item(), 5),
                                  "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
        train_seconds = time.perf_counter() - epoch_started
        decoder = Decoder(model, tokenizer, a.device)
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        # Bounded group size keeps host memory proportional to a few articles.
        for i in range(0, len(dev), 64):
            group = dev[i:i+64]
            pred, _ = decoder.fixed([x["text"] for x in group], a.batch_size, a.max_chars, min(300, a.max_chars))
            for row, y in zip(group, pred):
                cm += confusion(row["labels"], y)
        metrics = summarize(cm)
        entry = {"epoch": epoch, "optimizer_step": updates, "skipped_overflow_updates": skipped, "mean_train_loss": total_loss / len(loader),
                 "train_seconds": train_seconds, "dev": metrics}
        history.append(entry)
        dump_json(out / "history.json", history)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if metrics["punctuation_micro_f1"] > best:
            best = metrics["punctuation_micro_f1"]
            model.save_pretrained(out / "best", safe_serialization=True)
            dump_json(out / "best_epoch.json", {"epoch": epoch, "dev_punctuation_micro_f1": best, "optimizer_step": updates})
    dump_json(out / "completion.json", {"status": "completed", "optimizer_steps": updates, "processed_batches": step,
             "skipped_overflow_updates": skipped, "wall_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
