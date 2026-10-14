"""Document-level accuracy and synchronized repeated batch benchmarks."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from .data import LABELS, read_jsonl, dump_json, restore
from .engine import Decoder
from .metrics import confusion, summarize


def sync(decoder):
    if decoder.device.type == "cuda":
        torch.cuda.synchronize(decoder.device)


def score(decoder, docs, method, batch, window, core):
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    totals = {"windows": 0, "forward_calls": 0}
    examples = []
    for i in range(0, len(docs), 64):
        group = docs[i:i+64]
        texts = [x["text"] for x in group]
        ys, stats = decoder.sequential(texts, window, core) if method == "sequential" else decoder.fixed(texts, batch, window, core)
        for key in totals:
            totals[key] += stats[key]
        for row, y in zip(group, ys):
            cm += confusion(row["labels"], y)
            if len(examples) < 5:
                examples.append({"id": row["id"], "input": row["text"][:450],
                                 "gold": restore(row["text"][:450], row["labels"][:450]),
                                 "prediction": restore(row["text"][:450], y[:450].tolist())})
    return {**summarize(cm), **totals, "articles": len(docs)}, examples


def run_bench(decoder, docs, batches, window=400, core=300, repeats=3):
    texts = [d["text"] for d in docs]
    total_chars = sum(map(len, texts))
    rows, reference = [], None
    configurations = [("sequential", 1)] + [("fixed", b) for b in batches]
    # Reverse ordering on alternate repetitions to reduce order/thermal bias.
    records = {key: [] for key in configurations}
    result_info = {}
    for repeat in range(repeats):
        order = configurations if repeat % 2 == 0 else list(reversed(configurations))
        for method, batch in order:
            fn = (lambda ts: decoder.sequential(ts, window, core)) if method == "sequential" else (lambda ts: decoder.fixed(ts, batch, window, core))
            try:
                fn(texts[:2])  # untimed warm-up for this shape/batch
                sync(decoder)
                if decoder.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(decoder.device)
                started = time.perf_counter()
                ys, info = fn(texts)
                sync(decoder)
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated(decoder.device) / 1024 ** 2 if decoder.device.type == "cuda" else None
                flat = np.concatenate(ys)
                digest = hashlib.sha256(flat.astype(np.int8).tobytes()).hexdigest()
                if method == "fixed" and batch == 1:
                    reference = flat.copy()
                result_info[(method, batch)] = {**info, "peak_allocated_mib": peak, "prediction_sha256": digest, "predictions": flat}
                records[(method, batch)].append(elapsed)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                result_info[(method, batch)] = {"status": "cuda_oom"}
    baseline_seconds = statistics.median(records[("sequential", 1)])
    for method, batch in configurations:
        runs, info = records[(method, batch)], result_info[(method, batch)]
        if len(runs) != repeats:
            rows.append({"method": method, "batch_size": batch, "status": "cuda_oom", "completed_repeats": len(runs)})
            continue
        predictions = info.pop("predictions")
        median = statistics.median(runs)
        rows.append({"method": method, "batch_size": batch, "status": "completed", "seconds_each": runs,
                     "median_seconds": median, "chars_per_second": total_chars / median,
                     "speedup_vs_sequential": baseline_seconds / median,
                     "label_agreement_with_fixed_b1": float(np.mean(predictions == reference)) if method == "fixed" and reference is not None else None,
                     **info})
    fixed1 = next((r for r in rows if r["method"] == "fixed" and r["batch_size"] == 1 and r["status"] == "completed"), None)
    for row in rows:
        if row["status"] == "completed":
            row["speedup_vs_fixed_b1"] = fixed1["median_seconds"] / row["median_seconds"] if fixed1 else None
    return {"articles": len(docs), "characters": total_chars, "article_ids": [d["id"] for d in docs],
            "window_chars": window, "core_chars": core, "repeats": repeats,
            "timing_scope": "end-to-end character encoding, batch creation, synchronized GPU inference, CPU label merge; model loading and file I/O excluded",
            "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/main/best")
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--output", default="runs/main/evaluation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window", type=int, default=400)
    p.add_argument("--core", type=int, default=300)
    p.add_argument("--benchmark-docs", type=int, default=40)
    p.add_argument("--batches", default="1,4,8,16,32,64")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--skip-sequential-score", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(8)
    decoder = Decoder.load(a.model, a.device)
    docs = read_jsonl(a.data)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    fixed, examples = score(decoder, docs, "fixed", a.batch_size, a.window, a.core)
    dump_json(out / "test_fixed.json", fixed)
    dump_json(out / "examples.json", examples)
    print("fixed", json.dumps(fixed, ensure_ascii=False), flush=True)
    if not a.skip_sequential_score:
        sequential, _ = score(decoder, docs, "sequential", 1, a.window, a.core)
        dump_json(out / "test_sequential.json", sequential)
        print("sequential", json.dumps(sequential, ensure_ascii=False), flush=True)
    # Benchmark on the longest held-out documents, avoiding mostly-short texts.
    bench_docs = sorted(docs, key=lambda x: (-len(x["text"]), x["id"]))[:a.benchmark_docs]
    batches = list(dict.fromkeys([1] + [int(x) for x in a.batches.split(",")]))
    bench = run_bench(decoder, bench_docs, batches, a.window, a.core, a.repeats)
    dump_json(out / "benchmark.json", bench)
    print("benchmark", json.dumps(bench, ensure_ascii=False), flush=True)
    dump_json(out / "completion.json", {"status": "completed", "test_articles": len(docs)})


if __name__ == "__main__":
    main()
