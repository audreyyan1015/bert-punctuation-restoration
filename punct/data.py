"""Corpus parsing and the single authoritative character/label conversion."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

LABELS = ["NONE", "，", "。", "？", "！", "；", "：", "、"]
LABEL2ID = {s: i for i, s in enumerate(LABELS)}
PUNCT = {s: s for s in LABELS[1:]}
PUNCT.update({",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："})
PRIORITY = {"NONE": 0, "、": 1, "，": 2, "：": 3, "；": 4, "。": 5, "？": 6, "！": 7}
LINE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s+(.*)$")
PRONUNCIATION = re.compile(r"\{[^{}]*\}")


def strip_annotations(text: str) -> str:
    return "".join(PRONUNCIATION.sub("", text).split())


def to_example(text: str) -> tuple[str, list[int]]:
    """Predict a supported punctuation mark AFTER each remaining character.

    Other Unicode punctuation (quotes/brackets/ellipsis/dashes) is removed and
    is outside the label vocabulary. A cluster keeps the highest-priority
    supported mark; an ignored closing quote cannot erase a period.
    """
    chars, labels = [], []
    for ch in strip_annotations(text):
        if ch in PUNCT:
            if labels:
                new = PUNCT[ch]
                if PRIORITY[new] >= PRIORITY[LABELS[labels[-1]]]:
                    labels[-1] = LABEL2ID[new]
        elif unicodedata.category(ch).startswith("P"):
            continue
        else:
            chars.append(ch)
            labels.append(0)
    return "".join(chars), labels


def restore(text: str, labels: list[int]) -> str:
    if len(text) != len(labels):
        raise ValueError("Each character needs exactly one predicted label")
    return "".join(ch + (LABELS[y] if y else "") for ch, y in zip(text, labels))


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def dump_json(path: str | Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    docs, files, malformed, annotation_count = collections.OrderedDict(), [], 0, 0
    invalid_articles = set()
    paths = sorted(source.rglob("*.txt"))
    if not paths:
        raise ValueError(f"No txt corpus files under {source}")
    for path in paths:
        raw = path.read_bytes()
        try:
            content, encoding = raw.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError:
            content, encoding = raw.decode("gb18030", errors="replace"), "gb18030"
        files.append({"name": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "encoding": encoding,
                      "replacement_characters": content.count("\ufffd")})
        annotation_count += len(PRONUNCIATION.findall(content))
        for line in content.splitlines():
            if not line.strip():
                continue
            m = LINE.match(line.strip())
            if not m:
                malformed += 1
                continue
            if "\ufffd" in m[2]:
                invalid_articles.add(m[1])
                continue
            docs.setdefault(m[1], []).append(m[2])
    splits = {"train": [], "dev": [], "test": []}
    seen, duplicate, short = set(), 0, 0
    for doc_id, paragraphs in docs.items():
        if doc_id in invalid_articles:
            continue
        text, labels = to_example("".join(paragraphs))
        if len(text) < 32 or not any(labels):
            short += 1
            continue
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        month = int(doc_id[4:6])
        split = "train" if month <= 4 else "dev" if month == 5 else "test" if month == 6 else None
        if split is None:
            raise ValueError(f"Unexpected month: {doc_id}")
        splits[split].append({"id": doc_id, "text": text, "labels": labels, "text_sha256": key})
    stats = {"source_files": files, "label_order": LABELS, "raw_articles": len(docs), "removed_duplicate_articles": duplicate,
             "removed_short_or_unlabelled_articles": short, "malformed_nonempty_lines": malformed,
             "removed_articles_with_decoding_errors": len(invalid_articles),
             "removed_pronunciation_annotations": annotation_count, "split_policy": "Jan-Apr train; May dev; June test; exact unpunctuated article dedup across all splits", "splits": {}}
    for split, records in splits.items():
        if not records:
            raise ValueError(f"Empty {split}; supply the full January-June corpus")
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts = collections.Counter(y for r in records for y in r["labels"])
        stats["splits"][split] = {"articles": len(records), "characters": sum(len(r["text"]) for r in records),
                                   "max_article_chars": max(len(r["text"]) for r in records),
                                   "labels": {s: counts[i] for i, s in enumerate(LABELS)}}
    dump_json(output / "stats.json", stats)
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="Extracted January-June corpus directory")
    p.add_argument("--output", type=Path, default=Path("data"))
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output), ensure_ascii=False, indent=2))
