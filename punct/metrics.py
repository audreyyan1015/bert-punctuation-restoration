"""Metrics computed on original document positions, once per character."""
import numpy as np
from .data import LABELS


def confusion(gold, pred):
    g, p = np.asarray(gold, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    if g.shape != p.shape:
        raise ValueError("Prediction length mismatch")
    return np.bincount(g * len(LABELS) + p, minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))


def summarize(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp, support, predicted = cm.diagonal(), cm.sum(1), cm.sum(0)
    rows = {}
    for i, label in enumerate(LABELS):
        precision = float(tp[i] / predicted[i]) if predicted[i] else 0.0
        recall = float(tp[i] / support[i]) if support[i] else 0.0
        rows[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                       "support": int(support[i]), "predicted": int(predicted[i])}
    correct = int(tp[1:].sum())
    precision = correct / int(predicted[1:].sum()) if predicted[1:].sum() else 0.0
    recall = correct / int(support[1:].sum()) if support[1:].sum() else 0.0
    terminal = [2, 3, 4]
    seg_tp = int(cm[np.ix_(terminal, terminal)].sum())
    seg_p = seg_tp / int(predicted[terminal].sum()) if predicted[terminal].sum() else 0.0
    seg_r = seg_tp / int(support[terminal].sum()) if support[terminal].sum() else 0.0
    return {"characters": int(cm.sum()), "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
            "punctuation_micro_precision": precision, "punctuation_micro_recall": recall,
            "punctuation_micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "punctuation_macro_f1": float(np.mean([rows[s]["f1"] for s in LABELS[1:]])),
            "sentence_boundary_f1": 2 * seg_p * seg_r / (seg_p + seg_r) if seg_p + seg_r else 0.0,
            "per_label": rows, "confusion_matrix": cm.tolist()}
