from __future__ import annotations

from typing import Iterable


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return 1.0 if any(a in relevant_ids for a in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    for i, aid in enumerate(ranked_ids, start=1):
        if aid in relevant_ids:
            return 1.0 / i
    return 0.0


def aggregate_metrics(records: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(records)
    if not rows:
        return {"Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "MRR": 0.0, "num_videos": 0}
    n = len(rows)
    return {
        "Recall@1": sum(r["Recall@1"] for r in rows) / n,
        "Recall@3": sum(r["Recall@3"] for r in rows) / n,
        "Recall@5": sum(r["Recall@5"] for r in rows) / n,
        "MRR": sum(r["MRR"] for r in rows) / n,
        "num_videos": n,
    }
