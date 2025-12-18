"""Accumulates multiple OCR hits before trusting a plate value."""
from typing import Dict, Optional

from config import MIN_PLATE_HITS

# in-memory map of plate -> vote data
_plate_votes: Dict[str, Dict[str, float]] = {}


def _target_hits(required_hits: Optional[int]) -> int:
    if required_hits is None:
        return max(1, MIN_PLATE_HITS)
    return max(1, required_hits)


def register_plate_vote(plate: str, confidence: float, required_hits: Optional[int] = None) -> bool:
    """Record an OCR hit and return True once the threshold is met."""
    entry = _plate_votes.setdefault(plate, {"count": 0, "best_conf": 0.0})
    entry["count"] += 1
    entry["best_conf"] = max(entry["best_conf"], confidence)
    return entry["count"] >= _target_hits(required_hits)


def clear_plate_vote(plate: str) -> None:
    _plate_votes.pop(plate, None)
