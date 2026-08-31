"""Score extraction helpers for ARC-BPO preference datasets.

This module deliberately has no dependency on ``datasets`` or ``torch`` so
the dataset-schema logic can be unit-tested without a training environment.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple


def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _response_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        return content if isinstance(content, str) else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        # Some preference datasets store a response as a one-message chat.
        for item in reversed(value):
            text = _response_text(item)
            if text is not None:
                return text
    return None


def _direct_score(example: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        score = _finite_float(example.get(key))
        if score is not None:
            return score
    return None


def _matching_scores(
    target_text: str,
    responses: Sequence[Any],
    scores: Sequence[Any],
) -> list[float]:
    matches = []
    for response, raw_score in zip(responses, scores):
        if _response_text(response) != target_text:
            continue
        score = _finite_float(raw_score)
        if score is not None:
            matches.append(score)
    return matches


def extract_preference_scores(
    example: Mapping[str, Any],
    chosen_text: str,
    rejected_text: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Return response-level chosen/rejected scores when the schema provides them.

    Direct pair-score columns are preferred.  Princeton's
    ``llama3-ultrafeedback-armorm`` instead stores every sampled response and
    its ArmoRM score in aligned ``all_generated_responses`` and
    ``all_rm_scores`` arrays; in that schema we recover the scores by exact
    response-text matching.  We intentionally do not guess with global
    max/min values when the text cannot be matched, because doing so could
    silently attach a score to the wrong response.
    """

    chosen_score = _direct_score(
        example,
        ("score_chosen", "chosen_score", "chosen_rm_score"),
    )
    rejected_score = _direct_score(
        example,
        ("score_rejected", "rejected_score", "rejected_rm_score"),
    )
    if chosen_score is not None and rejected_score is not None:
        return chosen_score, rejected_score

    responses = example.get("all_generated_responses")
    scores = example.get("all_rm_scores")
    if not isinstance(responses, Sequence) or isinstance(responses, (str, bytes)):
        return chosen_score, rejected_score
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        return chosen_score, rejected_score
    if len(responses) != len(scores):
        return chosen_score, rejected_score

    if chosen_score is None:
        chosen_matches = _matching_scores(chosen_text, responses, scores)
        if chosen_matches:
            chosen_score = max(chosen_matches)
    if rejected_score is None:
        rejected_matches = _matching_scores(rejected_text, responses, scores)
        if rejected_matches:
            rejected_score = min(rejected_matches)
    return chosen_score, rejected_score
