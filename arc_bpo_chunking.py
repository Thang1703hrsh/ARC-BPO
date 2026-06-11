import re
from bisect import bisect_right
from typing import Iterable, List, Sequence, Tuple, Union


Span = Tuple[int, int]


_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"')\]]*(?:\s+|$)")
_LINE_BREAK_RE = re.compile(r"\n+")
_SOFT_BREAK_RE = re.compile(r"(?:;\s+|:\s+)")


def _response_text(response: Union[str, dict, Sequence[dict]]) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response["content"]
    if len(response) != 1:
        raise ValueError("ARC-BPO chunking expects one assistant response at a time.")
    return response[0]["content"]


def _dedupe_sorted_boundaries(boundaries: Iterable[int], text_len: int) -> List[int]:
    clipped = {0, text_len}
    for boundary in boundaries:
        clipped.add(min(max(int(boundary), 0), text_len))
    return sorted(clipped)


def semantic_char_boundaries(text: str) -> List[int]:
    """Return deterministic semantic character boundaries that cover the text.

    The splitter prefers paragraph, line, and sentence boundaries. It keeps
    delimiters and following whitespace inside the preceding chunk so spans stay
    contiguous and preserve the original response exactly.
    """
    if not text:
        return [0]

    boundaries = [0, len(text)]

    for match in _PARAGRAPH_BREAK_RE.finditer(text):
        boundaries.append(match.end())

    for match in _LINE_BREAK_RE.finditer(text):
        boundaries.append(match.end())

    for match in _SENTENCE_END_RE.finditer(text):
        boundaries.append(match.end())

    for match in _SOFT_BREAK_RE.finditer(text):
        boundaries.append(match.end())

    return _dedupe_sorted_boundaries(boundaries, len(text))


def _offset_boundaries(text: str, tokenizer, char_boundaries: Sequence[int]) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    input_ids = encoded["input_ids"]
    if not offsets:
        return _prefix_boundaries(text, tokenizer, char_boundaries)

    token_ends = [end for _, end in offsets]
    token_boundaries = []
    for boundary in char_boundaries:
        token_boundaries.append(bisect_right(token_ends, boundary))

    token_boundaries[0] = 0
    token_boundaries[-1] = len(input_ids)
    return token_boundaries


def _prefix_boundaries(text: str, tokenizer, char_boundaries: Sequence[int]) -> List[int]:
    token_boundaries = []
    for boundary in char_boundaries:
        prefix = text[:boundary]
        token_boundaries.append(len(tokenizer(prefix, add_special_tokens=False)["input_ids"]))
    return token_boundaries


def _split_long_spans(spans: Sequence[Span], max_tokens_per_chunk: int | None) -> List[Span]:
    if max_tokens_per_chunk is None or max_tokens_per_chunk <= 0:
        return list(spans)

    split_spans: List[Span] = []
    for start, end in spans:
        cursor = start
        while cursor < end:
            next_end = min(cursor + max_tokens_per_chunk, end)
            split_spans.append((cursor, next_end))
            cursor = next_end
    return split_spans


def _merge_short_spans(spans: Sequence[Span], min_tokens_per_chunk: int | None) -> List[Span]:
    """Merge sub-floor fragments forward so every chunk has >= min tokens.

    Spans must already be contiguous. The last span may absorb the trailing
    short fragment; if the final span is still short it is merged backward so no
    chunk falls below the floor (unless the whole response is shorter).
    """
    if min_tokens_per_chunk is None or min_tokens_per_chunk <= 1:
        return list(spans)
    if not spans:
        return list(spans)

    merged: List[Span] = []
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if (cur_end - cur_start) < min_tokens_per_chunk:
            cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    # If the final chunk is still below the floor, fold it into its predecessor.
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_tokens_per_chunk:
        last_start, last_end = merged.pop()
        prev_start, _ = merged.pop()
        merged.append((prev_start, last_end))

    return merged


def retarget_chunk_spans_to_length(chunk_spans: Sequence[Span], n_tokens: int) -> List[Span]:
    """Retarget chunk spans to a new contiguous token length.

    Chat templates can change tokenizer boundaries at the prompt/response join:
    tokenizing raw assistant text may produce a slightly different token count
    than the same assistant content inside the full conversation. ARC-BPO sums
    chunks over shifted response log-probs, so spans must cover that exact
    shifted length.
    """
    if n_tokens <= 0:
        return []
    if not chunk_spans:
        return [(0, n_tokens)]

    source_total = int(chunk_spans[-1][1])
    if source_total <= 0:
        return [(0, n_tokens)]
    if source_total == n_tokens:
        return [(int(start), int(end)) for start, end in chunk_spans]

    boundaries = [0]
    boundaries.extend(
        round(float(end) * float(n_tokens) / float(source_total))
        for _, end in chunk_spans
    )
    boundaries[0] = 0
    boundaries[-1] = n_tokens

    clipped = [min(max(int(boundary), 0), n_tokens) for boundary in boundaries]
    retargeted: List[Span] = []
    previous = 0
    for boundary in clipped[1:]:
        boundary = max(previous, boundary)
        if boundary > previous:
            retargeted.append((previous, boundary))
        previous = boundary

    if not retargeted:
        return [(0, n_tokens)]
    retargeted[0] = (0, retargeted[0][1])
    retargeted[-1] = (retargeted[-1][0], n_tokens)
    return retargeted


def chunk_response_tokens(
    response: Union[str, dict, Sequence[dict]],
    tokenizer,
    max_tokens_per_chunk: int | None = None,
    min_tokens_per_chunk: int | None = None,
) -> List[Span]:
    """Chunk one response into tokenizer-aligned token spans.

    Returned spans are [start, end) over response-content tokens only. The
    function is deterministic and depends only on the response text and
    tokenizer, never on policy parameters.
    """
    text = _response_text(response)
    token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if token_count == 0:
        return []

    char_boundaries = semantic_char_boundaries(text)
    try:
        token_boundaries = _offset_boundaries(text, tokenizer, char_boundaries)
    except NotImplementedError:
        token_boundaries = _prefix_boundaries(text, tokenizer, char_boundaries)

    token_boundaries = _dedupe_sorted_boundaries(token_boundaries, token_count)
    spans = [
        (start, end)
        for start, end in zip(token_boundaries, token_boundaries[1:])
        if end > start
    ]
    spans = _split_long_spans(spans, max_tokens_per_chunk)
    spans = _merge_short_spans(spans, min_tokens_per_chunk)

    if not spans:
        return [(0, token_count)]

    # Make coverage explicit even when a tokenizer reports unusual offsets.
    spans[0] = (0, spans[0][1])
    spans[-1] = (spans[-1][0], token_count)
    return spans


def chunk_preference_pair(
    chosen: Union[str, dict, Sequence[dict]],
    rejected: Union[str, dict, Sequence[dict]],
    tokenizer,
    max_tokens_per_chunk: int | None = None,
    min_tokens_per_chunk: int | None = None,
) -> dict:
    return {
        "chosen_chunk_spans": chunk_response_tokens(
            chosen,
            tokenizer,
            max_tokens_per_chunk=max_tokens_per_chunk,
            min_tokens_per_chunk=min_tokens_per_chunk,
        ),
        "rejected_chunk_spans": chunk_response_tokens(
            rejected,
            tokenizer,
            max_tokens_per_chunk=max_tokens_per_chunk,
            min_tokens_per_chunk=min_tokens_per_chunk,
        ),
    }
