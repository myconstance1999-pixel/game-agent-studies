"""Pure action and scoring contracts for the paused ViZDoom basic/map01 pilot.

The upstream parser remains responsible for text parsing. This module checks its
effective Action and body for unsupported or silently discarded content. It does
not import a game engine, model, parser, or network client.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

MODES = ("mode-0", "mode-1")
LOGICAL_KEYS = ("A", "D", "LB")
BUTTON_ORDER = ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK")
TICKS_PER_SECOND = 35
TICK_SCHEDULE = (1, 1, 2, 1, 1, 1)
EMPTY_STEPS: tuple[tuple[str, ...], ...] = ((),) * 6


class ParsedAction(Protocol):
    """Fields consumed from the pinned upstream ``gamescaling.action.Action``."""

    mouse: tuple[float, float, float]
    steps: list[list[str]]
    parsed: bool
    truncated: bool
    body: str
    dropped: list[str]


def _canonical_keys(keys: Iterable[str]) -> tuple[str, ...]:
    tokens = tuple(keys)
    if any(key not in LOGICAL_KEYS for key in tokens):
        raise ValueError("Only exact logical keys A, D, and LB are supported")
    return tuple(key for key in LOGICAL_KEYS if key in tokens)


def inverse_keys(keys: Iterable[str]) -> tuple[str, ...]:
    """Oracle inverse-key baseline: swap A/D, preserve LB, and canonicalize.

    This known-control transformation uses no teacher or game outcome. Applying
    it twice returns the canonical original key set, including simultaneous A/D.
    """
    inverse = {"A": "D", "D": "A", "LB": "LB"}
    return _canonical_keys(inverse[key] for key in _canonical_keys(keys))


def canonical_history(steps: Sequence[Sequence[str]]) -> str:
    """Serialize exactly six logical-key groups, deduplicated in A/D/LB order.

    History records the issued logical controls, never remapped engine buttons.
    Unsupported keys or a different number of groups raise ``ValueError``.
    """
    if len(steps) != 6:
        raise ValueError("History requires exactly six key groups")
    groups = " ; ".join(" ".join(_canonical_keys(group)) for group in steps)
    return f"<|action_start|>0 0 0 ; {groups}<|action_end|>"


@dataclass(frozen=True)
class CompiledAction:
    """One accepted chunk or declared full-chunk no-op after rejection."""

    accepted: bool
    rejection_reason: str | None
    logical_steps: tuple[tuple[str, ...], ...]
    button_steps: tuple[tuple[int, int, int], ...]
    history_text: str
    tick_schedule: tuple[int, ...] = TICK_SCHEDULE


def _rejection_reason(action: ParsedAction) -> str | None:
    if action.truncated:
        return "truncated_action"
    if not action.parsed:
        return "unparsed_action"
    if action.dropped:
        return "dropped_tokens"
    if not isinstance(action.body, str):
        return "malformed_body"
    parts = action.body.split(";")
    if len(parts) != 7 or len(parts[0].split()) != 3:
        return "malformed_body"
    try:
        source_mouse = tuple(float(token) for token in parts[0].split())
    except ValueError:
        return "malformed_body"
    if any(not math.isfinite(value) for value in source_mouse):
        return "nonfinite_mouse_or_scroll"
    if any(value != 0 for value in source_mouse):
        return "unsupported_mouse_or_scroll"
    if len(action.mouse) != 3:
        return "malformed_mouse"
    if any(not math.isfinite(value) for value in action.mouse):
        return "nonfinite_mouse_or_scroll"
    if any(value != 0 for value in action.mouse):
        return "unsupported_mouse_or_scroll"
    if len(action.steps) != 6:
        return "malformed_steps"
    for source, effective in zip(parts[1:], action.steps):
        tokens = source.split()
        if any(key not in LOGICAL_KEYS for key in tokens):
            return "unsupported_keys"
        # The upstream parser silently truncates each group to four tokens.
        if len(tokens) > 4:
            return "excess_keys"
        if tokens != list(effective):
            return "body_effective_mismatch"
    return None


def compile_action(action: ParsedAction, mode: str) -> CompiledAction:
    """Validate a parsed Action and produce six engine vectors plus tic counts.

    Call the pinned upstream ``parse_action(text, lenient=False)`` first. Any
    rejected chunk becomes six empty logical groups and six zero button vectors
    for all seven tics (200 ms at 35 Hz), with an explicit rejection reason.
    ``mode-0`` maps A/D normally; ``mode-1`` swaps their engine mapping. Both
    modes preserve the same logical-key history. Unknown modes raise ValueError.
    """
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode!r}")
    reason = _rejection_reason(action)
    steps = EMPTY_STEPS if reason else tuple(_canonical_keys(g) for g in action.steps)
    physical = steps if mode == "mode-0" else tuple(inverse_keys(g) for g in steps)
    buttons = tuple(tuple(int(key in group) for key in LOGICAL_KEYS) for group in physical)
    return CompiledAction(
        accepted=reason is None,
        rejection_reason=reason,
        logical_steps=steps,
        button_steps=buttons,
        history_text=canonical_history(steps),
    )


def score_pairs(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Score strict case_id/mode/success records with both modes for every case.

    ``case_id`` must be a nonempty string; ``success`` must be an actual bool.
    Duplicate case/mode records, missing modes, unknown modes, and empty input
    raise ValueError. Returned rates describe these records only: without a
    before-update baseline they do not establish retention or forgetting.
    """
    pairs: dict[str, dict[str, bool]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each record must be a mapping")
        case_id, mode, success = (record.get(key) for key in ("case_id", "mode", "success"))
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a nonempty string")
        if mode not in MODES:
            raise ValueError(f"Unsupported or missing mode: {mode!r}")
        if type(success) is not bool:
            raise ValueError("success must be a bool, not an integer or string")
        pair = pairs.setdefault(case_id, {})
        if mode in pair:
            raise ValueError(f"Duplicate case/mode: {case_id!r}, {mode!r}")
        pair[mode] = success
    if not pairs:
        raise ValueError("At least one complete pair is required")
    for case_id, pair in pairs.items():
        if set(pair) != set(MODES):
            raise ValueError(f"Missing mode for case: {case_id!r}")
    count = len(pairs)
    successes = {mode: sum(pair[mode] for pair in pairs.values()) for mode in MODES}
    joint = sum(all(pair.values()) for pair in pairs.values())
    return {
        "n_pairs": count,
        "per_mode": {
            mode: {"successes": successes[mode], "count": count, "rate": successes[mode] / count}
            for mode in MODES
        },
        "joint": {"successes": joint, "count": count, "rate": joint / count},
    }
