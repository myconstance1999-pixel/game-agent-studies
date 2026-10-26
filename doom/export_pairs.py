"""Export counterfactual supervised pairs from successful normal train rollouts.

These are frozen-policy success trajectories, not expert-optimal labels or
corrections at failure states. This module performs no gameplay or training.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import tempfile

from protocol import MODES, TICK_SCHEDULE, canonical_history, compile_action, inverse_keys

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party"
sys.path.insert(0, str(UPSTREAM))
from gamescaling.action import parse_action

PROVENANCE = "counterfactual_relabeling_of_successful_normal_rollout"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _identity(row: dict) -> tuple[str, str]:
    case_id, mode = row.get("case_id"), row.get("mode")
    if row.get("split") != "train" or not isinstance(case_id, str) or not case_id.startswith("train-"):
        raise ValueError("Every episode and step must identify a train case")
    if mode not in MODES:
        raise ValueError("Unknown or missing mode")
    return case_id, mode


def _frame(root: Path, name: str) -> str:
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing frame or frame outside run directory: {name!r}")
    return str(path)


def _action(text: str, *, swap: bool = False):
    compiled = compile_action(parse_action(text, lenient=False), "mode-0")
    if not compiled.accepted or compiled.history_text != text:
        raise ValueError("issued_action must be a canonical supported action")
    if swap:
        return canonical_history(tuple(inverse_keys(group) for group in compiled.logical_steps))
    return compiled


def export_pairs(run_dir: str | Path, output: str | Path) -> int:
    """Atomically create JSONL pairs; raise on invalid traces or existing output.

    Require a COMPLETE/train/frozen receipt and five-action history contract.
    Each accepted, fully executed decision in a successful mode-0 episode yields
    two rows with identical frame paths. Mode-1 inverts both target and preceding
    five issued actions, including rejected full-chunk no-ops. Partial terminal
    decisions are omitted as targets; a partial decision followed by another
    decision is rejected because the actual prompt timeline cannot be recovered.
    """
    root, destination = Path(run_dir).resolve(), Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    receipt = json.loads((root / "receipt.json").read_text())
    if (receipt.get("status"), receipt.get("split"), receipt.get("arm")) != ("COMPLETE", "train", "frozen"):
        raise ValueError("Only COMPLETE train frozen runs may be exported")
    if receipt.get("config", {}).get("history_len") != 5:
        raise ValueError("Expected the declared five-action history contract")
    episodes = {}
    for row in _rows(root / "episodes.jsonl"):
        key = _identity(row)
        if key in episodes or type(row.get("success")) is not bool:
            raise ValueError("Duplicate episode or non-boolean success")
        episodes[key] = row
    traces = {key: {} for key in episodes}
    for row in _rows(root / "steps.jsonl"):
        key, step = _identity(row), row.get("step")
        if key not in episodes or type(step) is not int or step < 0 or step in traces[key]:
            raise ValueError("Unknown episode, invalid step, or duplicate decision")
        traces[key][step] = row
    samples = []
    for key, episode in episodes.items():
        if key[1] != "mode-0" or not episode["success"]:
            continue
        steps = traces[key]
        if sorted(steps) != list(range(len(steps))):
            raise ValueError("Missing decision would erase executed history")
        history = deque(maxlen=5)
        for index in range(len(steps)):
            row = steps[index]
            issued = row["issued_action"]
            compiled = _action(issued)
            if row.get("policy_history_action") != issued or type(row.get("accepted")) is not bool:
                raise ValueError("Invalid frozen-policy history or acceptance flag")
            if not row["accepted"] and any(compiled.logical_steps):
                raise ValueError("A rejected chunk must have issued a no-op")
            ticks = row["executed_ticks"]
            expected = [(group, list(buttons)) for group, (buttons, count) in enumerate(
                zip(compiled.button_steps, TICK_SCHEDULE)) for _ in range(count)]
            actual = [(tick["group"], tick["buttons"]) for tick in ticks]
            if len(ticks) > 7 or actual != expected[:len(ticks)]:
                raise ValueError("Executed ticks do not match the issued action schedule")
            if len(ticks) != 7:
                if index != len(steps) - 1:
                    raise ValueError("Partial execution before a later decision")
                continue
            frame = _frame(root, row["frame"])
            if row["accepted"]:
                for mode in MODES:
                    swap = mode == "mode-1"
                    samples.append({
                        "case_id": key[0], "step": index, "split": "train", "mode": mode,
                        "frame": frame,
                        "history": [{"frame": old_frame, "action": _action(action, swap=True) if swap else action}
                                    for old_frame, action in history],
                        "target_action": _action(issued, swap=True) if swap else issued,
                        "provenance": PROVENANCE,
                    })
            # Rejected full-chunk no-ops remain in the actual prompt timeline.
            history.append((frame, issued))
    if not samples:
        raise ValueError("No eligible successful normal-mode train decisions")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            for sample in samples:
                handle.write(json.dumps(sample) + "\n")
        # Same-directory hard link publishes the complete file without replacing
        # a destination created concurrently. The temporary name is then removed.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink()
    return len(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("output")
    arguments = parser.parse_args()
    print(json.dumps({"samples": export_pairs(arguments.run_dir, arguments.output), "output": str(Path(arguments.output).resolve())}))
