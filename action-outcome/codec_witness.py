"""Exact mathematical action-codec witnesses; no game, model, or empirical rates.

A raw step changes the button at the current cursor, then applies displacement.
Down edges select that cursor location. The optional drag model contacts each
visited raw-step endpoint while held. Fraction arithmetic avoids rounding.
"""

from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import time


def aggregate_codec(trace):
    """Encode total commanded displacement and the complete button-slot trace."""
    return {"delta": [sum(s[0] for s in trace), sum(s[1] for s in trace)],
            "buttons": [s[2] for s in trace]}


def edge_codec(trace):
    """Encode cumulative command displacement at button edges plus final total.

These are relative positions under additive motion, not measured cursor state.
They are equivalent to displacement sums over constant-button intervals.
"""
    pos, down, events = [0, 0], False, []
    for i, (dx, dy, button) in enumerate(trace):
        if button != down:
            events.append({"slot": i, "down": button, "offset": pos[:]})
            down = button
        pos = [pos[0] + dx, pos[1] + dy]
    return {"slots": len(trace), "edges": events, "delta": pos}


def decode_aggregate(code):
    """Declared decoder: equal displacement after every slot's button event."""
    n = len(code["buttons"])
    dx, dy = (Fraction(v, n) for v in code["delta"])
    return [(dx, dy, b) for b in code["buttons"]]


def decode_edges(code):
    """Place each interval's displacement in its last slot, before the next edge."""
    trace = [[0, 0, False] for _ in range(code["slots"])]
    for event in code["edges"]:
        for step in trace[event["slot"]:]:
            step[2] = event["down"]
    previous = [0, 0]
    boundaries = [(e["slot"], e["offset"]) for e in code["edges"]]
    boundaries.append((code["slots"], code["delta"]))
    for boundary, offset in boundaries:
        if boundary:
            trace[boundary - 1][:2] = [offset[i] - previous[i] for i in (0, 1)]
        else:
            assert offset == [0, 0]
        previous = offset
    return trace


def execute(trace, start=(0, 0), *, bounds=None, drag=False):
    """Run the declared abstract transition system; bounds clip after each move."""
    cursor = [Fraction(x) for x in start]
    down, edges, clicks, contacts = False, [], [], set()
    for i, (dx, dy, button) in enumerate(trace):
        if type(button) is not bool:
            raise ValueError("Button values must be actual booleans")
        if button != down:
            edges.append({"slot": i, "down": button, "cursor": cursor[:]})
            if button:
                clicks.append(cursor[:])
            down = button
        if drag and down:
            contacts.add(tuple(cursor))
        cursor = [cursor[0] + dx, cursor[1] + dy]
        if bounds is not None:
            cursor = [min(max(v, lo), hi) for v, (lo, hi) in zip(cursor, bounds)]
        if drag and down:
            contacts.add(tuple(cursor))
    return {"final_cursor": cursor, "button_edges": edges,
            "click_positions": clicks, "drag_contacts": sorted(contacts)}


def trace_metrics(reference, candidate, **world):
    """Exact endpoint and event comparisons, with no empirical success estimate."""
    a, b = execute(reference, **world), execute(candidate, **world)
    return {"endpoint_l1": sum(abs(x-y) for x, y in zip(a["final_cursor"], b["final_cursor"])),
            "click_log_equal": a["click_positions"] == b["click_positions"],
            "button_edges_equal": a["button_edges"] == b["button_edges"],
            "drag_contacts_equal": a["drag_contacts"] == b["drag_contacts"]}


def describe(trace, **world):
    return {"trace": trace, "aggregate": aggregate_codec(trace),
            "button_edges": edge_codec(trace), "outcome": execute(trace, **world)}


def witnesses():
    """Construct seven explicit witnesses and an exhaustive conditional identity check."""
    buttons = [False, False, True, False, False, False]
    early = [(int(i == 0), 0, b) for i, b in enumerate(buttons)]
    late = [(int(i == 4), 0, b) for i, b in enumerate(buttons)]
    a, b = describe(early), describe(late)
    assert a["aggregate"] == b["aggregate"]
    assert a["outcome"]["final_cursor"] == b["outcome"]["final_cursor"]
    assert a["outcome"]["click_positions"] != b["outcome"]["click_positions"]
    collision = {"left": a, "right": b, "asserted": ["aggregate equal", "endpoint equal", "click positions unequal"]}

    # Complete bounded domain, not independently sampled trials or game episodes.
    for dxs in product((-1, 0, 1), repeat=6):
        trace = [(dx, 0, button) for dx, button in zip(dxs, buttons)]
        assert execute(trace) == execute(decode_edges(edge_codec(trace)))
    decoded = decode_edges(edge_codec(early))
    isolated = {"reference": a, "decoded_trace": decoded,
                "metrics": trace_metrics(early, decoded),
                "exhaustive_domain": {"dx_alphabet": [-1, 0, 1], "dy": 0,
                    "slots": 6, "buttons": buttons, "cases": 3**6},
                "assumptions": ["additive unclipped motion", "only button-edge interactions",
                    "no hover, dwell, drag, collision, or timing-dependent effects"],
                "asserted": ["all enumerated edge positions, click logs and endpoints preserved"]}

    tail = [(0, 0, False)] * 4
    horizontal = [(1, 0, True), (0, 1, True)] + tail
    vertical = [(0, 1, True), (1, 0, True)] + tail
    a, b = describe(horizontal, drag=True), describe(vertical, drag=True)
    assert a["button_edges"] == b["button_edges"]
    assert a["outcome"]["button_edges"] == b["outcome"]["button_edges"]
    assert a["outcome"]["drag_contacts"] != b["outcome"]["drag_contacts"]
    drag = {"left": a, "right": b, "assumptions": ["held drag contacts raw-step endpoints"],
            "asserted": ["edge codec equal", "actual edge positions equal", "drag contacts unequal"]}

    negative_first = [(-1, 0, False), (1, 0, False)] + tail
    positive_first = [(1, 0, False), (-1, 0, False)] + tail
    bounds = [(0, 1), (0, 1)]
    a, b = describe(negative_first, bounds=bounds), describe(positive_first, bounds=bounds)
    assert a["aggregate"] == b["aggregate"] and a["button_edges"] == b["button_edges"]
    assert a["outcome"]["final_cursor"] == [1, 0]
    assert b["outcome"]["final_cursor"] == [0, 0]
    clipping = {"bounds": bounds, "left": a, "right": b,
                "assumptions": ["codec sums commands; it does not store measured clipped endpoints"],
                "asserted": ["both codecs equal", "actual endpoints unequal"]}

    quiet_a = [(dx, dy, False) for dx, dy, _ in early]
    quiet_b = [(dx, dy, False) for dx, dy, _ in late]
    uniform = decode_aggregate(aggregate_codec(quiet_a))
    assert execute(quiet_a) == execute(quiet_b) == execute(uniform)
    harmless = {"left": describe(quiet_a), "right": describe(quiet_b),
                "uniform_replay": describe(uniform),
                "asserted": ["same endpoint-only outcome despite different motion timing"]}

    move_packet = aggregate_codec(quiet_a)
    click_packet = {"delta": [0, 0], "buttons": buttons}
    failed = decode_aggregate(aggregate_codec(early))
    split = decode_aggregate(move_packet) + decode_aggregate(click_packet)
    repair = failed + decode_aggregate(click_packet)
    def goal(trace):
        outcome = execute(trace)
        return outcome["click_positions"][-1] == [1, 0] and outcome["final_cursor"] == [1, 0]
    assert not goal(failed) and goal(early) and goal(split) and goal(repair)
    recovery = {"goal": "last selection at (1,0), final cursor (1,0)",
                "failed_aggregate_replay": describe(failed),
                "preplanned_split_packets": [move_packet, click_packet],
                "split_outcome": execute(split), "repair_outcome": execute(repair),
                "repair_packet": click_packet, "packet_duration_ms": 200,
                "original_packets": 1, "split_packets": 2, "repair_total_packets": 2,
                "native_duration_ms": {"original": 200, "split": 400, "repair": 400},
                "assumptions": ["selection is overwritten by a later click; no irreversible penalty",
                    "split/repair chosen by an oracle, not a learned agent", "no 200-ms goal deadline"],
                "asserted": ["aggregate replay misses goal", "split and later repair reach goal",
                    "repair retains the original wrong click in its log"]}

    reference = [(0, 0, button) for button in buttons]
    candidate = early
    state_cases = []
    for start in ((0, 0), (1, 0)):
        ref_out, cand_out = execute(reference, start=start), execute(candidate, start=start)
        ref_slots = ["left" if x < 2 else "right" for x, _ in ref_out["click_positions"]]
        cand_slots = ["left" if x < 2 else "right" for x, _ in cand_out["click_positions"]]
        action_error = sum(abs(a[i] - b[i]) for a, b in zip(reference, candidate) for i in (0, 1))
        assert [a[2] for a in reference] == [b[2] for b in candidate]
        state_cases.append({"initial_cursor": start, "reference_outcome": ref_out,
                            "candidate_outcome": cand_out, "reference_slots": ref_slots,
                            "candidate_slots": cand_slots, "action_displacement_l1": action_error,
                            "slot_outcome_distortion": int(ref_slots != cand_slots)})
    assert [c["action_displacement_l1"] for c in state_cases] == [1, 1]
    assert [c["reference_slots"] for c in state_cases] == [["left"], ["left"]]
    assert [c["slot_outcome_distortion"] for c in state_cases] == [0, 1]
    state_conditioned = {
        "reference_trace": reference, "candidate_trace": candidate,
        "reference_encodings": {"aggregate": aggregate_codec(reference), "edges": edge_codec(reference)},
        "candidate_encodings": {"aggregate": aggregate_codec(candidate), "edges": edge_codec(candidate)},
        "slot_partition": {"left": "x < 2", "right": "x >= 2", "y": "ignored"},
        "outcome_projection": "clicked slot identity only; within-slot cursor differences are irrelevant",
        "click_semantics": "button edge precedes same-slot movement; candidate moves in slot 0, clicks in slot 2",
        "cases": state_cases,
        "assumptions": ["unclipped additive motion", "same layout and identical action pair in both states",
                        "one isolated click; button states match exactly"],
        "asserted": ["identical action-only L1 error", "reference selects left slot in both states",
                     "candidate distortion is 0 inside the slot and 1 across its boundary"],
        "limit": "A declared state-conditioned metric distinction, not novelty or empirical model evidence."}
    return {"aggregate_click_collision": collision, "isolated_edge_preservation": isolated,
            "held_drag_counterexample": drag, "clipping_counterexample": clipping,
            "harmless_no_event": harmless, "split_and_repair": recovery,
            "state_conditioned_outcome_distortion": state_conditioned}


def exact_json(value):
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else str(value)
    if isinstance(value, dict):
        return {k: exact_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact_json(v) for v in value]
    return value


if __name__ == "__main__":
    started = time.perf_counter()
    result = {"kind": "deterministic mathematical witnesses, not a model/game benchmark",
              "transition_order": "button edge at current cursor, then mouse movement",
              "fraction_encoding": "nonintegral rationals are exact numerator/denominator strings",
              "default_initial_cursor": [0, 0], "initial_button_down": False,
              "witnesses": witnesses(), "all_assertions_passed": True}
    source = Path(__file__).resolve()
    result["receipt"] = {"command": [sys.executable, str(source)],
                         "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "runtime_seconds": time.perf_counter() - started}
    output = source.parent / "results" / "witness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(exact_json(result), indent=2) + "\n")
    print(f"PASS: 7 mathematical cases; 729 exhaustive conditional identities. Saved {output}")
