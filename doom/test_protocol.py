"""Offline protocol tests; synthetic scoring fixtures are not gameplay results."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

from protocol import (
    BUTTON_ORDER,
    EMPTY_STEPS,
    MODES,
    TICK_SCHEDULE,
    TICKS_PER_SECOND,
    canonical_history,
    compile_action,
    inverse_keys,
    score_pairs,
)

# Load only the pinned standalone parser, not the package or any inference client.
PINNED_ACTION = (
    Path(__file__).resolve().parents[1]
    / "third_party/gamescaling/action.py"
)
SPEC = importlib.util.spec_from_file_location("pinned_gamescaling_action", PINNED_ACTION)
assert SPEC is not None and SPEC.loader is not None
UPSTREAM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = UPSTREAM
SPEC.loader.exec_module(UPSTREAM)


def parsed_body(body: str):
    return UPSTREAM.parse_action(f"<|action_start|>{body}<|action_end|>", lenient=False)


class ProtocolTests(unittest.TestCase):
    def assert_rejected(self, action, reason=None):
        result = compile_action(action, "mode-0")
        self.assertFalse(result.accepted)
        self.assertTrue(result.rejection_reason)
        if reason:
            self.assertEqual(result.rejection_reason, reason)
        self.assertEqual(result.logical_steps, EMPTY_STEPS)
        self.assertEqual(result.button_steps, ((0, 0, 0),) * 6)
        self.assertEqual(result.history_text, canonical_history(EMPTY_STEPS))
        self.assertEqual(sum(result.tick_schedule), 7)

    def test_button_order_and_exact_tick_conservation(self):
        action = parsed_body("0 0 0 ; A ; D ; LB ; A D ; D LB ; A D LB")
        result = compile_action(action, "mode-0")
        self.assertTrue(result.accepted)
        self.assertEqual(BUTTON_ORDER, ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"))
        self.assertEqual(result.button_steps, (
            (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 1, 1),
        ))
        self.assertEqual(result.tick_schedule, (1, 1, 2, 1, 1, 1))
        self.assertEqual(sum(TICK_SCHEDULE) * 1000, 200 * TICKS_PER_SECOND)

    def test_swap_symmetry_for_all_eight_allowed_key_subsets(self):
        for bits in itertools.product((False, True), repeat=3):
            keys = tuple(k for k, active in zip(("A", "D", "LB"), bits) if active)
            with self.subTest(keys=keys):
                self.assertEqual(inverse_keys(inverse_keys(keys)), keys)
                self.assertEqual("LB" in inverse_keys(keys), "LB" in keys)
                normal = compile_action(UPSTREAM.parse_action(canonical_history((keys,) * 6)), "mode-0")
                swapped = compile_action(UPSTREAM.parse_action(canonical_history((inverse_keys(keys),) * 6)), "mode-1")
                self.assertEqual(normal.button_steps, swapped.button_steps)

    def test_history_is_canonical_logical_action_not_raw_or_physical(self):
        action = parsed_body("0.0 -0 0 ; LB A A D ; D A ; ; LB ; D D ; A")
        result = compile_action(action, "mode-1")
        self.assertTrue(result.accepted)
        self.assertEqual(result.logical_steps[0], ("A", "D", "LB"))
        self.assertNotEqual(result.history_text, action.history_text)
        replay = UPSTREAM.parse_action(result.history_text)
        self.assertEqual(tuple(tuple(g) for g in replay.steps), result.logical_steps)
        self.assertEqual(result.logical_steps[-1], ("A",))
        self.assertEqual(result.button_steps[-1], (0, 1, 0))
        normal = compile_action(action, "mode-0")
        self.assertEqual(result.history_text, normal.history_text)

    def test_aliases_and_unsupported_keys_reject_entire_chunk(self):
        for key in ("LMB", "a", "ArrowLeft", "Left", "W", "RB"):
            with self.subTest(key=key):
                self.assert_rejected(parsed_body(f"0 0 0 ; A ; {key} ; ; ; ;"))

    def test_mouse_scroll_and_nonfinite_source_values_rejected(self):
        for head in ("1 0 0", "0 -1 0", "0 0 1", "1e999 0 0", "nan 0 0"):
            with self.subTest(head=head):
                self.assert_rejected(parsed_body(f"{head} ; A ; ; ; ; ;"))
        self.assert_rejected(parsed_body("nan 0 0 ; A ; ; ; ; ;"), "nonfinite_mouse_or_scroll")

    def test_parser_silent_padding_and_truncation_are_rejected(self):
        for body in (
            "0 0 0 ; A",                    # Upstream pads missing groups.
            "0 0 0 ; A ; ; ; ; ; ; D",      # Upstream drops the seventh group.
            "0 0 0 1 ; A ; ; ; ; ;",        # Upstream ignores the fourth number.
            "0 0 0 ; A A A A D ; ; ; ; ;",  # Upstream drops the fifth key.
        ):
            with self.subTest(body=body):
                action = parsed_body(body)
                self.assertTrue(action.parsed)
                self.assert_rejected(action)

    def test_malformed_and_truncated_reply_rejected(self):
        self.assert_rejected(UPSTREAM.parse_action("not an action"), "unparsed_action")
        self.assert_rejected(UPSTREAM.parse_action("<think>unfinished"), "truncated_action")
        self.assert_rejected(parsed_body("0 0 ; A ; ; ; ; ;"), "unparsed_action")

    def test_effective_action_must_match_body(self):
        action = parsed_body("0 0 0 ; A ; ; ; ; ;")
        action.steps[0] = ["D"]
        self.assert_rejected(action, "body_effective_mismatch")

    def test_invalid_mode_and_history_raise(self):
        with self.assertRaises(ValueError):
            compile_action(parsed_body("0 0 0 ; ; ; ; ; ;"), "unknown")
        with self.assertRaises(ValueError):
            canonical_history(((),))
        with self.assertRaises(ValueError):
            inverse_keys(("LMB",))


class PairedScoringTests(unittest.TestCase):
    def test_per_mode_and_joint_are_distinct(self):
        records = [
            {"case_id": "left", "mode": "mode-0", "success": True},
            {"case_id": "left", "mode": "mode-1", "success": False},
            {"case_id": "right", "mode": "mode-0", "success": False},
            {"case_id": "right", "mode": "mode-1", "success": True},
        ]
        score = score_pairs(records)
        self.assertEqual(score["n_pairs"], 2)
        self.assertEqual(score["per_mode"]["mode-0"]["rate"], 0.5)
        self.assertEqual(score["per_mode"]["mode-1"]["rate"], 0.5)
        self.assertEqual(score["joint"]["rate"], 0.0)
        self.assertNotIn("retention", score)

    def test_synthetic_always_action_shortcut_is_not_gameplay(self):
        # Synthetic desired-button labels vary by state; no engine was run.
        records = []
        for case_id, desired in (("left", (1, 0, 0)), ("right", (0, 1, 0))):
            for mode in MODES:
                shortcut = compile_action(parsed_body("0 0 0 ; A ; A ; A ; A ; A ; A"), mode)
                records.append({"case_id": case_id, "mode": mode, "success": shortcut.button_steps[0] == desired})
        score = score_pairs(records)
        self.assertEqual([score["per_mode"][m]["rate"] for m in MODES], [0.5, 0.5])
        self.assertEqual(score["joint"]["rate"], 0.0)

    def test_invalid_missing_or_duplicate_pairs_fail(self):
        first = {"case_id": "case", "mode": "mode-0", "success": True}
        second = {"case_id": "case", "mode": "mode-1", "success": False}
        invalid = [
            [], [first], [first, second, first],
            [first, dict(second, success=1)],
            [first, dict(second, success="false")],
            [first, dict(second, mode="unknown")],
            [first, dict(second, case_id="")],
            [{"case_id": "case", "mode": "mode-0"}, second],
        ]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                score_pairs(records)


if __name__ == "__main__":
    unittest.main(verbosity=2)
