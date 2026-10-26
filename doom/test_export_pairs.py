"""Temporary synthetic trace fixtures only; dummy frame bytes are not images."""
import json
from pathlib import Path
import tempfile
import unittest

from export_pairs import PROVENANCE, export_pairs
from protocol import EMPTY_STEPS, TICK_SCHEDULE, canonical_history, compile_action
from gamescaling.action import parse_action


class ExportPairsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "pairs.jsonl"
        self.receipt = {"status": "COMPLETE", "split": "train", "arm": "frozen", "config": {"history_len": 5}}
        self.episodes = [{"case_id": "train-2000", "split": "train", "mode": "mode-0", "success": True}]
        self.steps = []

    def step(self, index, key="A", *, accepted=True, count=7, case_id="train-2000"):
        action = canonical_history(((key,),) * 6 if accepted else EMPTY_STEPS)
        compiled = compile_action(parse_action(action), "mode-0")
        ticks = [{"group": group, "buttons": list(buttons)}
                 for group, (buttons, duration) in enumerate(zip(compiled.button_steps, TICK_SCHEDULE))
                 for _ in range(duration)][:count]
        frame = f"{case_id}-mode-0-{index:03d}.png"
        (self.root / frame).write_bytes(b"SYNTHETIC TEST FIXTURE; NOT AN IMAGE")
        row = {"case_id": case_id, "split": "train", "mode": "mode-0", "step": index,
               "frame": frame, "issued_action": action, "policy_history_action": action,
               "accepted": accepted, "executed_ticks": ticks}
        self.steps.append(row)
        return row

    def write_fixture(self):
        (self.root / "receipt.json").write_text(json.dumps(self.receipt))
        for name, rows in (("episodes", self.episodes), ("steps", self.steps)):
            (self.root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def export(self):
        self.write_fixture()
        export_pairs(self.root, self.output)
        return [json.loads(line) for line in self.output.read_text().splitlines()]

    def test_targets_and_all_five_prior_actions_are_swapped(self):
        for index, key in enumerate(("A", "LB", "D", "A", "D", "LB", "A")):
            self.step(index, key, accepted=index != 3)
        samples = self.export()
        self.assertEqual(len(samples), 12)  # The rejected step is history, not a target.
        normal, swapped = samples[-2:]
        self.assertEqual((normal["step"], swapped["step"]), (6, 6))
        self.assertEqual(normal["frame"], swapped["frame"])
        self.assertTrue(Path(normal["frame"]).is_absolute())
        self.assertEqual(normal["provenance"], PROVENANCE)
        self.assertEqual(len(normal["history"]), 5)
        self.assertEqual([Path(h["frame"]).name for h in normal["history"]],
                         [self.steps[i]["frame"] for i in range(1, 6)])
        self.assertEqual(normal["history"][2]["action"], canonical_history(EMPTY_STEPS))
        self.assertEqual(swapped["history"][2]["action"], canonical_history(EMPTY_STEPS))
        self.assertEqual(swapped["history"][1]["action"], canonical_history((("A",),) * 6))
        self.assertEqual(swapped["history"][0]["action"], canonical_history((("LB",),) * 6))
        self.assertEqual(normal["target_action"], canonical_history((("A",),) * 6))
        self.assertEqual(swapped["target_action"], canonical_history((("D",),) * 6))
        self.assertNotIn("reasoning", normal)

    def test_failed_trajectory_and_terminal_partial_target_are_excluded(self):
        self.step(0)
        self.step(1, count=2)
        self.episodes.append({"case_id": "train-2001", "split": "train", "mode": "mode-0", "success": False})
        self.step(0, "LB", case_id="train-2001")
        samples = self.export()
        self.assertEqual([(row["case_id"], row["step"]) for row in samples], [("train-2000", 0)] * 2)

    def test_wrong_receipt_split_arm_and_status_fail(self):
        self.step(0)
        for field, value in (("split", "dev"), ("split", "test"), ("split", "calibration"),
                             ("arm", "key_swap"), ("status", "INCOMPLETE")):
            original = self.receipt[field]
            self.receipt[field] = value
            self.write_fixture()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                export_pairs(self.root, self.output)
            self.assertFalse(self.output.exists())
            self.receipt[field] = original

    def test_nontrain_record_duplicate_or_missing_decision_fail(self):
        first = self.step(0)
        original = list(self.steps)
        for rows in ([first, first], [dict(first, step=1)], [dict(first, split="test")],
                     [dict(first, case_id="dev-3000")]):
            self.steps = rows
            self.write_fixture()
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                export_pairs(self.root, self.output)
        self.steps = original
        self.episodes.append(dict(self.episodes[0]))
        self.write_fixture()
        with self.assertRaises(ValueError):
            export_pairs(self.root, self.output)

    def test_missing_frame_fails_without_output(self):
        row = self.step(0)
        (self.root / row["frame"]).unlink()
        self.write_fixture()
        with self.assertRaises(ValueError):
            export_pairs(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_early_partial_execution_cannot_be_removed_from_history(self):
        self.step(0, count=2)
        self.step(1)
        self.write_fixture()
        with self.assertRaises(ValueError):
            export_pairs(self.root, self.output)

    def test_executed_buttons_must_match_target(self):
        row = self.step(0)
        row["executed_ticks"][0]["buttons"] = [0, 1, 0]
        self.write_fixture()
        with self.assertRaises(ValueError):
            export_pairs(self.root, self.output)

    def test_existing_output_is_preserved(self):
        self.step(0)
        self.write_fixture()
        self.output.write_text("existing evidence\n")
        with self.assertRaises(FileExistsError):
            export_pairs(self.root, self.output)
        self.assertEqual(self.output.read_text(), "existing evidence\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
