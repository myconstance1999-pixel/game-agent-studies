"""Mock-only runtime boundary checks; no engine, HTTP, or model is exercised."""
from __future__ import annotations

from contextlib import ExitStack
import itertools
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import requests

import run as runtime
import protocol
from gamescaling.action import parse_action


def action(key):
    return protocol.canonical_history(((key,),) * 6)


class FakeGame:
    def __init__(self, terminal_tic=10, pixels=b"same initial pixels", kills=1):
        self.terminal_tic = terminal_tic
        self.pixels = pixels
        self.kills = kills
        self.tic = 0
        self.issued = []
        self.closed = False

    def init(self):
        pass

    def set_seed(self, seed):
        self.seed = seed

    def new_episode(self):
        self.tic = 0
        self.issued.clear()

    def get_state(self):
        return SimpleNamespace(screen_buffer=SimpleNamespace(tobytes=lambda: self.pixels))

    def is_episode_finished(self):
        return self.tic >= self.terminal_tic

    def get_episode_time(self):
        return self.tic

    def make_action(self, buttons, ticks):
        if self.is_episode_finished():
            raise AssertionError("advanced a terminated episode")
        if ticks != 1:
            raise AssertionError("runtime must check termination between individual tics")
        self.issued.append(tuple(buttons))
        self.tic += 1
        return 100.0

    def get_game_variable(self, variable):
        if variable != "KILLCOUNT":
            raise AssertionError("unexpected game-variable dependency")
        return 5 + (self.kills if self.is_episode_finished() else 0)

    def get_total_reward(self):
        return 1000.0  # Deliberately positive even when no kill occurs.

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, reply, fail=False):
        self.reply = reply
        self.fail = fail
        self.bodies = []
        self.closed = False

    def post(self, url, *, json, timeout):
        self.bodies.append(json)
        if self.fail:
            if isinstance(self.fail, BaseException):
                raise self.fail
            raise requests.ConnectionError("mock connection failure")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": self.reply}, "finish_reason": "stop"}]},
        )

    def close(self):
        self.closed = True


class RuntimeTests(unittest.TestCase):
    def run_mock_evaluation(self, root, *, arm="frozen", reply=None, games=None,
                            fail=False, compile_override=None):
        games = games or [FakeGame(), FakeGame()]
        session = FakeSession(reply or action("A"), fail=fail)
        settings = runtime.config()
        settings["max_decisions"] = 3
        args = SimpleNamespace(
            base_url="http://127.0.0.1:8000/v1", model="mock", out=str(root / "out"),
            checkpoint_label="MOCK_NOT_WEIGHTS", split="calibration", arm=arm,
            max_seconds=60,
        )
        vzd = SimpleNamespace(__version__="MOCK_NOT_ENGINE", GameVariable=SimpleNamespace(KILLCOUNT="KILLCOUNT"))
        error = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"vizdoom": vzd}))
            stack.enter_context(patch.object(runtime, "config", return_value=settings))
            stack.enter_context(patch.object(runtime, "cases", return_value=[{"case_id": "mock-1", "seed": 1, "split": "calibration"}]))
            stack.enter_context(patch.object(runtime, "make_game", side_effect=games))
            stack.enter_context(patch.object(requests, "Session", return_value=session))
            stack.enter_context(patch.object(Image, "fromarray", side_effect=lambda _: Image.new("RGB", (2, 2))))
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")))
            stack.enter_context(patch.object(socket, "getaddrinfo", side_effect=AssertionError("network forbidden")))
            if compile_override:
                stack.enter_context(patch.object(protocol, "compile_action", side_effect=compile_override))
            try:
                runtime.evaluate(args)
            except BaseException as exc:
                error = exc
        out = root / "out"
        receipt = json.loads((out / "receipt.json").read_text())
        traces = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
        episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
        return SimpleNamespace(error=error, receipt=receipt, traces=traces, episodes=episodes,
                               session=session, games=games)

    def test_terminal_mid_chunk_never_advances_extra_tics(self):
        compiled = protocol.compile_action(parse_action(action("A")), "mode-0")
        game = FakeGame(terminal_tic=3)
        ticks = runtime.execute_chunk(game, compiled)
        self.assertEqual(len(game.issued), 3)
        self.assertEqual([row["group"] for row in ticks], [0, 1, 2])
        self.assertEqual([(row["tic_before"], row["tic_after"]) for row in ticks], [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(runtime.execute_chunk(game, compiled), [])

    def test_history_uses_compiled_executed_d_not_proposed_a(self):
        compile_original = protocol.compile_action
        # Emulate a correction/normalization at the compiler seam. The actual
        # runtime must connect its result, rather than the raw reply, to history.
        def issue_d(proposal, mode):
            return compile_original(parse_action(action("D")), mode)
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_mock_evaluation(Path(tmp), compile_override=issue_d)
        self.assertIsNone(result.error)
        self.assertEqual(result.receipt["status"], "COMPLETE")
        for index in (1, 3):
            past = [m["content"] for m in result.session.bodies[index]["messages"] if m["role"] == "assistant"]
            self.assertEqual(past[0], action("D"))
            self.assertNotEqual(past[0], action("A"))
        self.assertEqual([len(row["executed_ticks"]) for row in result.traces], [7, 3, 7, 3])

    def test_key_swap_oracle_uses_consistent_virtual_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_mock_evaluation(Path(tmp), arm="key_swap")
        self.assertIsNone(result.error)
        self.assertEqual(result.games[0].issued, result.games[1].issued)
        self.assertTrue(all(buttons == (1, 0, 0) for buttons in result.games[1].issued))
        changed = next(row for row in result.traces if row["mode"] == "mode-1")
        self.assertEqual(changed["issued_action"], action("D"))
        self.assertEqual(changed["policy_history_action"], action("A"))
        request = result.session.bodies[3]["messages"]
        self.assertIn("Control context: mode-0.", request[0]["content"])
        self.assertEqual([m["content"] for m in request if m["role"] == "assistant"][0], action("A"))

    def test_connection_failure_remains_incomplete_without_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_mock_evaluation(Path(tmp), fail=True)
        self.assertIsInstance(result.error, requests.ConnectionError)
        self.assertEqual(result.receipt["status"], "INCOMPLETE")
        self.assertNotIn("scores", result.receipt)
        self.assertEqual(result.episodes, [])
        self.assertEqual(result.traces, [])
        self.assertTrue(result.games[0].closed)
        self.assertTrue(result.session.closed)

    def test_different_initial_pixels_prevent_paired_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_mock_evaluation(Path(tmp), games=[FakeGame(), FakeGame(pixels=b"different")])
        self.assertIsInstance(result.error, RuntimeError)
        self.assertIn("paired initial pixels differ", str(result.error))
        self.assertEqual(result.receipt["status"], "INCOMPLETE")
        self.assertNotIn("scores", result.receipt)
        self.assertEqual(len(result.episodes), 1)
        self.assertTrue(all(game.closed for game in result.games))

    def test_interruption_marks_receipt_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_mock_evaluation(Path(tmp), fail=KeyboardInterrupt())
        self.assertIsInstance(result.error, KeyboardInterrupt)
        self.assertEqual(result.receipt["status"], "INCOMPLETE")
        self.assertNotIn("scores", result.receipt)
        self.assertTrue(result.games[0].closed)
        self.assertTrue(result.session.closed)

    def test_success_is_killcount_delta_not_reward_or_existing_kills(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_mock_evaluation(Path(tmp), games=[FakeGame(kills=0), FakeGame(kills=1)])
        self.assertIsNone(result.error)
        self.assertEqual([row["success"] for row in result.episodes], [False, True])
        self.assertEqual([row["kills"] for row in result.episodes], [0, 1])
        self.assertEqual(result.receipt["scores"]["joint"]["rate"], 0.0)

    def test_frozen_seed_splits_are_nonempty_and_disjoint(self):
        splits = {name: runtime.cases(name) for name in runtime.config()["splits"]}
        for name, members in splits.items():
            self.assertTrue(members, name)
            self.assertEqual(len(members), len({case["seed"] for case in members}))
        for left, right in itertools.combinations(splits, 2):
            self.assertTrue({case["seed"] for case in splits[left]}.isdisjoint(case["seed"] for case in splits[right]), (left, right))

    def test_scripted_baselines_never_call_model(self):
        for arm in ("fire_only", "sweep_fire"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                result = self.run_mock_evaluation(Path(tmp), arm=arm, fail=True)
            self.assertIsNone(result.error)
            self.assertEqual(result.session.bodies, [])
            self.assertEqual(result.receipt["model_requests"], 0)
            self.assertTrue(all(buttons[2] == 1 for game in result.games for buttons in game.issued))
            if arm == "sweep_fire":
                self.assertEqual(result.games[0].issued[0], (1, 0, 1))
                self.assertEqual(result.games[1].issued[0], (0, 1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
