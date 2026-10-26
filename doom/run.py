"""Prepared ViZDoom basic evaluator. Engine/model execution requires separate authorization.

The default command only prints the frozen case plan. No install, model download,
server startup, or network request occurs at import or during `plan`/`score`.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "third_party"
sys.path.insert(0, str(UPSTREAM))


def config():
    return json.loads((HERE / "experiment.json").read_text())


def cases(split):
    spec = config()["splits"][split]
    return [{"case_id": f"{split}-{seed}", "seed": seed, "split": split}
            for seed in range(spec["seed_start"], spec["seed_start"] + spec["count"])]


def system_prompt(mode, arm):
    from gamescaling.prompt import build_system_prompt
    if mode not in ("mode-0", "mode-1"):
        raise ValueError("unknown mode")
    # The oracle controller exposes normal coordinates to its frozen policy.
    visible_mode = "mode-0" if arm == "key_swap" else mode
    rules = (
        "Kill the single target before the episode ends. Move laterally to line up a shot. "
        "Only A, D and LB are supported. Set mouse X, Y and scroll to zero. "
        "The simulator pauses while you decide. "
        f"Control context: {visible_mode}."
    )
    if arm == "rule_prompt":
        rules += (" In mode-0, A moves left and D moves right."
                  " In mode-1, A moves right and D moves left.")
    return build_system_prompt(
        "ViZDoom Basic", rules=rules,
        key_bindings={"A": "Lateral control A", "D": "Lateral control D", "LB": "Fire"},
        mouse_aim=False,
    )


def request_messages(prompt, history, image_url):
    from gamescaling.policy import user_turn, assistant_turn
    messages = [{"role": "system", "content": prompt}]
    for old_image, executed in history:
        messages.extend([user_turn(old_image), assistant_turn(executed)])
    messages.append(user_turn(image_url))
    return messages


def execute_chunk(game, compiled):
    """Execute one tic at a time; retain only actually issued ticks on termination."""
    executed = []
    for group, (buttons, ticks) in enumerate(zip(compiled.button_steps, compiled.tick_schedule)):
        for _ in range(ticks):
            if game.is_episode_finished():
                return executed
            before = game.get_episode_time()
            reward = game.make_action(list(buttons), 1)
            executed.append({"group": group, "buttons": list(buttons),
                             "tic_before": before, "tic_after": game.get_episode_time(),
                             "reward": float(reward)})
    return executed


def scripted_action(arm, step):
    """Observation-free baselines; success would weaken the task's diagnostic value."""
    from protocol import canonical_history
    if arm == "fire_only":
        keys = ("LB",)
    elif arm == "sweep_fire":
        keys = ("A" if (step // 7) % 2 == 0 else "D", "LB")
    else:
        raise ValueError("unknown scripted policy")
    return canonical_history((keys,) * 6)


def make_game(vzd, seed):
    cfg = config()
    game = vzd.DoomGame()
    game.load_config(str(HERE.parent / "evidence/vizdoom-basic.cfg"))
    game.set_doom_scenario_path(str(Path(vzd.scenarios_path) / cfg["scenario"]))
    game.set_mode(vzd.Mode.PLAYER)
    game.set_seed(seed)
    game.set_ticrate(cfg["ticrate"])
    game.set_episode_start_time(cfg["episode_start_time"])
    game.set_episode_timeout(cfg["episode_timeout"])
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.set_available_game_variables([vzd.GameVariable.AMMO2, vzd.GameVariable.KILLCOUNT])
    game.set_depth_buffer_enabled(False)
    game.set_labels_buffer_enabled(False)
    game.set_automap_buffer_enabled(False)
    game.set_objects_info_enabled(False)
    game.set_sectors_info_enabled(False)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    # A fresh instance per member gives identical RNG initialization, subject to
    # engine verification. We hash the initial pixels rather than assuming it.
    return game


def evaluate(args):
    parsed_url = urlparse(args.base_url)
    if parsed_url.scheme != "http" or parsed_url.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("only an already-authorized loopback model server is supported")
    import vizdoom as vzd
    import requests
    from PIL import Image
    from gamescaling.action import parse_action
    from gamescaling.policy import PolicyConfig, encode_image, build_request, finalize
    from protocol import compile_action, canonical_history, inverse_keys, score_pairs

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings = config()
    pcfg = PolicyConfig(base_url=args.base_url, model=args.model, reasoning=False,
                        temperature=0, max_tokens=settings["max_tokens"], image_format="png")
    receipt = {"status": "RUNNING", "config": settings, "arm": args.arm,
               "checkpoint_label": args.checkpoint_label, "split": args.split,
               "engine_version": vzd.__version__, "model_requests": 0,
               "note": "checkpoint_label is declared, not verified against server weights"}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    results = []
    session = requests.Session()
    session.trust_env = False
    start = time.monotonic()
    try:
        with (out / "steps.jsonl").open("x") as trace, (out / "episodes.jsonl").open("x") as episodes:
            for case in cases(args.split):
                pair_hash = None
                for mode in ("mode-0", "mode-1"):
                    game = make_game(vzd, case["seed"])
                    history = deque(maxlen=settings["history_len"])
                    rejects = 0
                    calls = 0
                    try:
                        game.init()
                        game.set_seed(case["seed"])
                        game.new_episode()
                        state = game.get_state()
                        if state is None:
                            raise RuntimeError("no initial state")
                        digest = hashlib.sha256(state.screen_buffer.tobytes()).hexdigest()
                        if pair_hash is not None and digest != pair_hash:
                            raise RuntimeError("paired initial pixels differ; no paired score allowed")
                        pair_hash = digest
                        initial_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                        for step in range(settings["max_decisions"]):
                            if game.is_episode_finished():
                                break
                            state = game.get_state()
                            if state is None:
                                raise RuntimeError("missing state before termination")
                            image = Image.fromarray(state.screen_buffer)
                            frame = f"{case['case_id']}-{mode}-{step:03d}.png"
                            image.save(out / frame)
                            image_url = encode_image(image, pcfg)
                            if time.monotonic() - start > args.max_seconds:
                                raise RuntimeError("wall-clock budget reached; partial run, no final score")
                            t0 = time.monotonic()
                            if args.arm in ("fire_only", "sweep_fire"):
                                raw = scripted_action(args.arm, step)
                                text, payload, choice = raw, {}, {}
                            else:
                                body = build_request(request_messages(system_prompt(mode, args.arm), history, image_url), pcfg)
                                receipt["model_requests"] += 1
                                calls += 1
                                response = session.post(args.base_url.rstrip("/") + "/chat/completions", json=body, timeout=120)
                                response.raise_for_status()
                                payload = response.json()
                                choice = payload["choices"][0]
                                raw = choice["message"]["content"]
                                text = finalize(raw, choice.get("finish_reason"), pcfg)
                            proposal = parse_action(text, lenient=False)
                            compiled = compile_action(proposal, mode)
                            policy_action = compiled.history_text
                            if args.arm == "key_swap" and mode == "mode-1" and compiled.accepted:
                                swapped = canonical_history(tuple(inverse_keys(g) for g in compiled.logical_steps))
                                compiled = compile_action(parse_action(swapped), mode)
                            latency = (time.monotonic() - t0) * 1000
                            ticks = execute_chunk(game, compiled)
                            rejects += int(not compiled.accepted)
                            row = {**case, "mode": mode, "step": step, "frame": frame,
                                   "raw_reply": raw, "proposed_action": text,
                                   "issued_action": compiled.history_text,
                                   "policy_history_action": policy_action if args.arm == "key_swap" else compiled.history_text,
                                   "accepted": compiled.accepted, "rejection_reason": compiled.rejection_reason,
                                   "executed_ticks": ticks, "request_and_parse_ms": latency,
                                   "usage": payload.get("usage", {}), "finish_reason": choice.get("finish_reason")}
                            trace.write(json.dumps(row) + "\n")
                            trace.flush()
                            if len(ticks) == 7 and not game.is_episode_finished():
                                history.append((image_url, row["policy_history_action"]))
                        if not game.is_episode_finished():
                            raise RuntimeError("decision cap reached before engine termination")
                        kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT) - initial_kills
                        result = {**case, "mode": mode, "success": bool(kills >= 1), "kills": kills,
                                  "reward": game.get_total_reward(), "initial_frame_sha256": digest,
                                  "rejected_chunks": rejects, "model_requests": calls}
                        episodes.write(json.dumps(result) + "\n")
                        episodes.flush()
                        results.append(result)
                    finally:
                        game.close()
        receipt["status"] = "COMPLETE"
        receipt["scores"] = score_pairs(results)
        receipt["distinct_initial_frames"] = len({r["initial_frame_sha256"] for r in results})
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        receipt["status"] = "INCOMPLETE"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        session.close()
        receipt["wall_seconds"] = time.monotonic() - start
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--split", choices=list(config()["splits"]), default="calibration")
    score = sub.add_parser("score")
    score.add_argument("episodes")
    run = sub.add_parser("evaluate", help="requires authorized installed ViZDoom and local model server")
    run.add_argument("--split", choices=list(config()["splits"]), default="calibration")
    run.add_argument("--arm", choices=["frozen", "rule_prompt", "key_swap", "fire_only", "sweep_fire"], default="frozen")
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--model", default="gamescaling")
    run.add_argument("--checkpoint-label", required=True)
    run.add_argument("--max-seconds", type=int, default=1800)
    run.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps({"status": "NOT_RUN", "config": config(), "cases": cases(args.split)}, indent=2))
    elif args.command == "score":
        from protocol import score_pairs
        records = [json.loads(line) for line in Path(args.episodes).read_text().splitlines() if line.strip()]
        print(json.dumps(score_pairs(records), indent=2))
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
