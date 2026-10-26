# Game Agent Studies

Check how instructions become actions, what actually executes, and whether an action reaches its intended outcome. This package brings together small Minecraft-inspired arithmetic examples and a prepared Doom comparison kit, with a source study of DSH's Minecraft agent.

## Try the offline checks

Use Python 3.12. The two arithmetic scripts need only the standard library. The Doom suite also needs Pillow and requests; use an existing environment containing them. No installer, game, model or server is bundled or started by these commands.

```sh
python mc/mouse_quantization_probe.py
python action-outcome/codec_witness.py
python -m unittest discover -s doom -p 'test_*.py' -v
python doom/run.py plan
```

- **Movement rounding:** compare splitting a displacement into independent rounded steps with carrying the fractional remainder. The script checks 4,002 integer-axis cases and writes `mc/mouse_quantization_result.json`.
- **Action versus outcome:** seven exact examples and 729 finite identities cover click ambiguity, dragging, clipping and slot boundaries. The same action discrepancy can leave one state unchanged and select a different slot in another. Results go to `action-outcome/results/witness.json`.
- **Action/history contracts:** the Doom tools reject unsupported actions, record issued controls, map normal/reversed lateral keys and score both members of a paired trial. Offline tests use synthetic records and fake engine/HTTP boundaries.
- **Comparison records:** `doom/export_pairs.py` turns complete successful normal-mode **training** rollouts into paired labels and histories. Invalid, incomplete or non-training inputs are refused. No real rollout dataset is included.

`doom/run.py plan` prints eight calibration cases marked `NOT_RUN`. Its separate `score` command scores an episode JSONL; `evaluate` is a prepared live-game path requiring ViZDoom and an explicitly configured model server. That path and real game/model outcomes have not been validated. The configuration includes no runnable training pipeline.

## DSH and Minecraft

[The pinned source study](DSH_MINECRAFT.md) follows Whale Craft's watchdog events and stop handling. It distinguishes steering, a fallback prompt, and conditional turn cancellation rather than treating every stop as the same operation. This is source analysis, not a tested owner integration or a pause/resume facility.

See [PROVENANCE.md](PROVENANCE.md) for the selected-source boundary and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the five necessary GameScaling modules and their Apache-2.0 license. Other original code has no selected repository-wide license. The [export manifest](EXPORT_MANIFEST.json) lists the exact files. No author videos, screenshots, game assets, weights, private research records or actual experiment traces are included.
