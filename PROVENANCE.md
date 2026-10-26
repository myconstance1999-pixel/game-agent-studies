# Selected-source provenance

AI-assisted curation on 2026-10-25 from existing local game-agent studies. Local additions are the Minecraft rounding probe, exact state/action witness checker, and Doom execution/history/scoring/export wrappers with their synthetic offline tests. They are studies and prepared tooling, not a trained policy or measured improvement in live games.

The Doom wrappers originally referenced a local pinned GameScaling checkout. Only their dependency path literals are relocated to `third_party/`; the five required upstream modules are unchanged. The evaluator's JSON configuration omits the non-executable, unrun training draft. No ongoing hypothesis diary, original experiment chronology, real trace, frame, video, full paper or model is exported. Generated arithmetic fixtures are synthetic; machine-specific command paths are omitted from the shared witness receipt while source hashes and assertions remain.

GameScaling source is pinned to `THUSI-Lab/GameScaling` commit `6764efabfbb4121d6e5d3cfcbc54dedf1469bf5a`. The complete upstream checkout is not included. Actual included modules and their license are listed in THIRD_PARTY_NOTICES.md. Whale Craft is linked solely for source analysis at its separately pinned commit; none of its implementation is copied.

No repository-wide license has been selected for the local original code. Public source availability does not grant an MIT or Apache license over that code. Third-party licenses apply only to the identified upstream material.
