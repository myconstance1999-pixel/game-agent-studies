# DSH / Minecraft: events, steering and stopping

Source analysis of [Whale Craft](https://github.com/yzi1b/whale-craft/blob/ddc21637dc48b0818f2159dc69e4f4b26f64cb83/README.md), pinned to `ddc21637dc48b0818f2159dc69e4f4b26f64cb83`. Its README describes one agent with a watchdog and workspace memory. No Whale Craft source or runtime is bundled here.

The [watchdog](https://github.com/yzi1b/whale-craft/blob/ddc21637dc48b0818f2159dc69e4f4b26f64cb83/src/watchdog.mjs#L630) constructs a `plugin:whale_craft` notice and calls `agent.steer(message)`. When that facility is unavailable, [the fallback](https://github.com/yzi1b/whale-craft/blob/ddc21637dc48b0818f2159dc69e4f4b26f64cb83/src/watchdog.mjs#L645) uses `sessionController.prompt`. These are two different delivery paths; source availability does not establish how either behaves in an installed host.

In [stopSession](https://github.com/yzi1b/whale-craft/blob/ddc21637dc48b0818f2159dc69e4f4b26f64cb83/index.js#L786), turn cancellation is conditional on `cancelTurn`; the function also disarms the watchdog, disconnects and clears session jobs. [mc_stop](https://github.com/yzi1b/whale-craft/blob/ddc21637dc48b0818f2159dc69e4f4b26f64cb83/index.js#L2048) passes `cancelTurn: false` to avoid aborting its own current turn. Therefore “stop” does not imply every call cancels the current LLM turn, nor does it establish pause-in-place/resumption.

This note identifies concrete event and lifecycle paths to inspect when judging a game agent. It contains no host integration, provider invocation or gameplay result.
