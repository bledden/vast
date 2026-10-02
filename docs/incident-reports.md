# Report an incident

Click **Report incident** beside the viewer controls, or press **R** while not typing.
The next overlay render captures the current camera without pausing the stream or changing
AI subscriptions. A non-modal panel shows the capture. Optionally select an issue and add a
note, then **Download report**. Nothing is uploaded. Discard closes the local report.

The ZIP contains `incident.json`, `detections.json`, `events.json`, `README.txt`, available
`frame.png` / `overlay.png`, and SHA-256 hashes when the browser supports Web Crypto.
Boxes are in detection coordinates. Video and overlay dimensions are recorded independently;
no calibrated mapping is asserted. Layer toggles and AI-off are preserved. History is bounded.

This is a snapshot, **not a recording**: the viewer has no rewindable video buffer. The renderer
timestamp's units follow the existing viewer's millisecond assumption, which has not been
independently validated against the player. Its difference from a detection timestamp is
alignment information, not end-to-end latency. Canvas readback may fail or yield a player
surface whose content cannot be verified; metadata still exports, and missing images are
explicit. Report data and predictions are not human-reviewed training labels.

Weave correlation and a Breakpoint importer are not implemented by this change. No Open in
Breakpoint button is shown. Events have pending/unavailable trace metadata, not fabricated
references. Portable reports can be inspected without either product.

No worker, detector, gating, model weights, event decisions, stream protocol or default launch
command is changed. Capture has browser-side copying/encoding cost only when requested.
No video/history recorder or new network subscription is introduced. Relay URLs and query
strings are not serialized. Camera names, event summaries, images and user notes are report
content: review them before sharing the downloaded archive.

## Checks

`node --test tests/incident.test.mjs`

For a synthetic browser check, serve the repository with a local static server and open
`tests/incident-browser.html`. It exercises real canvas encoding and download without starting
camera/model work. It does not establish live moq-watch pixel or timestamp correctness.
