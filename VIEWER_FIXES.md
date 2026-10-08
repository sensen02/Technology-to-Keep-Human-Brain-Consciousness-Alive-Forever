# Viewer integration and correctness fixes

Scope: `viewer/app.js` and `viewer/style.css`; index, bootstrap, data and backend are owned by the parent agent. These are manually implemented prototype corrections, not a validated biological simulator.

## Actual scene corrections
- Neural bounds include reference/return positions; focusing computes the actual transformed visible geometry bounding sphere, including shafts, and fits both vertical and horizontal FOV. Body focus consequently targets the root-relative displayed body rather than an assumed origin.
- Labels reject behind-camera projection and are constrained within viewport margins. The mobile stage has an explicit bounded height; canvas CSS sizing is independent of WebGL drawing-buffer dimensions and ResizeObserver observes container changes.
- Invalid points no longer become origin points. Compact geometry retains source-index mapping for activity colors and selection.
- Picking checks actual screen point footprint, depth range, parent visibility and clipping planes, rather than a distance-scaled world threshold. Mesh hits are filtered for clipping. Frontmost visible candidates win. Points now write opaque depth so they cannot incorrectly overlay opaque shafts; translucent body surfaces retain depthWrite=false.
- Electrode current contact = translated shaft end, not shaft center or original target. Insertion moves ring and enabled 50 µm neighborhood, and recalculates listed geometric nearest neighbors. Checking capture after selection immediately copies current contact position. Both original and current illustrative contact coordinates are displayed.
- Record-chain inspection clears scene selection so playback cannot overwrite the panel. Selected inspectors use frame/mode/depth signatures instead of rebuilding on every RAF; geometry selection markers still track moving bodies.
- Valid monotonic times_s clocks determine each independent dataset's current frame by binary search. Datasets without usable clocks fall back to normalized frame index. The 20-second presentation cycle is normalized replay, not wall-clock simulation. First RAF and page visibility resets prevent initial jumps; expensive frame updates occur only when source frame indices change.

## Project service integration
- GET /api/status reports actual service/job/bench state and job details. POST /api/jobs/recording starts only the bounded recording-budget task. GET /api/recording renders bands/tests summary, never drives world physics or scene animation. Status polls sequentially every 2.5 seconds; HTTP/network errors are visible, not silently treated as success.
- #service-panel is populated when supplied by bootstrap/index, otherwise created in the right sidebar. Test IDs: service-health, run-recording, service-job-log, recording-result.
- Two file inputs bench-config and bench-csv submit POST /api/bench/evaluate with JSON {config, csv}. Server report response is used directly; INVALID_INPUT responses are surfaced. bench-evaluate triggers evaluation.
- Existing report import is separate and explicitly read-only/unverified (bench-report), not a substitute for raw evidence evaluation. Inputs are limited to 2 MB each.
- All scene dynamics remain imported replay. Service recording jobs do not imply live electrophysiology, export or world physics.

## Verification
- `node --check viewer/app.js` passes (rerun after final changes).
- Parent agent owns live browser/CDP regression testing at the integrated project URL and reports its results separately. No visual-validation claim is made here.
- No heavy simulation was launched by this viewer task. Remaining prototype limitations include independent coordinate systems, schematic insertion, no biological registration, and no physical evidence attestation of uploaded CSV measurements.
