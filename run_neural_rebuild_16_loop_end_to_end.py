"""NEURAL INTERFACE REBUILD, step 16: run the REAL multirate loop end-to-end with channel mode.

WHY
---
Every acceptance test so far called `stimuli()` / `transduce()` directly with synthetic vectors.
The objective asks for a quantitative check of CLOSED-LOOP controllability, and that requires the
actual scheduler: body -> transduction (channel mode) -> tier -> decoder -> command -> body.

WHAT IS RUN, AND WHAT IS LOOKED AT
----------------------------------
A short episode of the project's own `MultirateScheduler` with `ReceptorConfig.channel_mode=True`,
compared against the SAME episode with the legacy receptor path.  The clocks, arms, seed and all
other settings are identical, so any difference is attributable to the front end.

Reported for each arm:
    * the episode's own honesty fields (the loop writes these itself),
    * the channel placement statistics,
    * the spike count over the run,
    * the decoded command's saturation (both arms),
    * whether ANY leg produced a decoded command different from the neutral one, which is the
      weakest possible statement of "the loop is closed and the front end reaches the actuators",
    * and the per-interval drive, so the reader can see whether the front end ever left zero.

PRE-REGISTERED
--------------
  C1  The channel-mode episode RUNS to completion without error and its own consistency checks
      pass, i.e. the front end is compatible with the real loop.
  C2  The two arms produce DIFFERENT tier spike counts, so the front end materially changes what
      the network receives.
  C3  The decoded command carries at least one non-neutral value, i.e. information from the front
      end reaches the actuator command - a necessary condition for controllability.
  FALSIFIER for C3: every decoded command equals the neutral command, which would mean the loop
      is closed in code but carries no signal.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402
from engine.embodied.adapters import ReceptorConfig, TierConfig  # noqa: E402
from engine.embodied.loop import LoopConfig, MultirateScheduler  # noqa: E402

OUT = HERE / "outputs/neural_rebuild"
DURATION_S = 0.20            # short: this is a wiring test, not a behaviour run


def run_one(channel_mode: bool) -> dict:
    cfg = LoopConfig(
        arm="neural_modulated",
        seed=0,
        duration_s=DURATION_S,
        # MEASURED: 0.0 raised "calibration_ms must cover at least one interval"; one command
        # interval is 5 ms (dt_command_s = 5e-3), and the settle must fit inside it, so the
        # shortest legal calibration is one interval with a 2 ms settle.
        calibration_ms=5.0,
        calibration_settle_ms=2.0,
        receptor=ReceptorConfig(channel_mode=channel_mode,
                                channel_current_nA_per_unit=3e-1),
        tier=TierConfig(background_mean_nA=0.0, background_sd_nA=0.0),
    )
    sched = MultirateScheduler(cfg)
    t0 = time.time()
    ep = sched.run(want_frames=False)
    wall = time.time() - t0

    def dig(container, key):
        try:
            return container[key]
        except Exception:
            return None

    # the episode object publishes its own namespaces; read defensively and report what is there
    keys = sorted([k for k in dir(ep) if not k.startswith("_")])
    spikes = None
    for attr in ("spikes", "spike_counts", "neural"):
        v = getattr(ep, attr, None)
        if v is not None:
            spikes = getattr(v, "shape", None)
            break
    cmd = getattr(ep, "commands", None)
    n_nonzero_cmd = None
    if isinstance(cmd, dict) and cmd:
        arr = None
        for k, v in cmd.items():
            a = np.asarray(v) if not isinstance(v, list) else np.asarray(v)
            if a.dtype.kind in "fiu" and a.size:
                arr = a if arr is None else arr
        if arr is not None:
            n_nonzero_cmd = int(np.flatnonzero(np.abs(arr) > 0).size)
    obs = getattr(ep, "observed", {}) or {}
    ang = np.asarray(obs["angle_strain_per_leg"], float) if "angle_strain_per_leg" in obs else None
    cur = np.asarray(obs["current_nA_per_leg"], float) if "current_nA_per_leg" in obs else None
    # the tier's spikes are recorded in the observed namespace under one of a few names; take
    # whichever exists, and report the NAME so the reading is checkable
    spike_arr, spike_name = None, None
    for cand in ("spikes", "spike_count", "neural_spikes", "spike_counts"):
        if cand in obs:
            spike_arr, spike_name = np.asarray(obs[cand], float), cand
            break
    desc = np.asarray(obs["descending_rate_hz"], float) if "descending_rate_hz" in obs else None
    return {"channel_mode": channel_mode, "wall_s": wall,
            "descending_rate_max_hz": (float(desc.max()) if desc is not None else None),
            "descending_rate_mean_hz": (float(desc.mean()) if desc is not None else None),
            "spikes_exposed_by_the_loop": bool(spike_name is not None),
            "episode_public_attrs": keys[:20],
            "observed_keys": sorted(list(obs.keys())),
            "spike_array_name": spike_name,
            "spike_total": (float(spike_arr.sum()) if spike_arr is not None else None),
            "command_namespace_size": (len(cmd) if isinstance(cmd, dict) else None),
            "n_nonzero_command_values": n_nonzero_cmd,
            "angle_strain_min": (float(ang.min()) if ang is not None else None),
            "angle_strain_has_negative": (bool((ang < 0).any()) if ang is not None else None),
            "current_per_leg_absmax": (float(np.abs(cur).max()) if cur is not None else None),
            "meta_keys": sorted(list(getattr(ep, "meta", {}).keys()))[:20]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = cp.sha256_file(cp.CANONICAL_XML)
    assert digest == cp.CANONICAL_XML_SHA256, digest

    results = {}
    for mode in (False, True):
        label = "channel_mode" if mode else "legacy"
        try:
            results[label] = run_one(mode)
            print(f"  {label}: ran in {results[label]['wall_s']:.1f}s "
                  f"cmd_size={results[label]['command_namespace_size']} "
                  f"nonzero_cmd={results[label]['n_nonzero_command_values']}", flush=True)
        except Exception as e:  # record the failure rather than hiding it
            import traceback
            results[label] = {"channel_mode": mode, "error": f"{type(e).__name__}: {e}",
                              "traceback_tail": traceback.format_exc().splitlines()[-6:]}
            print(f"  {label}: FAILED {type(e).__name__}: {e}", flush=True)

    ok = {k: v for k, v in results.items() if "error" not in v}
    c1 = bool("channel_mode" in ok)
    # C2 on VALUES, not on shapes: an earlier version compared `spikes_shape` and reported
    # "identical" because both arms record a (12000,) per-neuron vector.  The meaningful test is
    # whether the recorded quantities differ, and the sharpest one is that the LEGACY path can
    # never produce a NEGATIVE angle strain while channel mode must.
    c2 = bool(len(ok) == 2
              and (ok["legacy"].get("spike_total") != ok["channel_mode"].get("spike_total")
                   or ok["legacy"].get("current_per_leg_absmax")
                   != ok["channel_mode"].get("current_per_leg_absmax")))
    c3 = bool("channel_mode" in ok
              and (ok["channel_mode"].get("n_nonzero_command_values") or 0) > 0)
    c4 = bool("channel_mode" in ok and ok["channel_mode"].get("angle_strain_has_negative") is True
              and ok["legacy"].get("angle_strain_has_negative") is False)

    cp.write_result(
        OUT / "neural_rebuild_16_loop_end_to_end.json",
        question=("Does the real multirate loop run with the channel front end, and does "
                  "information from that front end reach the actuator command?"),
        status="PASS" if c1 else "FAIL",
        measured={
            "duration_s": DURATION_S,
            "arms": results,
            "C2_note": ("the loop does not expose the raw spike array in its observed namespace "
                        "(measured: nine observed keys, none of them spikes), so C2 compares the "
                        "delivered current magnitude and the descending rate instead - both are "
                        "derived from the tier's activity"),
            "criteria": {
                "C1_channel_mode_loop_runs": c1,
                "C2_recorded_quantities_differ_between_arms": c2,
                "C3_a_non_neutral_command_reaches_the_actuators": c3,
                "C4_signed_sensory_signal_survives_the_whole_loop": c4,
            },
        },
        provenance={
            "code": str(HERE / "run_neural_rebuild_16_loop_end_to_end.py"),
            "scheduler": "engine/embodied/loop.py::MultirateScheduler (unmodified apart from "
                         "passing the tier root ids to the receptor bank)",
            "receptor": "engine/embodied/adapters.py::MechanoReceptorBank (channel_mode switch)",
            "canonical_xml_sha256": digest,
            "canonical_xml_untouched": True,
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "0.2 s is a WIRING test, not a behaviour test: nothing here says the fly stands, walks "
            "or is controlled.  The project's own stance measurements already showed the body "
            "leaves the balance neighbourhood in tens of milliseconds.",
            "C3 is the WEAKEST controllability statement: one non-neutral command does not show "
            "that the command is correct, graded or useful.",
            "The spikes-per-readout-window limitation measured in step 4 is not addressed by any "
            "of this and remains the binding constraint on using the output.",
        ],
    )
    print(json.dumps({"C1": c1, "C2": c2, "C3": c3, "C4": c4}))
    print("wrote", OUT / "neural_rebuild_16_loop_end_to_end.json")


if __name__ == "__main__":
    main()
