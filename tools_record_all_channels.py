#!/usr/bin/env python3
"""ONE RUN, BOTH OBSERVATION CHANNELS: the camera ring AND the electrode, together.

WHY THIS IS THE POINT.  Each channel had been demonstrated alone: the cameras on the driven body
(`tools_render_neural_loop.py`) and the electrode on the driven tier (`tools_record_electrode_in_
loop.py`).  Two channels that have never run together are two separate demonstrations, not an
observation layer -- and the platform's claim is simultaneous observation of one behaving body.

WHAT "TOGETHER" MEANS HERE, CONCRETELY:
  * ONE call to `run_six_leg_neural_loop.run`, so there is one body, one tier and one clock;
  * both sinks are called from that single run, and both record the SAME interval index `c`, so the
    two series share a time base by construction rather than by a later alignment;
  * the electrode is fed the POST-lesion spikes, i.e. exactly the spikes that set the muscle
    activations for the interval the camera is about to render.

THE VISUAL CHANNEL IS STRUCTURALLY ABSENT FROM THIS BODY, AND THAT IS MEASURED, NOT ASSUMED.
`OmmatidiaFrontEnd.bind` needs a FlyGym `Simulation` (`get_raw_vision` / `get_ommatidia_readouts`);
stack B is a raw `mujoco.MjData` built from FlyMimic's MJCF, so binding raises
`AttributeError: 'MjData' object has no attribute 'get_raw_vision'`.  Vision therefore remains a
stack A channel, and no run here can contain it.  Saying so is the point: "three channels together"
would have been a claim this platform cannot support.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_record_all_channels.py --duration-s 0.4 --stride 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

EPISODE = HERE / "outputs" / "arena" / "stackb_all_channels"
DETECT_THRESHOLD = 240.0
DETECT_MIN_PX = 3


def vision_binding_probe(run: bool = True) -> dict:
    """MEASURE whether the visual front end can bind to this body, instead of assuming it cannot."""
    if not run:
        return {"can_bind": False, "error": "NOT_PROBED",
                "why": "the probe was skipped in this run",
                "reason": "see the unskipped run's markers.json"}
    import run_six_leg_neural_loop as loop
    from engine.embodied.vision import OmmatidiaFrontEnd, VisionConfig

    ctx = loop.build(None, observable=True)
    front = OmmatidiaFrontEnd(VisionConfig(n_channels=4, drive_scale_nA_per_unit=0.05))
    try:
        front.bind(ctx["d"], "nmf")
        return {"can_bind": True, "why": "binding succeeded"}
    except Exception as exc:                                   # noqa: BLE001
        return {"can_bind": False, "error": type(exc).__name__, "why": str(exc)[:200],
                "reason": ("the visual front end needs a FlyGym `Simulation` "
                           "(get_raw_vision / get_ommatidia_readouts); stack B is a raw "
                           "`mujoco.MjData` built from FlyMimic's MJCF, so vision is structurally "
                           "a stack A channel")}


def record(duration_s: float = 0.4, stride: int = 4, lesion_leg=None,
           probe_vision: bool = True, fly_markers: bool = True,
           stimulate: bool = False, stim_kwargs: dict | None = None) -> dict:
    import mujoco
    from PIL import Image
    from scipy import ndimage
    from arena import marker_grey
    import run_six_leg_neural_loop as loop
    from engine.embodied.adapters import TierConfig, build_neural_tier
    from engine.embodied.electrode_in_loop import InLoopElectrode, SomaPositions

    built = build_neural_tier(TierConfig())
    n_tier = int(built["n"])
    gi = np.asarray(built["global_index"], dtype=np.int64)
    root_ids = np.asarray(built["data"]["root_ids"], dtype=np.int64)[gi]
    elec = InLoopElectrode(SomaPositions.from_banc(root_ids), n_tier_neurons=n_tier,
                           n_channels=4, dt_neural_ms=float(loop.TIER_DT_MS))

    geom = json.loads((HERE / "outputs" / "arena" / "arena_geometry.json").read_text())
    res = geom.get("resolutions") or {}
    h, w = int(res.get("height", 640)), int(res.get("width", 800))

    state = {"renderer": None, "cams": None}
    rows, cols, peaks, ncan, areas = {}, {}, {}, {}, {}
    electrode_traces, interval_times = [], []
    frame_intervals: list[int] = []
    n_frames = 0
    # THE FLY'S OWN MARKERS, read out of the SAME geoms the camera rendered.  `marker_order` is the
    # declared column order, so the truth array and `marker_body_segments` cannot disagree.
    from tools_make_stackb_observable import STACKB_MARKER_BODY, fly_marker_truth
    marker_order = list(STACKB_MARKER_BODY) if fly_markers else []
    # `attach_fly_markers` names every marker geom `mk_<segment>` deterministically, so the map is
    # rebuilt here rather than threaded through the loop; `fly_marker_truth` then RAISES if any of
    # those geoms is absent from the compiled model, which is the guard that actually matters.
    marker_geoms = {seg: f"mk_{seg}" for seg in marker_order}
    fly_truth: list[np.ndarray] = []
    marker_guard: dict = {}

    def frame_sink(mm, d, c, act_trace, _report=None):
        nonlocal n_frames
        # THE FRAME'S OWN INTERVAL INDEX.  `frame_sink` fires only when `c % stride == 0`, so the
        # frame times are `interval_times[::stride]` -- NOT `interval_times[:n_frames]`.  Using the
        # latter understated the frame clock by exactly `stride` (measured: declared 0..0.095 s
        # while the frames were rendered at 0..0.38 s), which would have made "the two channels
        # share one clock" false in the artifact itself.
        frame_intervals.append(int(c))
        if state["renderer"] is None:
            state["cams"] = sorted(mm.camera(i).name for i in range(mm.ncam)
                                   if mm.camera(i).name.startswith("cam"))
            for cc in state["cams"]:
                (EPISODE / "frames" / cc).mkdir(parents=True, exist_ok=True)
                rows[cc], cols[cc], peaks[cc], ncan[cc], areas[cc] = [], [], [], [], []
            state["renderer"] = mujoco.Renderer(mm, height=h, width=w)
            print(f"  one run, {len(state['cams'])} cameras at {h}x{w}, stride {stride}")
        for cam in state["cams"]:
            state["renderer"].update_scene(d, camera=cam)
            img = state["renderer"].render()
            grey = marker_grey(img)
            Image.fromarray(img).save(EPISODE / "frames" / cam / f"f{n_frames:05d}.png")
            lab, n = ndimage.label(grey >= DETECT_THRESHOLD)
            # THE AREA IS RECORDED, NOT DISCARDED.  MEASURED: without it `limb_tracking`'s
            # identity-free front end cannot tell the fly's 0.12 mm markers (<=32 px) from the
            # arena's 0.30 mm fiducials (>=41 px), and the floor-level fiducial tier gets stolen by
            # the fly's own markers -- the fiducial triangulation fell from 7/12 to 3/12.
            rr, cc_, pk, ar = [], [], [], []
            for i in range(1, n + 1):
                yy, xx = np.nonzero(lab == i)
                if len(yy) >= DETECT_MIN_PX:
                    rr.append(float(yy.mean()))
                    cc_.append(float(xx.mean()))
                    pk.append(float(grey[yy, xx].max()))
                    ar.append(int(len(yy)))
            rows[cam].extend(rr)
            cols[cam].extend(cc_)
            peaks[cam].extend(pk)
            areas[cam].extend(ar)
            ncan[cam].append(len(rr))
        if marker_order:
            fly_truth.append(fly_marker_truth(mm, d, marker_geoms, marker_order))
        n_frames += 1

    def spike_sink(spikes, c):
        # POST-lesion spikes: the same ones that set the muscle activations for this interval
        electrode_traces.append(elec.record(spikes, t0_ms=c * loop.DT_COMMAND_S))
        interval_times.append(float(c * loop.DT_COMMAND_S))

    EPISODE.mkdir(parents=True, exist_ok=True)
    # ---- THE STIMULUS SEAM, ATTACHED THROUGH A FACTORY.  `make_in_loop_stimulus` takes the loop's
    # OWN context, so the electrode that injects is positioned against the very tier being stepped:
    # a stimulus built from a separate `build()` would address a different set of neurons, still
    # change the spikes, and still look correct.
    stim_factory = None
    if stimulate:
        kw = {"channel_index": 0, "amplitude_nA": 1000.0, "capture_radius_um": 50.0,
              "mode": "recruiting", **(stim_kwargs or {})}

        def stim_factory(ctx):                                          # noqa: F811
            stim, elec = loop.make_in_loop_stimulus(ctx, **kw)
            print(f"  STIMULUS: {stim.mode}, {kw['amplitude_nA']} nA on channel "
                  f"{kw['channel_index']}, capture {kw['capture_radius_um']} um -> "
                  f"{stim.n_recruited} of {elec.n_tier_neurons} tier neurons recruited")
            return stim

    summary = loop.run(lesion_leg=lesion_leg, duration_s=duration_s, verbose=False,
                       observable=True, frame_sink=frame_sink, frame_stride=stride,
                       spike_sink=spike_sink, fly_markers=fly_markers,
                       stimulus=stim_factory)
    if state["renderer"] is not None:
        state["renderer"].close()

    cams = state["cams"] or []
    arrays = {}
    for cam in cams:
        arrays[f"px/{cam}/rows"] = np.asarray(rows[cam], dtype=np.float32)
        arrays[f"px/{cam}/cols"] = np.asarray(cols[cam], dtype=np.float32)
        arrays[f"px/{cam}/offsets"] = np.concatenate([[0], np.cumsum(ncan[cam])]).astype(np.int64)
        arrays[f"px/{cam}/n_candidates"] = np.asarray(ncan[cam], dtype=np.int64)
        arrays[f"px/{cam}/peak_grey"] = np.asarray(peaks[cam], dtype=np.float64)
        arrays[f"px/{cam}/area"] = np.asarray(areas[cam], dtype=np.int64)
    from arena import FIDUCIAL_MARKERS, MARKER_GREY_PROVENANCE
    dec = np.asarray([x["pos_mm"] for x in FIDUCIAL_MARKERS], dtype=np.float64)
    arrays["truth/marker_static_world_mm"] = np.repeat(dec[None, :, :], n_frames, axis=0)

    # THE SHARED CLOCK, taken from the one interval clock rather than re-derived.
    if len(frame_intervals) != n_frames:
        raise RuntimeError(f"{len(frame_intervals)} frame intervals but {n_frames} frames rendered")
    f_times = np.asarray([interval_times[i] for i in frame_intervals], dtype=np.float64)
    if n_frames > 1:
        step = float(np.diff(f_times).mean())
        want = float(stride) * float(loop.DT_COMMAND_S)
        if abs(step - want) > 1e-9:
            raise RuntimeError(f"frame clock {step:.6g} s != stride*DT {want:.6g} s")
    arrays["truth/time_s"] = f_times
    arrays["truth/frame_interval_index"] = np.asarray(frame_intervals, dtype=np.int64)
    arrays["truth/stride"] = np.asarray([stride], dtype=np.int64)
    if marker_order:
        ft = np.asarray(fly_truth, dtype=np.float32)          # (n_frames, n_markers, 3)
        if ft.shape[0] != n_frames or ft.shape[1] != len(marker_order):
            raise RuntimeError(f"fly marker truth has shape {ft.shape} for {n_frames} frames and "
                               f"{len(marker_order)} markers; refusing to write a misaligned truth")
        arrays["truth/marker_fly_world_mm"] = ft
    np.savez_compressed(EPISODE / "episode.npz", **arrays)

    # THE ELECTRODE GETS ITS OWN FILE, with the SAME interval clock as the frames -- and with its
    # OWN sample axis, because it samples FINER than the frames do: `elec.record` returns one column
    # per NEURAL substep (N_NEURAL_SUB = 10 per 5 ms interval, i.e. every 0.5 ms), so a trace of 800
    # columns cannot be indexed by the 80 interval times.  Writing only the interval clock would
    # leave the electrode's own time base undeclared, which is the same class of gap as the stride
    # bug fixed above.
    etrace = np.concatenate(electrode_traces, axis=1) if electrode_traces else np.zeros((4, 0))
    dt_sub_s = float(loop.TIER_DT_MS) / 1000.0
    n_sub = int(round(float(loop.DT_COMMAND_S) / dt_sub_s))
    e_times = np.asarray([t + k * dt_sub_s
                          for t in interval_times for k in range(n_sub)], dtype=np.float64)
    if etrace.shape[1] != e_times.size:
        raise RuntimeError(f"electrode has {etrace.shape[1]} samples but the substep axis has "
                           f"{e_times.size} ({n_sub} per interval x {len(interval_times)} intervals)")
    np.savez_compressed(EPISODE / "electrode.npz",
                        trace_mV=etrace,
                        electrode_time_s=e_times,
                        interval_time_s=np.asarray(interval_times, dtype=np.float64),
                        samples_per_interval=np.asarray([n_sub], dtype=np.int64),
                        channel_um=np.asarray(elec.channel_um, dtype=np.float64),
                        n_neurons_with_soma=np.asarray([elec.n_neurons], dtype=np.int64),
                        n_tier_neurons=np.asarray([n_tier], dtype=np.int64))
    # THE STIMULUS GETS ITS OWN FILE, with the charge ceiling provenance carried through rather than
    # restated -- this module must not be able to upgrade an assumed ceiling into a measured one.
    if summary.get("stimulus"):
        st = summary["stimulus"]
        try:
            # A DATACLASS, NOT A DICT.  MEASURED: json.dumps raises
            # "float() argument must be a string or a real number, not 'ChargeInjectionLimits'",
            # because this module already declares its own default=float for numpy scalars.  The
            # limits are carried through as fields, with their provenances intact.
            import dataclasses
            from engine.electrode_stim import charge_injection_limits
            _lim = charge_injection_limits()
            limits = (dataclasses.asdict(_lim) if dataclasses.is_dataclass(_lim)
                      else dict(getattr(_lim, "__dict__", {})) or {"repr": repr(_lim)})
        except Exception as exc:                                        # noqa: BLE001
            limits = {"status": "UNVERIFIABLE", "why": f"{type(exc).__name__}: {exc}"}
        (EPISODE / "stimulus.json").write_text(json.dumps({
            "mode": st["mode"], "amplitude_nA": st["amplitude_nA"],
            "capture_radius_um": st["capture_radius_um"],
            "n_recruited_neurons": st["n_recruited_neurons"],
            "n_intervals": st["n_intervals"],
            "n_intervals_with_current": st["n_intervals_with_current"],
            "total_charge_nA_ms": st["total_charge_nA_ms"], "peak_abs_nA": st["peak_abs_nA"],
            "per_interval": st["per_interval"],
            "charge_limits_from_the_project": limits,
            "what_this_does_not_establish": (
                "that the stimulation is biologically meaningful or safe.  WHICH neurons receive how "
                "much current is an ASSUMED geometric rule -- the access map has soma voxel positions "
                "and nothing else, and its capture radius is DECLARED, not measured.  The sign that "
                "maps a cathodic phase onto a depolarising membrane current is a declared assumption, "
                "and at this tier's dt the recovery phase is never sampled, so the two modes differ "
                "by SIGN alone rather than by charge balance.  NO charge-safety claim."),
            "units": "nA, nA*ms, um",
        }, indent=2, sort_keys=True, default=float))

    (EPISODE / "markers.json").write_text(json.dumps({
        # THE FLY'S OWN 20 SEGMENTS, not the 3-segment placeholder this file used to declare.  It also
        # used to name `c_rostrum`, a stack A segment that does not exist on this body, so a reader
        # would have looked for a marker that was never attached.
        "marker_body_segments": marker_order,
        "fly_marker_truth_key": "truth/marker_fly_world_mm",
        "stack_a_logical_to_stack_b_body": (
            {k: v for k, v in STACKB_MARKER_BODY.items()} if marker_order else None),
        "static_markers": [{"name": x["name"], "pos_mm": list(x["pos_mm"])}
                           for x in FIDUCIAL_MARKERS],
        "stack": "B (FlyMimic muscles) DRIVEN by the connectome",
        "grey_conversion": MARKER_GREY_PROVENANCE,
        "channels_recorded_in_this_run": ["camera_ring", "electrode"],
        "channels_absent_from_this_body": {"vision": vision_binding_probe(probe_vision)},
        "lesion_leg": lesion_leg, "frame_stride": stride,
        "shared_interval_clock": {
            "interval_dt_s": float(loop.DT_COMMAND_S),
            "frame_dt_s": float(np.diff(f_times).mean()) if n_frames > 1 else None,
            "frame_interval_index": frame_intervals,
            "note": ("both sinks are called from ONE run and index the SAME interval counter; the "
                     "frames are the intervals with c % stride == 0"),
        },
    }, indent=2, sort_keys=True))

    return {"episode": str(EPISODE), "n_frames": n_frames, "cameras": cams,
            "lesion_leg": lesion_leg, "n_intervals": len(interval_times),
            # THESE THREE WERE MISSING FROM THE SUMMARY even after the fields existed inside the
            # function: two earlier attempts to insert them silently matched nothing and the record
            # simply did not say how many markers it carried or whether it was stimulated.  A
            # summary that omits what the run was is how a stimulated episode gets read as a baseline.
            "n_fly_markers": len(marker_order),
            "stimulated": bool(stimulate),
            "stimulus": summary.get("stimulus"),
            "frame_interval_indices": frame_intervals[:4] + ["..."] + frame_intervals[-1:],
            "frame_clock_dt_s": float(np.diff(f_times).mean()) if n_frames > 1 else None,
            "interval_clock_dt_s": float(loop.DT_COMMAND_S),
            "electrode_shape": list(etrace.shape),
            "electrode_peak_mV": float(np.abs(etrace).max()) if etrace.size else 0.0,
            "electrode_geometry": elec.geometry(),
            "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=0.4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--lesion-leg", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-fly-markers", action="store_true",
                    help="omit the fly's own markers; limb_tracking cannot run on such an episode")
    ap.add_argument("--stimulate", action="store_true",
                    help="attach the NeuralStimulus seam inside this loop (1000 nA, recruiting)")
    ap.add_argument("--no-vision-probe", action="store_true",
                    help="skip the (slow) vision bind attempt; it is recorded as NOT_PROBED")
    args = ap.parse_args()
    global EPISODE
    if args.out_dir:
        EPISODE = Path(args.out_dir)
    info = record(args.duration_s, args.stride, args.lesion_leg,
                  probe_vision=not args.no_vision_probe,
                  fly_markers=not args.no_fly_markers, stimulate=args.stimulate)
    print(json.dumps({k: v for k, v in info.items() if k != "summary"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
