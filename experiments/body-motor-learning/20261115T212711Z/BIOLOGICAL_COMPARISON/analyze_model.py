"""Read-only kinematic/contact audit for the two learned righting traces.

This script does not integrate the MuJoCo model.  It replays the saved qpos samples
through mj_forward only so foot positions and thorax orientation can be expressed in
the thorax frame.  Native saved contact arrays remain the source of all load metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(RUN))

from search import stand  # noqa: E402


CASES = {
    "side_fallen": RUN / "FIT_side_fallen" / "TRACE.npz",
    "near_supine": RUN / "FIT_near_supine" / "TRACE.npz",
}
LEG_ORDER = tuple(stand.physics.LEG_ORDER)
HIND = (LEG_ORDER.index("LH"), LEG_ORDER.index("RH"))
NATIVE_DT = 12.5e-6
W = float(stand.physics.W)


def first_true(values: np.ndarray) -> float | None:
    found = np.flatnonzero(values)
    return None if not len(found) else float(found[0])


def first_continuous(values: np.ndarray, width: int) -> float | None:
    if width > len(values):
        return None
    cumulative = np.r_[0, np.cumsum(values, dtype=np.int64)]
    found = np.flatnonzero(cumulative[width:] - cumulative[:-width] == width)
    return None if not len(found) else float(found[0])


def forward_geometry(bench, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return thorax-frame pad centroids and thorax rotation matrices."""
    body = bench.body
    data = mj.MjData(body.plant)
    pads = [body.pads[leg] for leg in LEG_ORDER]
    positions = np.empty((len(qpos), len(LEG_ORDER), 3), dtype=float)
    rotations = np.empty((len(qpos), 3, 3), dtype=float)
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mj.mj_forward(body.plant, data)
        rotation = data.xmat[body.th].reshape(3, 3).copy()
        center = data.xpos[body.th].copy()
        rotations[i] = rotation
        for leg, geoms in enumerate(pads):
            world = data.geom_xpos[geoms].mean(axis=0)
            positions[i, leg] = rotation.T @ (world - center)
    return positions, rotations


def rotation_path(rotations: np.ndarray, stop_ms: int) -> dict[str, float]:
    stop_ms = max(1, min(stop_ms, len(rotations) - 1))
    increments = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
    vectors = Rotation.from_matrix(increments[:stop_ms]).as_rotvec()
    absolute = np.abs(vectors).sum(axis=0)
    total = float(absolute.sum())
    names = ("body_x_roll", "body_y_pitch", "body_z_yaw")
    return {
        f"absolute_path_{name}_deg": float(np.rad2deg(value))
        for name, value in zip(names, absolute, strict=True)
    } | {
        f"path_fraction_{name}": (float(value / total) if total else 0.0)
        for name, value in zip(names, absolute, strict=True)
    }


def analyze(path: Path, bench) -> dict:
    with np.load(path, allow_pickle=False) as trace:
        foot = trace["foot"][:, :, 2].astype(float)
        upright = trace["upright"].astype(float)
        qpos = trace["qpos_ms"].astype(float)
        nonfoot_events = trace["nonfoot"].astype(float)
        body_names = trace["body_names"].astype(str)
        positions, rotations = forward_geometry(bench, qpos)

    tilt_deg = np.rad2deg(np.arccos(np.clip(upright, -1.0, 1.0)))
    first_under = {}
    for threshold in (90, 45, 30, 20, 10):
        native_index = first_true(tilt_deg <= threshold)
        first_under[f"first_tilt_le_{threshold}_ms"] = (
            None if native_index is None else native_index * NATIVE_DT * 1000.0
        )
    for duration_ms in (100, 200):
        width = int(round(duration_ms / 1000 / NATIVE_DT))
        native_index = first_continuous(tilt_deg <= 20, width)
        first_under[f"first_continuous_{duration_ms}ms_tilt_le_20_ms"] = (
            None if native_index is None else native_index * NATIVE_DT * 1000.0
        )
    upright_ms = first_under["first_tilt_le_20_ms"]
    stop_native = len(foot) if upright_ms is None else max(1, int(round(upright_ms / 1000 / NATIVE_DT)))
    stop_ms = len(qpos) - 1 if upright_ms is None else int(round(upright_ms))

    pre = np.maximum(foot[:stop_native], 0.0)
    impulses = pre.sum(axis=0) * NATIVE_DT
    impulse_total = float(impulses.sum())
    nonfoot_pre = nonfoot_events[nonfoot_events[:, 0] < stop_native]
    nonfoot_impulses = {}
    nonfoot_peak = {}
    for body_id in np.unique(nonfoot_pre[:, 1]).astype(int):
        selected = nonfoot_pre[nonfoot_pre[:, 1] == body_id]
        native_force = np.bincount(
            selected[:, 0].astype(int), weights=selected[:, 8], minlength=stop_native
        )
        name = str(body_names[body_id])
        nonfoot_impulses[name] = float(native_force.sum() * NATIVE_DT)
        nonfoot_peak[name] = float(native_force.max() / W)
    nonfoot_total = float(sum(nonfoot_impulses.values()))
    all_vertical_impulse = impulse_total + nonfoot_total
    contact = foot >= 0.02 * W
    first_contacts = []
    for leg in range(len(LEG_ORDER)):
        index = first_true(contact[:, leg])
        first_contacts.append(None if index is None else index * NATIVE_DT * 1000.0)

    initial_x = positions[0, :, 0]
    before_upright = positions[: stop_ms + 1]
    anterior_excursion = before_upright[:, :, 0].max(axis=0) - initial_x
    posterior_excursion = initial_x - before_upright[:, :, 0].min(axis=0)

    per_leg = {}
    for i, leg in enumerate(LEG_ORDER):
        per_leg[leg] = {
            "pre_upright_vertical_impulse_native_s": float(impulses[i]),
            "pre_upright_vertical_impulse_fraction": (
                float(impulses[i] / impulse_total) if impulse_total else 0.0
            ),
            "pre_upright_peak_vertical_load_weight": float(pre[:, i].max() / W),
            "pre_upright_contact_fraction": float(contact[:stop_native, i].mean()),
            "first_0p02_weight_contact_ms": first_contacts[i],
            "pad_initial_thorax_x": float(initial_x[i]),
            "max_anterior_excursion_before_upright": float(anterior_excursion[i]),
            "max_posterior_excursion_before_upright": float(posterior_excursion[i]),
        }

    loaded = (pre / W >= 0.02).mean(axis=0)
    dominant = [LEG_ORDER[i] for i in np.argsort(impulses)[::-1]]
    return {
        "trace": str(path),
        "initial_tilt_deg": float(tilt_deg[0]),
        **first_under,
        "pre_upright_native_samples": int(stop_native),
        "legs_with_at_least_5pct_pre_upright_contact": [
            LEG_ORDER[i] for i in np.flatnonzero(loaded >= 0.05)
        ],
        "pre_upright_impulse_rank": dominant,
        "hind_leg_impulse_fraction": (
            float(impulses[list(HIND)].sum() / impulse_total) if impulse_total else 0.0
        ),
        "pre_upright_foot_vertical_impulse_fraction_of_all_contacts": (
            float(impulse_total / all_vertical_impulse) if all_vertical_impulse else 0.0
        ),
        "pre_upright_nonfoot_vertical_impulse_fraction_of_all_contacts": (
            float(nonfoot_total / all_vertical_impulse) if all_vertical_impulse else 0.0
        ),
        "pre_upright_wing_vertical_impulse_fraction_of_all_contacts": (
            float(sum(value for name, value in nonfoot_impulses.items() if name.endswith("Wing"))
                  / all_vertical_impulse)
            if all_vertical_impulse else 0.0
        ),
        "pre_upright_nonfoot_vertical_impulse_native_s": dict(
            sorted(nonfoot_impulses.items(), key=lambda item: item[1], reverse=True)
        ),
        "pre_upright_nonfoot_peak_vertical_load_weight": dict(
            sorted(nonfoot_peak.items(), key=lambda item: item[1], reverse=True)
        ),
        "both_hind_legs_moved_anteriorly_by_0p1": bool(
            np.all(anterior_excursion[list(HIND)] >= 0.1)
        ),
        **rotation_path(rotations, stop_ms),
        "per_leg": per_leg,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="righting-biology-audit-") as directory:
        bench = stand.Bench(Path(directory))
        report = {
            "method": {
                "physics_integration_steps": 0,
                "geometry_operation": "saved qpos -> mj_forward only",
                "contact_source": "saved native TRACE.npz arrays",
                "leg_order": list(LEG_ORDER),
                "body_axes": {"x": "anterior", "y": "lateral", "z": "dorsal"},
                "upright_event": "first saved native sample with thorax tilt <= 20 deg",
            },
            "cases": {name: analyze(path, bench) for name, path in CASES.items()},
        }
    destination = HERE / "MODEL_KINEMATICS.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
