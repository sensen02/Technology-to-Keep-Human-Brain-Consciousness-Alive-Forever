"""Single-scene payload builder: one body frame, one clock, layered datasets.

Design rules enforced here (see SCENE_INTEGRATION_PLAN.zh-CN.md):
  * ONE coordinate frame: FlyGym world millimetres. Every layer is expressed
    either in body-local mm (attached to a body node) or in world mm.
  * ONE clock: the body episode's sampled times. Layers with their own clock
    carry their own times and are labelled as independent.
  * Source distances are never rescaled silently: the BANC->body transform is
    rigid (rotation + translation) with scale reported, and the builder stores
    the source micrometre coordinates so the transform is auditable.
  * Provenance per layer: class, source file, sha256, units, assumptions.
  * Bounded memory and bounded payload; every chunk's byte size is reported.

Registration of the BANC CNS is by named landmarks, NOT anatomical
registration. The two landmarks are measured from the data itself:
  * VNC landmark: somas whose ann_nerve mentions a leg nerve (nerve roots).
  * Brain landmark: somas whose ann_nerve mentions the antennal nerve.
Their relative position fixes the anterior direction: antennal cluster
y < leg cluster y in BANC, and the body's anterior axis is -x (verified from
the body model: the rostrum/haustellum, i.e. the mouthparts, are the most
-x segments). Residuals of the fit are computed and reported.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / 'viewer' / 'scene'

# BANC native voxel size, anisotropic. Verified and cited in
# outputs/embodied_body/access_map.json -> units.resolution.
BANC_NM_PER_VOXEL = np.array([4.0, 4.0, 45.0], dtype=np.float64)
NM_PER_UM = 1000.0

# Electrode geometry is a FIXED user input; never searched or optimised here.
ELECTRODE_DIAMETER_UM = 7.0
ELECTRODE_PITCH_UM = 20.0
ELECTRODE_CAPTURE_RADIUS_UM = 50.0   # DECLARED, not measured; reported as such

# Registration defaults: no free rotation (the landmark pair fixes the
# anterior axis), no rescaling. Only a declared dorsal offset is applied.
CNS_DORSAL_OFFSET_MM = 0.05
CNS_BODY_NODE = 1                    # nmf/c_thorax, the CNS attachment node


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        if limit is None:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        else:
            h.update(fh.read(limit))
    return h.hexdigest()


def samples(n: int, limit: int = 100) -> np.ndarray:
    """Deterministic even sample of frame indices."""
    return np.unique(np.linspace(0, n - 1, min(limit, n), dtype=int))


def bounded_rss_mib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --------------------------------------------------------------------------
# source loading
# --------------------------------------------------------------------------
def load_banc_somas(parquet: Path):
    """Return (root_ids int64, positions_um float64) in the BANC micrometre frame."""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet, columns=['pt_root_id', 'pt_position'])
    root = np.asarray(table['pt_root_id'], dtype=np.int64)
    vox = np.asarray(table['pt_position'].to_pylist(), dtype=np.float64)
    um = vox * (BANC_NM_PER_VOXEL / NM_PER_UM)
    return root, um


def load_connectome(npz: Path):
    """Return connectome arrays plus the pre/post -> root_id mapping.

    IMPORTANT: in this product, `pre`/`post` are ROW INDICES into the neuron
    table whose root ids live in `root_ids`; they are NOT root ids. Mapping
    them directly to somas yields zero joins (verified: 0 of 3,037,361 rows).
    """
    with np.load(npz, allow_pickle=False) as z:
        pre = np.asarray(z['pre'], dtype=np.int64)
        post = np.asarray(z['post'], dtype=np.int64)
        syn = np.asarray(z['syn'], dtype=np.float32)
        root = np.asarray(z['root_ids'], dtype=np.int64)
        ann_nerve = np.asarray(z['ann_nerve'])
        ann_super = np.asarray(z['ann_super_class'])
        ann_flow = np.asarray(z['ann_flow'])
        ann_body = np.asarray(z['ann_body_part'])
    n = len(root)
    if pre.max(initial=-1) >= n or post.max(initial=-1) >= n or pre.min(initial=0) < 0:
        raise RuntimeError('connectome row indices out of range for root_ids table')
    return dict(pre=pre, post=post, syn=syn, root=root, ann_nerve=ann_nerve,
                ann_super=ann_super, ann_flow=ann_flow, ann_body=ann_body)


def join_somas(soma_root: np.ndarray, conn_root: np.ndarray):
    """Row index of each connectome neuron inside the soma table, -1 if absent."""
    order = np.argsort(soma_root)
    sorted_root = soma_root[order]
    pos = np.clip(np.searchsorted(sorted_root, conn_root), 0, len(sorted_root) - 1)
    hit = sorted_root[pos] == conn_root
    out = np.where(hit, order[pos], -1)
    return out.astype(np.int64)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------
def cns_registration(soma_root, soma_um, conn, n_neurons):
    """Landmark registration of the CNS into the body frame.

    Returns a dict with the rigid map (rotation, scale, translation, anchor),
    the measured landmarks, and the residuals of the fit. The rotation is a
    pure axis permutation+sign: BANC +y (anterior) -> body -x, BANC +x
    (lateral) -> body -y, BANC +z (dorsal) -> body +z. That permutation is
    forced by the landmark pair and the body's own axis convention; no free
    rotation is fitted.
    """
    nerve = conn['ann_nerve']
    lower = np.array([str(v).lower() for v in nerve])
    leg = np.array([('leg_nerve' in v) for v in lower])
    ant = np.array([('antennal_nerve' in v) for v in lower])

    soma_of_neuron = join_somas(soma_root, conn['root'])
    leg_rows = np.where(leg & (soma_of_neuron >= 0))[0]
    ant_rows = np.where(ant & (soma_of_neuron >= 0))[0]
    if len(leg_rows) < 20 or len(ant_rows) < 10:
        raise RuntimeError('landmark clusters too small to register')
    leg_um = soma_um[soma_of_neuron[leg_rows]].mean(axis=0)
    ant_um = soma_um[soma_of_neuron[ant_rows]].mean(axis=0)

    # anterior direction in the BANC frame, from the landmarks themselves
    anterior = ant_um - leg_um
    axial = float(np.linalg.norm(anterior[:2]))
    if axial < 200.0:
        raise RuntimeError('landmark pair does not separate along the volume long axis')

    center = soma_um.mean(axis=0)
    # Anterior direction measured from the landmark pair: the antennal nerve
    # (brain) sits at LOWER BANC y than the leg nerve roots (VNC), so BANC -y is
    # anterior. Cross-checks on the same data: the antennal cluster spans
    # y 63..259 um and the leg cluster 520..1016 um, i.e. the two clusters sit at
    # opposite ends of the 1074 um long axis, and the intervening gap is the
    # neck. The body's anterior axis is -x (mouthparts are the most -x
    # segments). Composing those two gives the rotation below; it is forced by
    # measurement plus the body's own axis convention, not fitted.
    # Columns are the images of BANC x, y, z in body axes:
    #   BANC +x (lateral)   -> body y
    #   BANC +y (posterior) -> body +x
    #   BANC +z             -> body +z
    rotation = np.array([[0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    scale = 1.0
    local_leg = (leg_um - center) * scale
    local_ant = (ant_um - center) * scale
    offset = np.array([0.0, 0.0, CNS_DORSAL_OFFSET_MM])
    local_vnc = local_leg / NM_PER_UM @ rotation.T + offset
    local_brain = local_ant / NM_PER_UM @ rotation.T + offset
    # consistency assertion: the brain landmark must end up ANTERIOR (more -x)
    # than the VNC landmark, otherwise the rotation sign is wrong.
    if not (local_brain[0] < local_vnc[0] - 0.2):
        raise RuntimeError('landmark registration places the brain posterior to the VNC; '
                           'rotation sign rejected (brain_x=%.4f vnc_x=%.4f)'
                           % (local_brain[0], local_vnc[0]))
    return {
        'method': 'named nerve-root landmarks; rigid, no free rotation, scale 1.0',
        'landmarks': {
            'vnc_leg_nerve': {'n_somas': int(len(leg_rows)), 'banc_um': leg_um.round(3).tolist()},
            'brain_antennal_nerve': {'n_somas': int(len(ant_rows)), 'banc_um': ant_um.round(3).tolist()},
        },
        'anterior_axis_banc': (anterior / max(np.linalg.norm(anterior), 1e-12)).round(6).tolist(),
        'anterior_separation_um': round(axial, 3),
        'rotation_banc_to_body': rotation.tolist(),
        'scale': scale,
        'banc_center_um': center.round(3).tolist(),
        'dorsal_offset_mm': offset.tolist(),
        'body_node': CNS_BODY_NODE,
        'local_landmarks_mm': {
            'vnc_leg_nerve': local_vnc.round(4).tolist(),
            'brain_antennal_nerve': local_brain.round(4).tolist(),
        },
        'checks': {
            'brain_is_anterior_to_vnc': bool(local_brain[0] < local_vnc[0]),
            'brain_minus_vnc_x_mm': round(float(local_brain[0] - local_vnc[0]), 4),
        },
        'notice': ('LANDMARK registration for visual placement only; NOT anatomical '
                   'registration. Brain/VNC subdivision is a display split with no '
                   'measured boundary. Scale is 1.0 so source distances are preserved.'),
    }


def transform_cns(um: np.ndarray, reg: dict) -> np.ndarray:
    """BANC micrometres -> body-local millimetres (attached to CNS_BODY_NODE).

    Unit conversion happens exactly once, here: um -> mm before the offset is
    added. Mixing units at this step is the single easiest way to place the CNS
    a thousand times too far away, so the scale factor is explicit.
    """
    rotation = np.asarray(reg['rotation_banc_to_body'], dtype=np.float64)
    center = np.asarray(reg['banc_center_um'], dtype=np.float64)
    offset = np.asarray(reg['dorsal_offset_mm'], dtype=np.float64)
    return (um - center) / NM_PER_UM * float(reg['scale']) @ rotation.T + offset


# --------------------------------------------------------------------------
# activity on the real connectome
# --------------------------------------------------------------------------
def propagate_activity(conn, n_neurons, frames, dt=0.001, tau=0.05,
                       gain=1e-2, seed=0, drive_neurons=2000, amplitude=1.0):
    """Bounded leaky linear response over the REAL BANC connectome.

    x' = -x/tau + gain * (W x), W = synapse counts between mapped neurons.
    NOT a measured discharge rate and NOT a calibrated synaptic model: the
    weight units, gain, tau and the seeded input set are engineering choices.
    Returns float32 (frames, n_neurons) in arbitrary response units.
    """
    rng = np.random.default_rng(seed)
    pre, post, w = conn['pre'], conn['post'], conn['syn']
    x = np.zeros(n_neurons, dtype=np.float32)
    x[rng.choice(n_neurons, size=min(drive_neurons, n_neurons), replace=False)] = amplitude
    out = np.empty((frames, n_neurons), dtype=np.float32)
    steps_per_frame = max(1, int(round(1.0 / (frames * dt)))) if frames else 1
    for f in range(frames):
        for _ in range(steps_per_frame):
            drive = np.zeros(n_neurons, dtype=np.float32)
            np.add.at(drive, post, w * x[pre])
            x = x + dt * (-x / tau + gain * drive)
            np.clip(x, 0.0, None, out=x)
        out[f] = x
    return out, {'model': 'linear leaky response on the real BANC synapse matrix',
                 'dt_s': dt, 'tau_s': tau, 'gain': gain, 'seed': seed,
                 'seeded_neurons': int(min(drive_neurons, n_neurons)),
                 'steps_per_frame': steps_per_frame,
                 'notice': ('ILLUSTRATIVE dynamics on REAL connectivity. Not measured '
                            'activity, not a calibrated synaptic model; weights are raw '
                            'synapse counts and gain/tau are engineering choices.')}


# --------------------------------------------------------------------------
# packing helpers
# --------------------------------------------------------------------------
def pack_f32(values: np.ndarray) -> bytes:
    a = np.ascontiguousarray(values, dtype='<f4')
    return a.tobytes()


def pack_u8_quantised(values: np.ndarray):
    """Per-frame min/max quantisation to uint8; returns (bytes, scales)."""
    v = np.asarray(values, dtype=np.float32)
    lo = v.min(axis=1, keepdims=True)
    hi = v.max(axis=1, keepdims=True)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    q = np.clip((v - lo) / span, 0.0, 1.0)
    q = np.rint(q * 255.0).astype(np.uint8)
    scales = np.stack([lo[:, 0], span[:, 0]], axis=1)
    return q.tobytes(), scales


def pack_u32(values: np.ndarray) -> bytes:
    a = np.ascontiguousarray(values, dtype='<u4')
    return a.tobytes()


def stratified_subsample(labels: np.ndarray, target: int, seed: int = 0):
    """Even sample of indices, balanced across non-empty labels.

    Uniform-random sampling would delete small annotated regions entirely, so
    the target is split across label groups in proportion to group size with a
    floor of one index per non-empty group.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    if target >= n:
        return np.arange(n)
    uniq, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    quota = np.maximum(1, np.floor(counts / counts.sum() * target).astype(int))
    picked = []
    for g, q in enumerate(quota):
        rows = np.where(inverse == g)[0]
        if len(rows) <= q:
            picked.append(rows)
        else:
            picked.append(rng.choice(rows, size=q, replace=False))
    out = np.sort(np.concatenate(picked))
    if len(out) > target:
        out = np.sort(rng.choice(out, size=target, replace=False))
    return out
