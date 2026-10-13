"""NEURAL INTERFACE REBUILD: a sub-type-channel sensory front end.

WHAT THIS REPLACES AND WHY
--------------------------
`engine/embodied/adapters.py::MechanoReceptorBank` gives each leg ONE contact scalar and ONE
proprioceptive scalar and adds each to every mechanosensory cell of that leg's nerve
(measured span: 623-808 cells).  Two measured defects:

  1. SIGN BLINDNESS.  The proprioceptive scalar is `mean(|angle - reference|)`, so a flexion and
     an equal extension give the SAME value.  Step 9 measured the consequence on the tier: the
     sign-blind drive produced 6181 spikes for a flexion and 6181 for the matched extension
     (identical, by construction), while a signed drive produced 1187 vs 924.
  2. MODALITY MIXING.  The group that receives the value is the whole leg nerve.  Measured
     composition of the leg mechanosensory population: 2982 bristle (tactile), 342 club
     (vibration), 234 hair plate (joint angle), 162 claw (position), 142 hook (direction),
     62 campaniform (strain), 36 multidendritic.  So ~75% of the cells receiving the
     proprioceptive broadcast are contact/taste cells that have no proprioceptive role.

This module builds the channel front end from the dataset's OWN labels
(`data/banc888/banc_888_meta.feather`: `cell_sub_class` + `cell_function_detailed`), so the
channel assignment is read rather than invented:

    channel            dataset function   quantity handed to the cells
    claw_chordotonal   position           SIGNED tibia joint deviation
    hook_chordotonal   direction          SIGNED tibia angular velocity
    hair_plate         joint_angle        SIGNED multi-joint deviation
    campaniform        mechanical_strain  contact force magnitude (non-negative)
    club_chordotonal   vibro_tactile      NOT MODELLED: emitted as a declared zero with a flag
    bristle            tactile            binary foot contact
    multidendritic     nociception        NOT MODELLED: declared zero with a flag

WHAT IS NOT CLAIMED
-------------------
  * the sign convention (which direction of joint change is "positive" for a hook cell) is a
    DECLARED human decision: the dataset labels `direction` but carries no sign;
  * within a subtype the dataset gives no per-cell difference, so channels are the resolution
    limit (measured: 15 encodings inside `club` share one function label);
  * vibration and nociception have no counterpart physical quantity in this model and are
    emitted as declared zeros rather than proxied silently.

The canonical asset is not touched.  This module is additive: `MechanoReceptorBank` stays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
FEATHER = HERE / "data/banc888/banc_888_meta.feather"
#: CACHED CHANNEL LOOKUP.  The published mapping is exported once by
#: `tools_export_sensory_channel_lookup.py` (which needs pandas/pyarrow) and then read here as
#: plain JSON, because the interpreter that can run MuJoCo in this project does NOT have pandas
#: (measured: venv_body has mujoco but no pandas; venv has pandas but no mujoco).  Reading a
#: cache rather than a dependency keeps the runtime path installable.
LOOKUP = HERE / "outputs/neural_rebuild/sensory_channel_lookup.json"
# MEASURED INTERFACE MISMATCH (fixed): this module first used UPPERCASE leg keys while
# `engine/embodied/adapters.py` publishes LEGS as lowercase ("lf", "lm", ...).  The receptor bank
# then looked up `groups["lf"]` and raised KeyError, because the encoder had filed everything
# under "LF".  The keys are now lowercase, matching the bank, and the lookup file stores
# lowercase too.
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_BODY = {"lf": "front_leg", "lm": "middle_leg", "lh": "hind_leg",
            "rf": "front_leg", "rm": "middle_leg", "rh": "hind_leg"}

# channel -> (banc sub_class substring, dataset function, signed?, modelled?)
CHANNEL_SPEC = {
    "claw_position": ("claw_chordotonal_organ_neuron", "position", True, True),
    "hook_direction": ("hook_chordotonal_organ_neuron", "direction", True, True),
    "hairplate_angle": ("hair_plate_neuron", "joint_angle", True, True),
    "campaniform_strain": ("campaniform_sensillum_neuron", "mechanical_strain", False, True),
    "club_vibration": ("club_chordotonal_organ_neuron", "vibro_tactile", False, False),
    "bristle_touch": ("bristle_neuron", "tactile", False, True),
    "multidendritic_noxious": ("multidendritic_neuron", "nociception", False, False),
}


@dataclass
class ChannelQuantities:
    """The physical quantity each channel is fed, in SIGNED model units."""
    tibia_joint_dev_rad: float = 0.0
    tibia_joint_vel_rad_s: float = 0.0
    multi_joint_dev_rad: float = 0.0
    contact_force_norm: float = 0.0
    foot_contact: float = 0.0


@dataclass
class ChannelConfig:
    current_nA_per_unit: float = 3e-1       # matches step 9's measured usable scale
    reference_force: float = 1.0
    reference_angle_rad: float = 0.30
    reference_velocity_rad_s: float = 10.0
    unmodelled_channels_emit_zero: bool = True
    provenance: str = field(default=(
        "current_nA_per_unit is the declared scale that step 9 measured as the first one to "
        "produce spikes at all (1e-3 and 3e-2 nA gave zero); the reference values are the "
        "project's existing declared normalisers, not measurements"))


class ChannelSensoryEncoder:
    """Per-leg, per-channel sensory front end.

    `groups[leg][channel]` holds the tier-local indices of the cells of that channel on that
    leg.  `encode()` returns a current vector for the tier plus a full record of what each
    channel was fed, so any downstream claim can be traced to a signed physical quantity.
    """

    def __init__(self, channel_groups, n_neurons: int, config: ChannelConfig | None = None):
        self.cfg = config or ChannelConfig()
        self._validate_config()
        if (isinstance(n_neurons, (bool, np.bool_))
                or not isinstance(n_neurons, (int, np.integer)) or n_neurons < 0):
            raise ValueError("n_neurons must be a nonnegative integer, not bool")
        self.n = int(n_neurons)
        self.groups = {}
        for leg in LEGS:
            self.groups[leg] = {}
            for ch in CHANNEL_SPEC:
                raw = channel_groups.get(leg, {}).get(ch, [])
                # Object conversion preserves mixed-list bool/float identity, unlike
                # an integer cast (or ordinary array inference) which can erase it.
                values = np.asarray(raw, dtype=object)
                if values.ndim != 1 or any(
                    isinstance(i, (bool, np.bool_))
                    or not isinstance(i, (int, np.integer)) for i in values.flat
                ):
                    raise ValueError(f"channel group {leg}/{ch} must be a 1D integer index set")
                if any(i < 0 or i >= self.n for i in values):
                    raise ValueError(f"channel group {leg}/{ch} is not a valid index set")
                if len(set(values)) != values.size:
                    raise ValueError(f"channel group {leg}/{ch} contains duplicate indices")
                # Own validated identities; caller mutation must not reroute input.
                idx = np.array(values, dtype=np.int64, copy=True)
                self.groups[leg][ch] = idx
        self.reference_angles = None
        self._rows_by_leg = {}
        self._last = {}

    @staticmethod
    def _require_finite_scalar(value, name):
        if not np.isscalar(value) or not np.isrealobj(value):
            raise ValueError(f"{name} must be a finite scalar")
        try:
            finite = bool(np.isfinite(value))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite scalar") from None
        if not finite:
            raise ValueError(f"{name} must be a finite scalar")

    def _validate_config(self):
        for name in ("current_nA_per_unit", "reference_force", "reference_angle_rad",
                     "reference_velocity_rad_s"):
            value = getattr(self.cfg, name)
            self._require_finite_scalar(value, name)
            if name == "current_nA_per_unit":
                if value < 0:
                    raise ValueError(f"{name} must be nonnegative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")

    # -- reference pose ---------------------------------------------------------
    @staticmethod
    def _canonical_leg(leg):
        if not isinstance(leg, str) or leg.lower() not in LEGS:
            raise ValueError(f"unknown leg: {leg!r}")
        return leg.lower()

    @staticmethod
    def _finite_joint_vector(value, name):
        try:
            if not np.isrealobj(value):
                raise ValueError
            vector = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a nonempty finite 1D vector") from None
        if vector.ndim != 1 or not vector.size or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a nonempty finite 1D vector")
        return vector

    def set_reference_pose(self, joint_angles_rad):
        """Copy a finite reference vector; every successful reset clears row mappings."""
        q = self._finite_joint_vector(joint_angles_rad, "reference pose")
        self.reference_angles = q.copy()
        self._rows_by_leg.clear()

    def assign(self, leg: str, tibia_row: int, trochanter_row: int, multi_rows):
        """Store explicit rows for one leg, supplied by the caller's joint ordering.

        Leg names are case-insensitive. Different legs may intentionally share rows,
        but each leg must be assigned explicitly after every reference-pose reset.
        Trochanter is retained and validated for API compatibility; the current
        quantities use tibia and the explicitly supplied multi-joint rows only.
        """
        leg = self._canonical_leg(leg)
        if self.reference_angles is None:
            raise ValueError("set the reference pose before assigning joint rows")
        size = self.reference_angles.size
        for name, row in (("tibia", tibia_row), ("trochanter", trochanter_row)):
            if (isinstance(row, (bool, np.bool_))
                    or not isinstance(row, (int, np.integer)) or not 0 <= row < size):
                raise ValueError(f"{name} row must be an integer in reference bounds")
        values = np.asarray(multi_rows, dtype=object)
        if values.ndim != 1 or not values.size or any(
            isinstance(row, (bool, np.bool_))
            or not isinstance(row, (int, np.integer)) or not 0 <= row < size
            for row in values.flat
        ):
            raise ValueError("multi rows must be nonempty 1D integers in reference bounds")
        if len(set(values)) != values.size:
            raise ValueError("multi rows contain duplicate indices")
        self._rows_by_leg[leg] = {
            "tibia": int(tibia_row), "trochanter": int(trochanter_row),
            "multi": np.array(values, dtype=np.int64, copy=True),
        }

    # -- encoding ---------------------------------------------------------------
    def quantities(self, leg: str, joint_angles_rad, joint_velocity_rad_s,
                   contact_force_norm: float, foot_contact: float) -> ChannelQuantities:
        leg = self._canonical_leg(leg)
        if self.reference_angles is None:
            raise ValueError("set_reference_pose first")
        if leg not in self._rows_by_leg:
            raise ValueError(f"assign joint rows for {leg} after setting the reference pose")
        q = self._finite_joint_vector(joint_angles_rad, "joint angles")
        w = self._finite_joint_vector(joint_velocity_rad_s, "joint velocity")
        if q.shape != self.reference_angles.shape or w.shape != self.reference_angles.shape:
            raise ValueError("joint angles and velocity must exactly match reference shape")
        self._require_finite_scalar(contact_force_norm, "contact_force_norm")
        self._require_finite_scalar(foot_contact, "foot_contact")
        rows = self._rows_by_leg[leg]
        with np.errstate(over="ignore", invalid="ignore"):
            dev = q - self.reference_angles
            multi = float(dev[rows["multi"]].mean())
        if not np.all(np.isfinite(dev)) or not np.isfinite(multi):
            raise ValueError("joint deviations must remain finite")
        tib = float(dev[rows["tibia"]])
        tib_vel = float(w[rows["tibia"]])
        return ChannelQuantities(tibia_joint_dev_rad=tib,
                                 tibia_joint_vel_rad_s=tib_vel,
                                 multi_joint_dev_rad=multi,
                                 contact_force_norm=float(contact_force_norm),
                                 foot_contact=float(foot_contact))

    def encode(self, per_leg: dict) -> tuple:
        """`per_leg[leg]` is a ChannelQuantities.  Returns (current_nA, record)."""
        # Config and quantities are mutable dataclasses: validate at the boundary
        # as well as construction, including quantities for currently empty groups.
        self._validate_config()
        cfg = self.cfg
        current = np.zeros(self.n)
        record = {"per_leg": {}, "unmodelled_channels": [], "scale_nA_per_unit":
                  cfg.current_nA_per_unit}
        for leg in LEGS:
            q = per_leg.get(leg)
            if q is None:
                continue
            for name in ("tibia_joint_dev_rad", "tibia_joint_vel_rad_s",
                         "multi_joint_dev_rad", "contact_force_norm", "foot_contact"):
                self._require_finite_scalar(getattr(q, name), f"{leg}/{name}")
            vals = {
                "claw_position": q.tibia_joint_dev_rad / cfg.reference_angle_rad,
                "hook_direction": q.tibia_joint_vel_rad_s / cfg.reference_velocity_rad_s,
                "hairplate_angle": q.multi_joint_dev_rad / cfg.reference_angle_rad,
                "campaniform_strain": q.contact_force_norm / cfg.reference_force,
                "bristle_touch": q.foot_contact,
                "club_vibration": 0.0,
                "multidendritic_noxious": 0.0,
            }
            leg_record = {}
            for ch, v in vals.items():
                self._require_finite_scalar(v, f"{leg}/{ch} normalized quantity")
                _, fn, signed, modelled = CHANNEL_SPEC[ch]
                if not modelled:
                    record["unmodelled_channels"].append({"leg": leg, "channel": ch,
                                                          "dataset_function": fn,
                                                          "value": 0.0})
                idx = self.groups[leg][ch]
                if idx.size:
                    with np.errstate(over="ignore", invalid="ignore"):
                        current[idx] += cfg.current_nA_per_unit * float(v)
                    if not np.all(np.isfinite(current[idx])):
                        raise ValueError(f"{leg}/{ch} produces nonfinite current")
                leg_record[ch] = {"value": float(v), "signed": signed, "modelled": modelled,
                                  "n_cells": int(idx.size), "dataset_function": fn}
            record["per_leg"][leg] = leg_record
        return current, record

    # -- inspection -------------------------------------------------------------
    def coverage(self) -> dict:
        out = {}
        for leg in LEGS:
            out[leg] = {ch: int(self.groups[leg][ch].size) for ch in CHANNEL_SPEC}
            out[leg]["_total"] = int(sum(self.groups[leg][ch].size for ch in CHANNEL_SPEC))
        return out


def load_lookup(path: Path = LOOKUP) -> dict:
    """Read the exported root_id -> (leg, channel) mapping.  Raises if it is absent, because a
    silenter fallback would let a partially-built partition look complete."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; regenerate it with tools_export_sensory_channel_lookup.py "
            f"(needs pandas, i.e. venv/bin/python)")
    d = json.loads(p.read_text())
    return d


def channel_index_map(n_neurons: int, tier_root_ids, *, lookup_path: Path = LOOKUP):
    """Partition tier-local indices into (leg, channel) from the EXPORTED lookup.

    `tier_root_ids` is the tier's root-id vector in tier-local order (the tier publishes it).
    Returns (channel_groups, stats).  A cell the tier does not contain simply does not appear:
    a partial partition is REPORTED, never padded.

    The dataset read itself lives in `tools_export_sensory_channel_lookup.py` because reading
    the feather needs pandas, and the interpreter that runs MuJoCo here does not have pandas
    (measured: venv_body has mujoco without pandas; venv has pandas without mujoco).
    """
    lu = load_lookup(lookup_path)
    by_root = lu["leg_channel_by_root"]
    local_of_root = {int(r): i for i, r in enumerate(np.asarray(tier_root_ids))}
    groups = {leg: {ch: [] for ch in CHANNEL_SPEC} for leg in LEGS}
    matched = unmatched = 0
    for root_s, pair in by_root.items():
        leg, ch = pair[0], pair[1]
        if leg not in groups or ch not in groups[leg]:
            continue
        li = local_of_root.get(int(root_s))
        if li is None:
            unmatched += 1
            continue
        groups[leg][ch].append(li)
        matched += 1
    stats = {"cells_placed": matched, "leg_sensory_cells_outside_tier": unmatched,
             "n_neurons": int(n_neurons),
             "n_dataset_cells_with_a_channel": len(by_root),
             "lookup": str(lookup_path),
             "note": "a cell that is in the dataset but not in the tier cannot be driven; that "
                     "count is reported rather than redistributed"}
    return groups, stats
