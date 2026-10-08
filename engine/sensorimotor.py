"""engine.sensorimotor -- the sensorimotor interface to a virtual environment.

WHY THIS EXISTS
---------------
The whole-brain LIF run showed that the real FlyWire connectome is either
silent or saturated: with no spontaneous activity and no input, a purely
recurrent network has nothing to propagate.  That is not a parameter bug, it is
the absence of a BODY AND ENVIRONMENT.  A brain fires because it is driven by
sensation and because its output does something.

So before any attempt at realistic firing, the brain needs named ports through
which an environment can write sensation and read action.  Those ports do not
have to be invented: FlyWire annotates them, and this module reads them from
the data rather than asserting them:

  inputs  (flow == "afferent", super_class == "sensory"), by class:
      visual            vision
      mechanosensory    touch / proprioception
      olfactory         smell
      gustatory         taste
      hygrosensory      humidity
      thermosensory     temperature
  outputs (flow == "efferent"):
      descending        brain -> ventral nerve cord motor commands
      motor             motor neurons
      endocrine         endocrine output

Every group is a list of INDICES into the connectome's neuron ordering, so the
environment can inject current into exactly those neurons and read exactly
those readouts.

HONEST LIMITS
-------------
* Annotating a neuron as "visual" does not create a visual system.  This module
  provides the wiring boundary; the environment (retina, optics, body) does not
  exist yet, and nothing here claims the stimulated brain "sees".
* A neuron present in the classification table may be absent from the
  connection table (no connection above the 5-synapse threshold).  Those cannot
  be driven or read through the connectome, and the unmatched counts are
  reported rather than hidden.
"""

from __future__ import annotations

import csv
import gzip
import json
import os

import numpy as np

__all__ = ["SensorimotorInterface", "MODALITY_CLASSES", "OUTPUT_SUPERCLASSES"]

# sensory modality <- (super_class, class) as annotated by the FlyWire
# hierarchical classification (Schlegel et al., via the `classification`
# data product).  These strings are the DATA's vocabulary, not ours.
MODALITY_CLASSES = {
    "visual":          "vision (photoreceptor-driven; the retina itself is not "
                       "in this dataset, so these are the brain-side targets)",
    "mechanosensory":  "touch / mechanosensation / proprioception",
    "olfactory":       "smell",
    "gustatory":       "taste",
    "hygrosensory":    "humidity sensing",
    "thermosensory":   "temperature sensing",
}

OUTPUT_SUPERCLASSES = {
    "descending": "brain -> ventral nerve cord: the main motor command channel",
    "motor":      "motor neurons",
    "endocrine":  "endocrine / hormonal output",
}


class SensorimotorInterface:
    """Named sensory-input and motor-output ports, built from real annotations."""

    def __init__(self, classification_gz, root_ids):
        """
        classification_gz : path to fafb_classification.csv.gz
        root_ids          : the connectome's neuron ordering (n,) of root ids
        """
        self.root_ids = np.asarray(root_ids)
        self.index_of = {int(r): i for i, r in enumerate(self.root_ids)}
        n = self.root_ids.size

        # raw annotation tallies (over the WHOLE classification table)
        self.table_counts = {"flow": {}, "super_class": {}, "sensory_class": {},
                             "nerve": {}}
        self.total_rows = 0

        groups = {m: [] for m in MODALITY_CLASSES}
        outputs = {o: [] for o in OUTPUT_SUPERCLASSES}
        # neurons that matched the connectome, and those that did not
        matched = {m: 0 for m in MODALITY_CLASSES}
        missing = {m: 0 for m in MODALITY_CLASSES}
        matched_o = {o: 0 for o in OUTPUT_SUPERCLASSES}
        missing_o = {o: 0 for o in OUTPUT_SUPERCLASSES}
        all_other = []
        out_flow = {o: {} for o in OUTPUT_SUPERCLASSES}

        with gzip.open(classification_gz, "rt", errors="replace") as fh:
            for row in csv.DictReader(fh):
                self.total_rows += 1
                flow = row.get("flow", "")
                sc = row.get("super_class", "")
                cl = row.get("class", "")
                nerve = row.get("nerve", "") or "(none)"
                self.table_counts["flow"][flow] = \
                    self.table_counts["flow"].get(flow, 0) + 1
                self.table_counts["super_class"][sc] = \
                    self.table_counts["super_class"].get(sc, 0) + 1
                self.table_counts["nerve"][nerve] = \
                    self.table_counts["nerve"].get(nerve, 0) + 1
                if sc == "sensory":
                    self.table_counts["sensory_class"][cl] = \
                        self.table_counts["sensory_class"].get(cl, 0) + 1

                idx = self.index_of.get(int(row["root_id"]))
                in_conn = idx is not None
                hit = False
                if flow == "afferent" and sc == "sensory" and cl in MODALITY_CLASSES:
                    hit = True
                    if in_conn:
                        groups[cl].append(idx)
                        matched[cl] += 1
                    else:
                        missing[cl] += 1
                # NOT filtered on flow == "efferent": 4 of the 110 `motor`
                # neurons are annotated flow == "intrinsic", so requiring
                # efferent silently drops them.  We include every neuron whose
                # super_class is a motor-output class and report the flow
                # breakdown so the discrepancy is visible rather than hidden.
                if sc in OUTPUT_SUPERCLASSES:
                    hit = True
                    out_flow[sc][flow] = out_flow[sc].get(flow, 0) + 1
                    if in_conn:
                        outputs[sc].append(idx)
                        matched_o[sc] += 1
                    else:
                        missing_o[sc] += 1
                if not hit and in_conn:
                    all_other.append(idx)

        self.inputs = {k: np.asarray(sorted(v), dtype=np.int64)
                       for k, v in groups.items()}
        self.outputs = {k: np.asarray(sorted(v), dtype=np.int64)
                        for k, v in outputs.items()}
        self.matched = matched
        self.missing_from_connectome = missing
        self.matched_outputs = matched_o
        self.missing_outputs = missing_o
        self.output_flow_breakdown = {k: dict(v) for k, v in out_flow.items()}

        # everything not classified as sensory input or motor output
        spur = np.zeros(n, dtype=bool)
        for arr in list(self.inputs.values()) + list(self.outputs.values()):
            spur[arr] = True
        self.intrinsic = np.flatnonzero(~spur)
        self.n_neurons = n

    # ------------------------------------------------------------------
    def all_input_indices(self):
        return np.unique(np.concatenate(list(self.inputs.values()))) \
            if self.inputs else np.zeros(0, dtype=np.int64)

    def all_output_indices(self):
        return np.unique(np.concatenate(list(self.outputs.values()))) \
            if self.outputs else np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    def make_drive(self, signals, shape_n=None, default=0.0):
        """Build a per-neuron external-current array in mV.

        `signals` maps a modality name (e.g. "visual") to a drive level in mV.
        The value is applied to every neuron in that modality's group.  Later
        versions should replace this with a receptor model; this is the plug
        point for a virtual environment.
        """
        n = shape_n or self.n_neurons
        i_ext = np.full(n, float(default))
        for name, level in signals.items():
            arr = self.inputs.get(name)
            if arr is None or arr.size == 0:
                raise KeyError(
                    f"unknown or empty sensory modality {name!r}; "
                    f"available: {sorted(k for k, v in self.inputs.items() if v.size)}")
            i_ext[arr] = float(level)
        return i_ext

    def read_activity(self, spikes, dt_ms=1.0):
        """Read motor output from a spike vector.  Returns {name: rate in Hz}."""
        sp = np.asarray(spikes).astype(np.float64)
        out = {}
        for name, arr in self.outputs.items():
            if arr.size == 0:
                out[name] = 0.0
                continue
            out[name] = float(sp[arr].mean()) / (dt_ms / 1000.0)
        return out

    def read_population(self, spikes):
        """Also report the intrinsic population rate."""
        sp = np.asarray(spikes).astype(np.float64)
        return {"intrinsic_rate": float(sp[self.intrinsic].mean()) if self.intrinsic.size
                else 0.0}

    # ------------------------------------------------------------------
    def verify(self):
        """Check the groups against the annotation table, exactly."""
        exp = self.table_counts["sensory_class"]
        problems = []
        for m in MODALITY_CLASSES:
            got = int(self.inputs[m].size) + int(self.missing_from_connectome[m])
            want = int(exp.get(m, 0))
            if got != want:
                problems.append(f"{m}: {got} groups vs {want} in table")
            if self.inputs[m].size and np.unique(self.inputs[m]).size != self.inputs[m].size:
                problems.append(f"{m}: duplicate neuron indices")
        for o in OUTPUT_SUPERCLASSES:
            got = int(self.outputs[o].size) + int(self.missing_outputs[o])
            want = int(self.table_counts["super_class"].get(o, 0))
            if got != want:
                problems.append(f"{o}: {got} vs {want} in table")
        # groups must not overlap each other
        seen = {}
        for m, arr in self.inputs.items():
            for i in arr.tolist():
                if i in seen:
                    problems.append(f"neuron {i} in both {seen[i]} and {m}")
                seen[i] = m
        return {"ok": not problems, "problems": problems,
                "n_neurons": int(self.n_neurons),
                "n_input_neurons": int(self.all_input_indices().size),
                "n_output_neurons": int(self.all_output_indices().size),
                "n_intrinsic": int(self.intrinsic.size)}

    def describe(self):
        lines = [f"SensorimotorInterface  ({self.n_neurons:,} neurons in connectome order)",
                 f"  注释总行数 {self.total_rows:,}", ""]
        lines.append("  感觉输入端口（flow=afferent, super_class=sensory）：")
        for m, desc in MODALITY_CLASSES.items():
            lines.append(f"    {m:16s} {self.inputs[m].size:>6,} 个（连接组内）"
                         f" + {self.missing_from_connectome[m]:>5,} 个（无连接，不可驱动）"
                         f"   = {self.inputs[m].size + self.missing_from_connectome[m]:>6,}"
                         f"   [{desc}]")
        lines.append("")
        lines.append("  运动输出端口（flow=efferent）：")
        for o, desc in OUTPUT_SUPERCLASSES.items():
            lines.append(f"    {o:16s} {self.outputs[o].size:>6,} 个（连接组内）"
                         f" + {self.missing_outputs[o]:>5,} 个   = "
                         f"{self.outputs[o].size + self.missing_outputs[o]:>6,}"
                         f"   [{desc}]")
        lines.append("")
        lines.append(f"  既非感觉输入也非运动输出的内部神经元: {self.intrinsic.size:,}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "n_neurons": int(self.n_neurons),
            "annotation_rows": int(self.total_rows),
            "inputs": {m: {"in_connectome": int(self.inputs[m].size),
                           "missing_from_connectome": int(self.missing_from_connectome[m]),
                           "description": MODALITY_CLASSES[m]}
                       for m in MODALITY_CLASSES},
            "outputs": {o: {"in_connectome": int(self.outputs[o].size),
                            "missing_from_connectome": int(self.missing_outputs[o]),
                            "description": OUTPUT_SUPERCLASSES[o]}
                        for o in OUTPUT_SUPERCLASSES},
            "n_intrinsic": int(self.intrinsic.size),
            "table_counts": self.table_counts,
            "note": "端口来自 FlyWire 的 flow / super_class / class 标注，"
                    "不是我们发明的分类。",
        }

    def save_groups(self, path):
        np.savez_compressed(
            path,
            **{f"in_{k}": v for k, v in self.inputs.items()},
            **{f"out_{k}": v for k, v in self.outputs.items()},
            intrinsic=self.intrinsic)
        return path
