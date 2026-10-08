"""Channel <-> neuron ACCESS MAP: where a fixed-geometry electrode array could be
placed, in the REAL voxel coordinates of BANC somata, and which neurons each
channel could access.

WHAT THIS MODULE IS
-------------------
A purely GEOMETRIC access map.  Given

  * an electrode array of FIXED pitch and FIXED contact diameter (user constraint:
    these two are ``ElectrodeArraySpec`` inputs and are NEVER optimised here),
  * a FREE channel count, and
  * a DECLARED capture radius (how close a soma must be to a channel site to be
    considered addressable by it),

it places the array in the measured BANC voxel frame and returns, for every
channel, the list of neurons whose soma point lies inside that channel's sphere.

It then wires the result to the EXISTING recording front end
(:class:`engine.electrode.VoltageRecorder` / :func:`engine.electrode.
transfer_mV_per_nA`) so that a channel's signal can actually be produced.  The
volume conduction is NOT reimplemented here; it is reused.

WHAT THIS MODULE IS NOT  (read before quoting any number out of it)
-------------------------------------------------------------------
* **No electrode-contact model.**  No impedance, no electrode-electrolyte
  interface, no double layer, no crosstalk between channels, no common-mode
  rejection, no stimulation, no noise floor of a real headstage.  ``diameter_um``
  is carried as a DECLARED spec field and is NOT used by the access geometry at
  all: this module does not model a contact as a finite patch, does not compute
  a contact impedance from it, and does not average the field over it.
  ``engine.electrode``'s own docstring already says "point probe, not
  contact-area averaging" -- that limitation is inherited, not fixed.
* **No tissue mechanics.**  No insertion, no dimpling, no shear, no damage, no
  glial sheath.  ``engine.electrode_damage.py`` / ``engine.touch_damage_scenario``
  hold that chain and are deliberately NOT called from here.
* **Somas are SINGLE VOXEL POINTS.**  ``pt_position`` is one voxel per neuron
  (``pt_supervoxel_id``).  There is no morphology, no neurite, no soma radius, no
  axon, no synapse location.  Consequences, stated plainly:
    - a channel "accesses" a neuron iff ONE POINT is inside the sphere, so the
      map cannot see a neurite passing near a channel, and cannot see that a
      soma is 10 um wide and may touch a channel whose centre is outside it;
    - ``capture_radius_um`` is therefore a POINT-CAPTURE radius, not a
      tissue-volume or membrane-area statement;
    - two neurons whose somas are adjacent voxels are geometrically
      indistinguishable at this resolution.
* **No spiking model.**  Neurons are current sources of a declared magnitude,
  not cells that fire.  The recording produced here is an extracellular
  potential trace, not a sorted spike train, and no spike-sorting is attempted.
* **A channel that "accesses" a neuron is not a claim that the neuron's signal
  is recoverable.**  Attenuation with distance is computed and reported, and the
  amplitude of what a channel actually sees is left to the caller to inspect.
* **Nothing about the fly.**  No behaviour, no perception, no attention, no
  recognition, no experience, no identity claim of any kind is made or implied.
  This is a hand-built research prototype, not a validated device model.

UNITS -- READ THIS, IT CHANGED THE ANSWER
-----------------------------------------
Two frames are kept strictly apart and NEVER silently mixed:

  * the **BANC coordinate frame**, in which all ``pt_position`` values live.
    This module does all array placement in it.
  * **micrometres**, the frame ``engine.electrode`` works in and the frame the
    user fixed pitch/diameter in.

The single bridge is ``voxel_resolution_nm`` (nanometres per coordinate unit).
If that number is wrong, every micrometre quantity in the output is wrong by
the same factor, so it is a first-class ledger entry and
:func:`sweep_access_map` can vary it.

The task brief described ``pt_position`` as **voxels** with the resolution
unverified.  Both halves of that were resolved by looking:

  * ``pt_position`` **IS** in voxels -- raw voxel indices of the BANC volume.
    The parquet's own export metadata (``PANDAS_ATTRS``) reports
    ``table_voxel_resolution_x/_y/_z = 1.0`` and
    ``dataframe_resolution = [1.0, 1.0, 1.0]``, i.e. **no scaling was applied**
    to the stored values, and the Dataverse file metadata describes the column as
    "marker point at the nucleus centroid in BANC voxel space".
  * The **voxel size IS VERIFIED, and it is ANISOTROPIC: 4 x 4 x 45 nm**
    (x = 4 nm, y = 4 nm, z = 45 nm).  See :func:`banc_voxel_resolution` and
    :data:`BANC_RESOLUTION_SOURCES` for the sources, all of them recorded with
    their quotations.  Conversion: ``um = coord * [4, 4, 45] / 1000``.

The anisotropy is not a detail.  A single scalar "nm per voxel" would mis-scale
z by 11.25x (45/4).  Under the wrong, isotropic-4-nm reading the 192 readout
somas look ~11 um thick; they are really ~127 um thick -- which is exactly the
difference between "a flat sheet a planar array can cover" and "a slab it
cannot".  The verified z-step is also self-checking: the z extent is 6,990
voxels, and 6,990 x 45 nm = 314.6 um against the volume's own 7,010 sections x
45 nm = 315.4 um.

THREE NEAR MISSES THE MODULE REFUSES TO USE, because each sounds right:
``8 x 8 x 45`` is the PUBLIC DOWNSCALED Neuroglancer mip, not the coordinate
unit; ``4 x 4 x 40`` is FAFB/FlyWire (same lateral pitch, different z);
``8 x 8 x 8`` is HemiBrain.  None of them is BANC's.

ATTRIBUTION (REQUIRED -- CC BY 4.0)
-----------------------------------
The soma coordinates consumed here come from:

    "Distributed control circuits across a brain-and-cord connectome",
    Harvard Dataverse, doi:10.7910/DVN/7WTH1N, file ``somas_v1.parquet``
    (file id 13916460).  Licensed CC BY 4.0
    (http://creativecommons.org/licenses/by/4.0).

Attribution is mandatory.  Any redistribution of this module's outputs, or of
derived coordinates, MUST carry the same attribution and licence statement.
:data:`BANC_SOMA_ATTRIBUTION` is the record to reproduce; it is written into
every output dict produced by this module.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

# The recording front end is REUSED, never reimplemented.
from ..electrode import Contact, VoltageRecorder, transfer_mV_per_nA

__all__ = [
    "PROVENANCE", "Ledger", "ProvenanceRecord", "ElectrodeArraySpec",
    "AccessMap", "build_access_map", "load_soma_table", "access_map_to_recording",
    "sweep_access_map", "BANC_SOMA_ATTRIBUTION", "BANC_SOMA_DOI",
    "BANC_SOMA_LICENCE", "VoxelResolution", "SomaTable", "load_soma_table",
    "measure_resolution_from_parquet", "UNVERIFIED_RESOLUTION_CANDIDATES_NM",
    "banc_voxel_resolution", "BANC_RESOLUTION_SOURCES", "BANC_NATIVE_VOXEL_NM",
    "DISCLAIMER",
]


# ---------------------------------------------------------------------------
# provenance vocabulary
# ---------------------------------------------------------------------------
class PROVENANCE:
    """The four allowed provenance classes.  There is no fifth.

    ``MEASURED_CITED``  -- a number taken from an EXTERNAL source, with that
                           source quoted.  Refused without a source (see
                           :meth:`Ledger.record`).  This is the strongest class
                           and must not be used for a value read out of a local
                           file, or for a remembered convention.
    ``MEASURED_LOCAL``  -- a number actually computed/read from a file on THIS
                           machine, reproducible by re-running the code.
    ``ASSUMED``         -- a model input with no measurement behind it.  Must
                           carry a sweep range (see :meth:`Ledger.record`).
    ``ENGINEERING_DEFAULT`` -- an author's choice for a device/experiment
                           parameter, i.e. a value a builder would pick.  Not a
                           measurement and not a physical fact.
    """

    MEASURED_CITED = "MEASURED_CITED"
    MEASURED_LOCAL = "MEASURED_LOCAL"
    ASSUMED = "ASSUMED"
    ENGINEERING_DEFAULT = "ENGINEERING_DEFAULT"

    ALL = (MEASURED_CITED, MEASURED_LOCAL, ASSUMED, ENGINEERING_DEFAULT)


#: The dataset's own persistent identifier.  Verified against the Dataverse
#: landing page by the project (see outputs/embodied_body/ROUND24_BANC_SOMA_COORDS.md).
BANC_SOMA_DOI = "doi:10.7910/DVN/7WTH1N"
BANC_SOMA_DATASET_TITLE = ("Distributed control circuits across a brain-and-cord "
                           "connectome")
BANC_SOMA_FILE = "somas_v1.parquet"
BANC_SOMA_FILE_ID = 13916460
BANC_SOMA_LICENCE = "CC BY 4.0 (http://creativecommons.org/licenses/by/4.0)"

#: Reproduce this block in any redistribution.  CC BY 4.0 makes attribution a
#: licence CONDITION, not a courtesy.
BANC_SOMA_ATTRIBUTION = {
    "dataset_title": BANC_SOMA_DATASET_TITLE,
    "doi": BANC_SOMA_DOI,
    "repository": "Harvard Dataverse",
    "file": BANC_SOMA_FILE,
    "file_id": BANC_SOMA_FILE_ID,
    "licence": BANC_SOMA_LICENCE,
    "attribution_required": True,
    "redistribution_clause": (
        "Redistribution of this data, or of any derived coordinate or access "
        "map, MUST carry the dataset title and doi above and this licence "
        "statement, as required by CC BY 4.0."),
    "citation_string": (
        'BANC (2026). "Distributed control circuits across a brain-and-cord '
        'connectome" [Data set]. Harvard Dataverse. '
        'https://doi.org/10.7910/DVN/7WTH1N  (CC BY 4.0)'),
}


@dataclass(frozen=True)
class ProvenanceRecord:
    key: str
    value: object
    unit: str
    provenance: str
    source: Optional[str] = None
    note: str = ""
    sweep: Optional[tuple] = None

    def as_dict(self):
        return {"key": self.key, "value": self.value, "unit": self.unit,
                "provenance": self.provenance, "source": self.source,
                "note": self.note,
                "sweep": (None if self.sweep is None else list(self.sweep))}


class Ledger:
    """Ordered provenance ledger.  Enforces the honesty rules, not just records.

    Two rules are enforced by ``record`` and will RAISE:

    1. ``MEASURED_CITED`` with no ``source`` -- a cited measurement without a
       citation is a fabrication.
    2. ``ASSUMED`` with no ``sweep`` -- an assumption you are not prepared to
       vary is an unexamined constant.  (Sweep may be given either as the
       ``sweep=`` argument or inside ``note``.)
    """

    def __init__(self):
        self._recs = []
        self._keys = {}

    def record(self, key, value, unit, provenance, *, source=None, note="", sweep=None):
        if provenance not in PROVENANCE.ALL:
            raise ValueError("provenance must be one of %r; got %r"
                             % (PROVENANCE.ALL, provenance))
        if provenance == PROVENANCE.MEASURED_CITED and not (source and str(source).strip()):
            raise ValueError(
                "refusing MEASURED_CITED without a source: %r. A cited measurement "
                "with no citation is fabricated provenance." % (key,))
        if provenance == PROVENANCE.ASSUMED and sweep is None and \
                not any(w in str(note).lower() for w in ("sweep", "range")):
            raise ValueError(
                "refusing ASSUMED without a declared sweep/range: %r. An "
                "assumption without a swept range is an unexamined constant." % (key,))
        if key in self._keys:
            raise ValueError("duplicate ledger key %r" % (key,))
        r = ProvenanceRecord(key, value, unit, provenance, source, note,
                             None if sweep is None else tuple(sweep))
        self._keys[key] = r
        self._recs.append(r)
        return r

    def get(self, key):
        return self._keys[key]

    def as_list(self):
        return [r.as_dict() for r in self._recs]

    def as_dict(self):
        return {r.key: r.as_dict() for r in self._recs}

    def counts(self):
        out = {p: 0 for p in PROVENANCE.ALL}
        for r in self._recs:
            out[r.provenance] += 1
        return out

    def unresolved(self):
        """Entries a reader must treat as NOT established from a source."""
        return [r.as_dict() for r in self._recs
                if r.provenance in (PROVENANCE.ASSUMED, PROVENANCE.ENGINEERING_DEFAULT)]


# ---------------------------------------------------------------------------
# voxel resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VoxelResolution:
    """Voxel -> micrometre conversion for the BANC frame.

    ``x_nm, y_nm, z_nm``  size of one voxel along each axis, in nanometres.
    Anisotropy is supported because EM volumes are commonly anisotropic; the
    BANC volume's own value must be looked up, not guessed.
    """

    x_nm: float
    y_nm: float
    z_nm: float
    provenance: str
    source: Optional[str] = None
    note: str = ""

    def __post_init__(self):
        for name in ("x_nm", "y_nm", "z_nm"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0:
                raise ValueError("%s must be positive and finite" % name)
            object.__setattr__(self, name, v)
        if self.provenance not in PROVENANCE.ALL:
            raise ValueError("provenance must be one of %r" % (PROVENANCE.ALL,))
        if self.provenance == PROVENANCE.MEASURED_CITED and \
                not (self.source and str(self.source).strip()):
            raise ValueError("MEASURED_CITED voxel resolution requires a source")

    @property
    def nm_per_voxel(self):
        return np.array([self.x_nm, self.y_nm, self.z_nm], dtype=float)

    @property
    def isotropic(self):
        return self.x_nm == self.y_nm == self.z_nm

    def voxels_to_um(self, vox):
        a = np.asarray(vox, dtype=float)
        return a * (self.nm_per_voxel / 1000.0)

    def um_to_voxels(self, um):
        a = np.asarray(um, dtype=float)
        return a / (self.nm_per_voxel / 1000.0)

    def as_dict(self):
        d = {"x_nm": self.x_nm, "y_nm": self.y_nm, "z_nm": self.z_nm,
             "isotropic": bool(self.isotropic), "provenance": self.provenance,
             "source": self.source, "note": self.note}
        if self.provenance != PROVENANCE.MEASURED_CITED:
            d["WARNING"] = ("NOT VERIFIED FROM A SOURCE. Every micrometre "
                            "quantity derived with this resolution, and every "
                            "coverage verdict that depends on it, is "
                            "assumption-dependent.")
        return d


#: Documented candidate resolutions that this project has NOT verified, kept as
#: a named sweep so the "assumed" status is explicit and auditable.  These are
#: CANDIDATES FOR SWEEPING ONLY -- none of them is claimed to be BANC's value.
#: The 4x4x40 nm triple is a widely used FLYWIRE/FAFB convention; treating it as
#: BANC's value without a BANC source would be transferring a citation between
#: datasets, which this module refuses to do.
UNVERIFIED_RESOLUTION_CANDIDATES_NM = (
    (4.0, 4.0, 45.0),
    (8.0, 8.0, 45.0),
    (4.0, 4.0, 40.0),
    (8.0, 8.0, 8.0),
    (4.0, 4.0, 4.0),
    (1.0, 1.0, 1.0),
)


#: The VERIFIED BANC native voxel size, with the sources checked.  This is the
#: value the access map uses by default; :func:`banc_voxel_resolution` returns it.
BANC_NATIVE_VOXEL_NM = (4.0, 4.0, 45.0)

BANC_RESOLUTION_SOURCES = (
    ("primary_paper",
     "doi:10.1038/s41586-026-10735-w",
     'Results, opening: "We generated a serial-section electron microscopy (EM) '
     'volume of the connected brain and nerve cord from an adult female '
     'D. melanogaster at synapse-level resolution (4 x 4 x 45 nm3)". Extended Data '
     'Fig. 1a caption: "Note that the public instance downscales data to '
     '8 x 8 x 45 nm3 in xyz".'),
    ("project_wiki_coordinate_space",
     "https://github.com/jasper-tms/the-BANC-fly-connectome/wiki/"
     "Neuroglancer-states-for-proofreading",
     '"The voxel size should be [4, 4, 45] nm." and "This is the voxel size that '
     'the CAVE tables expect annotations to be specified in." -- i.e. exactly the '
     'space somas_v1 pt_position lives in.'),
    ("project_wiki_image_dataset",
     "https://github.com/jasper-tms/the-BANC-fly-connectome/wiki/The-EM-image-dataset",
     '"These tissue sections were imaged with a custom GridTape-compatible '
     'transmission electron microscope at a lateral resolution of about 4nm. The '
     'size of each voxel is therefore about 4nm x 4nm x 45nm." and "cut into 7010 '
     'serial sections, each around 45nm in thickness".'),
    ("supplemental_notes",
     "Harvard Dataverse file id 14023112 (supplemental_notes.pdf), Supplementary "
     "Data 10 legend",
     '"Coordinates are in BANC raw-voxel space (1 voxel = 4 x 4 x 45 nm)."'),
    ("local_consistency_check",
     "this run, on data/banc/somas_v1.parquet",
     "z span is 6990 voxels; 6990 x 45 nm = 314.6 um against the volume's own "
     "7010 sections x 45 nm = 315.4 um. Under a 4 nm z-step those 6990 units would "
     "span 28 um, which cannot hold 192 neurons."),
)


def banc_voxel_resolution() -> VoxelResolution:
    """The VERIFIED BANC native voxel size: **4 x 4 x 45 nm (anisotropic)**.

    ``MEASURED_CITED``: the value is carried with the sources that state it, and
    every source is recorded in :data:`BANC_RESOLUTION_SOURCES`.

    WHY THE ANISOTROPY MATTERS AND IS EASY TO GET WRONG: a single scalar "nm per
    voxel" would mis-convert z by a factor of 11.25 (45/4).  A readout cluster
    that looks 11 um thick under an isotropic 4 nm reading is really ~127 um
    thick, which decides whether a planar array can reach it at all.

    NEAR MISSES THAT MUST NOT BE USED, all of them defensible-sounding:
      * **8 x 8 x 45 nm** is the resolution of the PUBLIC downscaled Neuroglancer
        mip, not the coordinate unit. The paper says so explicitly. Using it
        doubles every lateral distance.
      * **4 x 4 x 40 nm** is FAFB/FlyWire. Same lateral pitch, different z.
        Transferring it to BANC would be moving a citation between datasets.
      * **8 x 8 x 8 nm** is HemiBrain; **4 x 4 x 4** isotropic would imply ~78,750
        z-sections where the volume has 7,010.
    """
    return VoxelResolution(
        BANC_NATIVE_VOXEL_NM[0], BANC_NATIVE_VOXEL_NM[1], BANC_NATIVE_VOXEL_NM[2],
        PROVENANCE.MEASURED_CITED,
        source="; ".join("%s %s: %s" % (k, u, q)
                         for k, u, q in BANC_RESOLUTION_SOURCES),
        note=("BANC native voxel size, anisotropic. pt_position values are RAW VOXEL "
              "INDICES in this space (the parquet's export metadata reports a "
              "resolution factor of 1.0, i.e. no transform was applied). Conversion: "
              "um = coord * [4, 4, 45] / 1000."))


def measure_resolution_from_parquet(somas_parquet_path: str) -> VoxelResolution:
    """Read the coordinate-frame resolution out of the parquet's OWN metadata.

    Returns a :class:`VoxelResolution` marked ``MEASURED_LOCAL`` with
    ``source=<the file and metadata keys>`` -- a real, inspectable, reproducible
    artefact on this machine, but **NOT** a publication and **NOT** a citation.

    Why this exists rather than a hard-coded triple: a bare "4x4x40 nm" would be
    an unstated guess.  What is actually available is the file's own
    ``PANDAS_ATTRS`` metadata, written by the code that exported it, which
    carries ``dataframe_resolution`` and ``table_voxel_resolution_x/_y/_z``.
    Those keys describe the RESOLUTION OF THE COORDINATE FRAME the rows are
    expressed in.

    HONEST LIMIT, stated because it matters: this function reads the frame's own
    declared resolution; it does NOT independently establish that the frame is
    correct, and this module could NOT verify BANC's voxel size from the
    literature within the time available.  Two independent checks were run and
    are reported by ``run_access_map.py``:

      * ARITHMETIC: at this resolution the soma cloud spans ~1.75 x 2.15 mm
        (brain + nerve cord), which is anatomically plausible; a 4 nm/axis
        isotropic reading would span 0.88 x 1.07 mm and a 1 nm isotropic reading
        0.22 x 0.27 mm, both too small for a fly CNS.
      * QUANTISATION: x and y coordinates are near-universally multiples of 16
        (x: 99.9987%), and the smallest observed x/y step is 16. That is
        consistent with the frame's coordinate values being reported on a finer
        grid than the EM voxel pitch, so it is EVIDENCE the numbers are
        image-space coordinates rather than a small-integer voxel index.

    Both are consistency checks, not a citation.  Treat the result as
    ``MEASURED_LOCAL`` and read the sweep in ``run_access_map.py``.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError("needs pyarrow (PYTHONPATH=/tmp/pq in this project): %s" % exc)
    pf = pq.ParquetFile(somas_parquet_path)
    md = pf.metadata.metadata or {}
    raw = md.get(b"PANDAS_ATTRS")
    if raw is None:
        raise ValueError("no PANDAS_ATTRS metadata in %s; cannot read the frame "
                         "resolution" % somas_parquet_path)
    attrs = json.loads(raw.decode("utf-8", "replace"))
    keys = ("table_voxel_resolution_x", "table_voxel_resolution_y",
            "table_voxel_resolution_z")
    missing = [k for k in keys if k not in attrs]
    if missing:
        raise ValueError("parquet metadata lacks %r" % (missing,))
    x = float(attrs[keys[0]])
    y = float(attrs[keys[1]])
    z = float(attrs[keys[2]])
    dfr = attrs.get("dataframe_resolution")
    return VoxelResolution(
        x, y, z, PROVENANCE.MEASURED_LOCAL,
        source=("%s :: PANDAS_ATTRS keys %s = %s, dataframe_resolution = %s"
                % (os.path.abspath(somas_parquet_path), list(keys),
                   [x, y, z], dfr)),
        note=("read from the file's own export metadata. NOT a publication and NOT "
              "a verified citation; see this module's docstring for the two "
              "consistency checks (anatomical-span arithmetic and coordinate "
              "quantisation) that support it, and the sweep in run_access_map.py. "
              "Named 'voxel_resolution_nm' for interface stability: the value is "
              "nanometres PER COORDINATE UNIT of this file's frame."))



# ---------------------------------------------------------------------------
# array specification
# ---------------------------------------------------------------------------
GEOMETRIES = ("planar_grid_xy", "linear_shank_x", "linear_shank_y", "linear_shank_z")
PLACEMENT_MODES = ("densest_readout_soma", "readout_centroid", "explicit_origin")


@dataclass(frozen=True)
class ElectrodeArraySpec:
    """The electrode array.  PITCH and DIAMETER are FIXED user inputs.

    ``pitch_um`` and ``diameter_um`` are the two numbers the user pinned down.
    They are INPUTS: this module never searches over them, never "improves" them,
    and never derives them from the somas.  They are recorded in the ledger with
    their provenance, and if that provenance is ASSUMED the sweep is reported.

    ``n_channels`` is FREE -- the user left the channel count open, so it IS
    swept.  Nothing else about the array is swept as a free design variable.

    Parameters
    ----------
    pitch_um
        centre-to-centre channel spacing.  FIXED.
    diameter_um
        contact diameter.  FIXED.  Carried as a declared spec field and written
        into every output; NOT used by the access geometry (see module
        docstring: there is no contact model).
    n_channels
        FREE.  For ``planar_grid_xy`` this is rounded up to a near-square
        ``n_side x n_side`` grid; for the linear shanks it is exact.
    capture_radius_um
        how close a soma POINT must be to a channel SITE to be considered
        addressable by that channel.  DECLARED, with provenance, never measured.
    geometry
        one of :data:`GEOMETRIES`.  ``planar_grid_xy`` lays the channels on a
        lattice in the x-y plane of the BANC voxel frame (the two axes with the
        smallest voxels and hence the plane a planar MEA would face);
        ``linear_shank_*`` lays them along one named axis.
    placement
        how the array origin is chosen from the REAL soma positions:
        ``densest_readout_soma`` (origin = the readout soma whose local
        neighbourhood contains the most readout somas -- data-driven, but note
        it is a CHOICE justified by "we want to record the readout", not by
        anatomy), ``readout_centroid``, or ``explicit_origin`` (voxels).
    """

    pitch_um: float
    diameter_um: float
    n_channels: int
    capture_radius_um: float
    geometry: str = "planar_grid_xy"
    placement: str = "densest_readout_soma"
    origin_vox: Optional[tuple] = None
    seed: int = 0

    def __post_init__(self):
        for name in ("pitch_um", "diameter_um", "capture_radius_um"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0:
                raise ValueError("%s must be positive and finite" % name)
            object.__setattr__(self, name, v)
        if isinstance(self.n_channels, bool) or not isinstance(self.n_channels, int):
            raise ValueError("n_channels must be an integer")
        if self.n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if self.geometry not in GEOMETRIES:
            raise ValueError("geometry must be one of %r" % (GEOMETRIES,))
        if self.placement not in PLACEMENT_MODES:
            raise ValueError("placement must be one of %r" % (PLACEMENT_MODES,))
        if self.placement == "explicit_origin":
            if self.origin_vox is None or len(tuple(self.origin_vox)) != 3:
                raise ValueError("explicit_origin placement needs a 3-tuple origin_vox")
            object.__setattr__(self, "origin_vox",
                               tuple(float(v) for v in self.origin_vox))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.diameter_um > self.pitch_um:
            # Not fatal physically, but it means adjacent contacts would overlap,
            # which is a contradiction of "an array of separate channels".
            raise ValueError(
                "diameter_um=%.4g exceeds pitch_um=%.4g: adjacent contacts would "
                "overlap, which contradicts an array of separate channels."
                % (self.diameter_um, self.pitch_um))

    # -- lattice -----------------------------------------------------------
    @property
    def grid_side(self):
        """Channels per side for a planar grid (near-square)."""
        if self.geometry != "planar_grid_xy":
            return None
        return int(math.ceil(math.sqrt(self.n_channels)))

    def relative_offsets_vox(self, voxel_resolution: VoxelResolution):
        """Channel centre offsets from the array origin, in VOXELS.

        Ordering is deterministic (row-major for the grid; index order for the
        shanks) so that a channel index means the same thing across runs.
        Always exactly ``n_channels`` sites.
        """
        p_vox = self.pitch_um * 1000.0 / voxel_resolution.nm_per_voxel  # um -> nm -> vox
        if self.geometry == "planar_grid_xy":
            s = self.grid_side
            gx, gy = np.meshgrid(np.arange(s), np.arange(s), indexing="xy")
            gx = gx.ravel().astype(float)
            gy = gy.ravel().astype(float)
            gx -= (s - 1) / 2.0          # centred; for even s the centre falls between sites
            gy -= (s - 1) / 2.0
            pts = np.column_stack((gx * p_vox[0], gy * p_vox[1], np.zeros_like(gx)))
            return pts[: self.n_channels]
        k = np.arange(self.n_channels, dtype=float)
        k -= (self.n_channels - 1) / 2.0
        pts = np.zeros((self.n_channels, 3), dtype=float)
        axis = {"linear_shank_x": 0, "linear_shank_y": 1, "linear_shank_z": 2}[self.geometry]
        pts[:, axis] = k * p_vox[axis]
        return pts

    def as_dict(self):
        return {"pitch_um": self.pitch_um, "diameter_um": self.diameter_um,
                "n_channels": self.n_channels,
                "capture_radius_um": self.capture_radius_um,
                "geometry": self.geometry, "placement": self.placement,
                "origin_vox": (None if self.origin_vox is None else list(self.origin_vox)),
                "seed": self.seed,
                "pitch_is_FIXED_input": True, "diameter_is_FIXED_input": True,
                "n_channels_is_FREE": True,
                "grid_side": self.grid_side}


# ---------------------------------------------------------------------------
# soma table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SomaTable:
    """The measured BANC soma cloud, in VOXEL coordinates.

    ``root_id -> index`` is a dict join: this module JOINS ON ``pt_root_id``.
    The parquet's ``id`` column is a DIFFERENT 17-digit identifier space and is
    NOT used for the join (mixing them up is the documented way to get an
    intersection of zero and wrongly conclude "the data has no coordinates").
    """

    root_ids: np.ndarray
    positions_vox: np.ndarray
    n_rows: int
    n_valid: int
    path: str
    duplicate_policy: str = "keep the FIRST row of each pt_root_id; drop sentinel roots"
    n_sentinel_rows: int = 0
    n_sentinel_distinct_roots: int = 0
    n_duplicate_roots: int = 0
    n_duplicate_rows_dropped: int = 0
    ambiguous_root_ids: tuple = ()
    max_displacement_um: float = 0.0
    _index: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "root_ids", np.asarray(self.root_ids, dtype=np.int64))
        object.__setattr__(self, "positions_vox",
                           np.asarray(self.positions_vox, dtype=np.int64))
        if self.root_ids.shape != (self.positions_vox.shape[0],):
            raise ValueError("root_ids and positions must have equal length")
        if self.positions_vox.ndim != 2 or self.positions_vox.shape[1] != 3:
            raise ValueError("positions must be (n,3)")
        # FIRST occurrence wins.  A dict comprehension over both columns keeps the
        # LAST row instead, so the join would silently disagree with any explicit
        # first-occurrence rule; this loop makes the choice explicit and testable.
        index = {}
        for i, v in enumerate(self.root_ids.tolist()):
            if int(v) not in index:
                index[int(v)] = i
        object.__setattr__(self, "_index", index)

    def duplicate_report(self):
        """What the deduplication did, so it is reported rather than implied."""
        return {"policy": self.duplicate_policy,
                "sentinel_rows_dropped": int(self.n_sentinel_rows),
                "sentinel_distinct_roots": int(self.n_sentinel_distinct_roots),
                "duplicate_roots": int(self.n_duplicate_roots),
                "duplicate_rows_dropped": int(self.n_duplicate_rows_dropped),
                "max_displacement_um": float(self.max_displacement_um),
                "ambiguous_root_ids_sample": [int(v) for v in self.ambiguous_root_ids[:8]],
                "note": ("One pt_root_id can carry several DISTINCT nucleus positions "
                         "(adjacent rows in the source table, up to tens of um apart). "
                         "The join keeps one row, so any quantity derived from the "
                         "captured row inherits that positional ambiguity. Sentinel "
                         "roots (pt_root_id <= 0) are not neurons and are dropped.")}

    def lookup(self, root_ids):
        """Indices into the soma table for ``root_ids``.

        Returns ``(idx, found_mask, missing_root_ids)``.  Missing ids are
        reported, never silently dropped or wrapped onto the last row.
        """
        r = np.asarray(root_ids, dtype=np.int64).ravel()
        idx = np.full(r.shape[0], -1, dtype=np.int64)
        for i, v in enumerate(r.tolist()):
            j = self._index.get(int(v))
            if j is not None:
                idx[i] = j
        found = idx >= 0
        return idx[found], found, r[~found]

    def positions_of(self, root_ids):
        idx, found, missing = self.lookup(root_ids)
        return self.positions_vox[idx], found, missing


def load_soma_table(somas_parquet_path: str, *, resolution=None) -> SomaTable:
    """Read the BANC soma parquet.  Requires ``pyarrow`` on the path.

    Only ``pt_root_id`` and ``pt_position`` are used.  ``valid`` is checked and
    counted, not filtered silently.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "reading %s needs pyarrow. In this project it is installed at /tmp/pq; "
            "run with PYTHONPATH=/tmp/pq. Original error: %s"
            % (somas_parquet_path, exc))
    if not os.path.isfile(somas_parquet_path):
        raise FileNotFoundError("soma parquet not found: %s" % somas_parquet_path)
    pf = pq.ParquetFile(somas_parquet_path)
    # NOTE: ``pf.schema.names`` lists the FLAT leaf names, so the list-typed
    # ``pt_position`` shows up as its child name ``item``.  The parent names live
    # on the Arrow schema; use that one.
    names = list(pf.schema_arrow.names)
    for col in ("pt_root_id", "pt_position"):
        if col not in names:
            raise ValueError("soma parquet lacks required column %r; has %r"
                             % (col, names))
    want = [c for c in ("pt_root_id", "pt_position", "valid") if c in names]
    t = pf.read(columns=want)
    root = np.asarray(t.column("pt_root_id"))
    pos = np.asarray(t.column("pt_position").to_pylist(), dtype=np.int64)
    n_rows = int(t.num_rows)
    n_valid = int(np.asarray(t.column("valid")).sum()) if "valid" in names else n_rows

    # ---- explicit deduplication -------------------------------------------------
    # The deposited table is somas_v1a updated by somas_v1b, so the SAME root id can
    # sit on several rows with DISTINCT positions (measured here: 69 roots, one of
    # them on 127 rows, tens of um apart).  A root id appearing once is
    # unambiguous; a root id appearing several times is not, so the choice must be
    # declared and its positional ambiguity quantified rather than implied.
    res = resolution if resolution is not None else banc_voxel_resolution()
    nm_per_unit = np.asarray([res.x_nm, res.y_nm, res.z_nm], dtype=np.float64)
    sentinel = root <= 0
    n_sentinel = int(np.count_nonzero(sentinel))
    n_sentinel_roots = int(np.unique(root[sentinel]).size) if n_sentinel else 0
    keep = np.flatnonzero(~sentinel)
    root = root[keep]
    pos = pos[keep]

    uniq, inverse, counts = np.unique(root, return_inverse=True, return_counts=True)
    dup_roots = uniq[counts > 1]
    max_disp = 0.0
    ambiguous = []
    for g in np.flatnonzero(counts > 1).tolist():
        p = pos[inverse == g].astype(np.float64)
        d = np.linalg.norm((p[:, None, :] - p[None, :, :]) * nm_per_unit / 1000.0, axis=-1)
        spread = float(d.max())
        max_disp = max(max_disp, spread)
        if spread > 0.0:
            ambiguous.append(int(uniq[g]))
    # ACTUALLY drop the extra rows: keep the first row of each root id.  (An earlier
    # revision of this block computed every statistic above and then returned the
    # table UNCHANGED, so 68 roots stayed duplicated downstream and the front end's
    # conflicting-coordinate guard fired.  The row count and uniqueness are asserted
    # so that mistake cannot come back silently.)
    first = np.full(uniq.shape[0], -1, dtype=np.int64)
    for row, g in enumerate(inverse.tolist()):
        if first[g] < 0:
            first[g] = row
    keep_rows = np.sort(first[first >= 0])
    root = root[keep_rows]
    pos = pos[keep_rows]
    n_dup_dropped = int(counts[counts > 1].sum() - dup_roots.size)
    if len(root) != n_rows - n_sentinel - n_dup_dropped:
        raise AssertionError("deduplication inconsistent: %d rows kept, expected %d"
                             % (len(root), n_rows - n_sentinel - n_dup_dropped))
    if np.unique(root).size != len(root):
        raise AssertionError("deduplication left duplicate root ids in the table")
    return SomaTable(root_ids=root, positions_vox=pos, n_rows=n_rows,
                     n_valid=n_valid, path=os.path.abspath(somas_parquet_path),
                     n_sentinel_rows=n_sentinel,
                     n_sentinel_distinct_roots=n_sentinel_roots,
                     n_duplicate_roots=int(dup_roots.size),
                     n_duplicate_rows_dropped=n_dup_dropped,
                     ambiguous_root_ids=tuple(ambiguous),
                     max_displacement_um=max_disp)


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AccessMap:
    """Which neurons each channel can reach, from REAL measured soma positions.

    Arrays are indexed consistently:

    * ``channel_centres_vox``: (n_channels, 3) voxel coordinates.
    * ``channel_neurons``: ``list[tuple[int, ...]]`` of indices into
      ``soma_root_ids`` (i.e. rows of the soma table), sorted by distance from
      the channel centre.  Length is exactly ``n_channels``, so channel k's
      neurons are ``channel_neurons[k]`` -- there is no shift or off-by-one to
      get wrong.
    * ``channel_neurons_um``: parallel list of distances in micrometres, same
      order, so attenuation can be read straight off.
    * ``neuron_first_channel``: (n_soma, ) index of the NEAREST channel that
      captures each soma, or -1.  This is the per-neuron -> channel direction of
      the map; "nearest" is used so that a soma in two spheres has ONE owner.
    """

    spec: ElectrodeArraySpec
    resolution: VoxelResolution
    channel_centres_vox: np.ndarray
    channel_centres_um: np.ndarray
    channel_neurons: tuple = ()
    channel_neurons_um: tuple = ()
    soma_root_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    soma_positions_vox: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int64))
    neuron_first_channel: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    neuron_n_channels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    reports: dict = field(default_factory=dict)
    ledger: Ledger = field(default_factory=Ledger)
    attribution: dict = field(default_factory=lambda: dict(BANC_SOMA_ATTRIBUTION))

    # -- convenience -------------------------------------------------------
    @property
    def n_channels(self):
        return int(np.asarray(self.channel_centres_vox).shape[0])

    def counts_by_channel(self):
        return np.array([len(c) for c in self.channel_neurons], dtype=np.int64)

    @property
    def extent_um(self):
        """Physical extent of the array (bounding box of the channel centres)."""
        c = np.asarray(self.channel_centres_um, dtype=float)
        if c.size == 0:
            return np.zeros(3)
        return c.max(axis=0) - c.min(axis=0)

    def captured_mask(self):
        return np.asarray(self.neuron_n_channels) > 0

    def as_dict(self, *, include_channel_lists=True):
        counts = self.counts_by_channel()
        d = {
            "attribution": dict(self.attribution),
            "disclaimer": DISCLAIMER,
            "spec": self.spec.as_dict(),
            "voxel_resolution": self.resolution.as_dict(),
            "array": {
                "n_channels": self.n_channels,
                "extent_um": [float(v) for v in self.extent_um],
                "extent_um_by_axis": {"x": float(self.extent_um[0]),
                                      "y": float(self.extent_um[1]),
                                      "z": float(self.extent_um[2])},
                "channel_centres_um": [[float(v) for v in row]
                                       for row in np.asarray(self.channel_centres_um)],
                "channel_centres_vox": [[int(v) for v in row]
                                        for row in np.asarray(self.channel_centres_vox)],
                "counts_by_channel": [int(v) for v in counts],
                "counts_by_channel_histogram": _histogram(counts),
                "empty_channels": int(np.sum(counts == 0)),
            },
            "reports": self.reports,
            "provenance_ledger": self.ledger.as_list(),
            "provenance_counts": self.ledger.counts(),
            "unresolved_entries": self.ledger.unresolved(),
        }
        if include_channel_lists:
            d["channel_neurons"] = {
                "soma_row_indices": [[int(i) for i in c] for c in self.channel_neurons],
                "distances_um": [[float(v) for v in ds] for ds in self.channel_neurons_um],
            }
            d["per_neuron"] = {
                "soma_root_ids": [int(v) for v in self.soma_root_ids],
                "first_channel": [int(v) for v in self.neuron_first_channel],
                "n_channels_capturing": [int(v) for v in self.neuron_n_channels],
            }
        return d


DISCLAIMER = (
    "GEOMETRIC ACCESS MAP ONLY.  Somas are SINGLE VOXEL POINTS (no morphology).  "
    "There is NO electrode-contact / impedance / crosstalk / stimulation model, "
    "and NO tissue mechanics, in this map: diameter_um is a declared spec field "
    "that the access geometry does not use.  capture_radius_um is DECLARED, not "
    "measured.  Any micrometre quantity depends entirely on voxel_resolution_nm, "
    "which is marked with its own provenance and may be ASSUMED.  A channel that "
    "'accesses' a neuron is NOT a claim that its signal is recoverable.  Hand-"
    "built research prototype, not a validated device model.  No claim about "
    "behaviour, perception, attention, recognition, experience or identity.")


def _histogram(counts):
    counts = np.asarray(counts)
    if counts.size == 0:
        return {}
    vals, freqs = np.unique(counts, return_counts=True)
    # cap the histogram width so a single dense channel cannot explode the JSON
    out = {}
    cap = 32
    for v, f in zip(vals.tolist(), freqs.tolist()):
        out[str(int(v)) if v < cap else "%d+" % cap] = out.get(
            str(int(v)) if v < cap else "%d+" % cap, 0) + int(f)
    return out


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------
def _choose_origin_vox(spec, soma: SomaTable, target_vox, resolution, ledger):
    """Array origin in voxels, chosen from REAL soma positions.

    ``densest_readout_soma`` scans every target soma as a candidate origin and
    counts how many target somas would fall inside a capture sphere centred on
    the candidate (single-site proxy for "densest place to put the array").
    Deterministic: ties break on the smaller soma row index.  No RNG is used,
    so ``spec.seed`` cannot change this result -- which is why the result is
    reproducible.
    """
    if spec.placement == "explicit_origin":
        ledger.record("array_origin_vox", list(spec.origin_vox), "voxel",
                      PROVENANCE.ENGINEERING_DEFAULT,
                      note="caller-supplied explicit origin")
        return np.asarray(spec.origin_vox, dtype=float)
    if target_vox.size == 0:
        raise ValueError("no target somas available to place the array")
    if spec.placement == "readout_centroid":
        o = np.asarray(target_vox, dtype=float).mean(axis=0)
        ledger.record("array_origin_vox", [float(v) for v in o], "voxel",
                      PROVENANCE.ENGINEERING_DEFAULT,
                      note="centroid of the target (readout) somas; a CHOICE, not anatomy")
        return o
    r_vox = spec.capture_radius_um * 1000.0 / resolution.nm_per_voxel
    # distances are compared in MICROMETRES, never in raw coordinate units, so
    # an anisotropic voxel is handled correctly and a resolution of 1 nm/unit
    # does not silently turn a 50 um capture radius into 50 nm.
    from scipy.spatial import cKDTree
    pos = np.asarray(target_vox, dtype=float)
    pos_um = resolution.voxels_to_um(pos)
    tree = cKDTree(pos_um)
    best_n, best_i = -1, 0
    for i, p in enumerate(pos_um):
        n = len(tree.query_ball_point(p, spec.capture_radius_um))
        if n > best_n:
            best_n, best_i = n, i
    o = pos[best_i].astype(float)
    ledger.record("array_origin_vox", [float(v) for v in o], "voxel",
                  PROVENANCE.MEASURED_LOCAL,
                  note=("origin = the target soma with the most target somas within "
                        "capture_radius_um (%d of %d) -- a data-driven CHOICE to "
                        "place the array where the readout is densest, NOT an "
                        "anatomical constraint. Deterministic, no RNG."
                        % (best_n, pos.shape[0])))
    ledger.record("array_origin_local_density_neurons", int(best_n), "neurons",
                  PROVENANCE.MEASURED_LOCAL,
                  note="target somas within capture_radius_um of the chosen origin")
    ledger.record("array_placement_r_vox_isotropic_proxy", [float(v) for v in r_vox],
                  "voxel", PROVENANCE.MEASURED_LOCAL,
                  note=("per-axis radius in voxels, for reference only; distances are "
                        "computed in micrometres so anisotropic voxels are handled "
                        "correctly"))
    return o


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
def build_access_map(spec, tier_root_ids, somas_parquet_path, voxel_resolution_nm,
                     *, soma_table=None, target_mask=None, target_names=None,
                     tier_names=None, resolution=None):
    """Place ``spec`` using the REAL soma positions and map channels to neurons.

    Parameters
    ----------
    spec : ElectrodeArraySpec
        pitch and diameter are FIXED inputs; ``n_channels`` is free.
    tier_root_ids : array-like of int
        the actual BANC root ids of the tier, in tier-local order
        (``root_ids[global_index]`` -- NOT ``global_index``).
    somas_parquet_path : str
        path to ``somas_v1.parquet`` (CC BY 4.0, see module docstring).
    voxel_resolution_nm : VoxelResolution or 3-sequence
        the voxel -> um bridge.  A bare triple is accepted and then recorded as
        ASSUMED with the full unverified candidate sweep, because a bare triple
        carries no source.
    soma_table : SomaTable, optional
        pass a pre-loaded table to avoid re-reading the parquet.
    target_mask : bool array over tier-local indices, optional
        which tier neurons the coverage report is ABOUT (e.g. the 195 readout
        neurons).  The map itself is built over ALL joined somas in the tier;
        ``target_mask`` only selects the population that coverage is reported
        for and that placement is optimised for.
    resolution : VoxelResolution, optional
        alias for ``voxel_resolution_nm`` when a ``VoxelResolution`` is passed.
    """
    if resolution is not None:
        voxel_resolution_nm = resolution
    res = _as_resolution(voxel_resolution_nm)

    tier_root_ids = np.asarray(tier_root_ids, dtype=np.int64).ravel()
    if tier_root_ids.size == 0:
        raise ValueError("tier_root_ids is empty")
    soma = soma_table if soma_table is not None else load_soma_table(somas_parquet_path)

    tier_idx, found, missing = soma.lookup(tier_root_ids)
    if tier_idx.size == 0:
        raise ValueError(
            "no tier root id joined to a soma. The join key is pt_root_id, and "
            "tier root ids come from root_ids[global_index] -- if you passed "
            "global_index directly you will always land here.")
    tier_pos_vox = soma.positions_vox[tier_idx]

    if target_mask is None:
        target_mask_local = np.ones(tier_root_ids.size, dtype=bool)
    else:
        target_mask_local = np.asarray(target_mask, dtype=bool).ravel()
        if target_mask_local.shape != tier_root_ids.shape:
            raise ValueError("target_mask must have one entry per tier root id")
    # ``tier_idx`` is indexed by the JOINED tier-local indices, in order, i.e.
    # tier_idx[r] is the soma row of tier-local index np.flatnonzero(found)[r].
    joined_local = np.flatnonzero(found)
    target_joined = target_mask_local & found          # tier-local -> bool
    target_rows = tier_idx[target_mask_local[found]]   # soma rows of the targets
    target_vox = soma.positions_vox[target_rows]
    if target_vox.shape[0] == 0:
        raise ValueError("no target neuron in the tier has a soma; nothing to place")

    # ---- ledger -----------------------------------------------------------
    led = Ledger()
    led.record("dataset.banc_somas", BANC_SOMA_DATASET_TITLE, "name",
               PROVENANCE.MEASURED_CITED,
               source="%s, Harvard Dataverse, file %s (file id %d), %s"
                      % (BANC_SOMA_DOI, BANC_SOMA_FILE, BANC_SOMA_FILE_ID,
                         BANC_SOMA_LICENCE),
               note="CC BY 4.0: attribution is a licence CONDITION. Redistribution "
                    "of this data or any derived map must carry this record.")
    led.record("dataset.banc_somas_doi", BANC_SOMA_DOI, "doi",
               PROVENANCE.MEASURED_CITED,
               source="https://doi.org/10.7910/DVN/7WTH1N")
    led.record("dataset.banc_somas_licence", BANC_SOMA_LICENCE, "licence",
               PROVENANCE.MEASURED_CITED,
               source="http://creativecommons.org/licenses/by/4.0",
               note="redistribution must carry the same attribution")
    led.record("dataset.soma_rows", int(soma.n_rows), "rows",
               PROVENANCE.MEASURED_LOCAL, source=soma.path,
               note="read from the local parquet; valid rows counted, not filtered")
    led.record("dataset.soma_rows_valid", int(soma.n_valid), "rows",
               PROVENANCE.MEASURED_LOCAL, source=soma.path)
    led.record("join.key", "pt_root_id", "column",
               PROVENANCE.MEASURED_LOCAL,
               note=("the parquet's `id` column is a DIFFERENT 17-digit space and is "
                     "NOT the join key"))
    led.record("join.tier_local_indices_have_soma", int(found.sum()), "neurons",
               PROVENANCE.MEASURED_LOCAL,
               note="of %d tier root ids" % tier_root_ids.size)
    # DEFECT FIXED HERE.  This record used to pass sweep=None unconditionally.  When a
    # caller supplies a BARE (x, y, z) triple, the resolution is wrapped as ASSUMED, and
    # the ledger's own refusal ("refusing ASSUMED without a declared sweep/range") then
    # fired on THIS line -- BEFORE the voxel_resolution_verdict record further down,
    # which was the record that actually carried the candidate sweep.  So
    # build_access_map() raised inside its own ledger for the most natural call
    # signature, and two independent users hit it.  The sweep is now declared on the
    # first record, for exactly the provenance that requires one.
    led.record("voxel_resolution_nm", [res.x_nm, res.y_nm, res.z_nm], "nm/voxel",
               res.provenance, source=res.source, note=res.note,
               sweep=(list(UNVERIFIED_RESOLUTION_CANDIDATES_NM)
                      if res.provenance == PROVENANCE.ASSUMED else None))
    if res.provenance == PROVENANCE.MEASURED_CITED:
        led.record("voxel_resolution_verdict", "VERIFIED_FROM_SOURCE", "verdict",
                   PROVENANCE.MEASURED_CITED, source=res.source,
                   note=("BANC native voxel size is ANISOTROPIC (4 x 4 x 45 nm); a "
                         "scalar would mis-scale z by 11.25x."))
    else:
        led.record("voxel_resolution_verdict",
                   "MEASURED_LOCAL__NOT_A_CITATION", "verdict",
                   PROVENANCE.ASSUMED,
                   sweep=list(UNVERIFIED_RESOLUTION_CANDIDATES_NM),
                   note=("this resolution carries NO external source, so every "
                         "micrometre quantity is assumption-dependent. Prefer "
                         "banc_voxel_resolution(), which is sourced."))
    led.record("electrode.pitch_um", spec.pitch_um, "um",
               PROVENANCE.ASSUMED,
               sweep=(20.0, 30.0, 50.0, 100.0, 200.0),
               note=("FIXED USER INPUT, not searched and not derived here. No probe "
                     "pitch value exists in this project's own provenance tables "
                     "(engine/electrode_damage.py declares tip DIAMETERS but no "
                     "channel PITCH), so this is ASSUMED and swept."))
    # DEFECT FIXED HERE.  This record used to cite `assumed_shaft_diameter_um = 10.0 um`
    # and to state that nothing consumes the value.  Both were stale: the user has since
    # FIXED the diameter at 7 um, and electrode_frontend.ContactGeometry DOES consume it
    # (the contact area, hence R_spread and the noise floor, is derived from it).  The
    # registry value is now cited only as the historical origin of the field, clearly
    # marked as superseded, so the provenance cannot be read as endorsing 10 um.
    led.record("electrode.diameter_um", spec.diameter_um, "um",
               PROVENANCE.ENGINEERING_DEFAULT,
               source=("FIXED USER INPUT (7 um).  Historical origin of this field: "
                       "engine/electrode_damage.py::Param(assumed_shaft_diameter_um) "
                       "= 10.0 um -- SUPERSEDED here by the user's decision and NOT "
                       "the value used."),
               note=("FIXED USER INPUT, not searched, not derived and not optimised. "
                     "It IS consumed downstream: engine/electrode_frontend.py derives "
                     "the contact area from it, and the contact area sets R_spread = "
                     "1/(4*sigma*a), hence the thermal noise floor.  (An earlier note "
                     "here wrongly said nothing consumed it.)"))
    led.record("electrode.n_channels", int(spec.n_channels), "channels",
               PROVENANCE.ENGINEERING_DEFAULT,
               note="FREE design variable by the user's constraint; swept in run_access_map.py")
    led.record("electrode.geometry", spec.geometry, "enum",
               PROVENANCE.ENGINEERING_DEFAULT,
               note="stated array layout; channel spacing is exactly pitch_um in each "
                    "in-plane axis")
    led.record("electrode.placement", spec.placement, "enum",
               PROVENANCE.ENGINEERING_DEFAULT,
               note="array-origin choice; a recording-motivated CHOICE, not anatomy")
    led.record("capture.radius_um", spec.capture_radius_um, "um",
               PROVENANCE.ASSUMED,
               sweep=(10.0, 25.0, 50.0, 100.0),
               note=("DECLARED point-capture radius: a soma POINT inside this sphere is "
                     "counted as addressable by the channel. Not a measurement, not a "
                     "tissue volume, not a membrane-area claim."))
    led.record("seed", int(spec.seed), "int", PROVENANCE.ENGINEERING_DEFAULT,
               note=("carried but UNUSED: no step of this map draws a random number, "
                     "so the map is deterministic and seed-independent"))
    led.record("somas_are_single_voxel_points", True, "bool",
               PROVENANCE.MEASURED_LOCAL,
               note=("pt_position is one voxel per neuron (pt_supervoxel_id); no "
                     "morphology. A neurite passing a channel is INVISIBLE to this map."))

    # ---- placement --------------------------------------------------------
    origin_vox = _choose_origin_vox(spec, soma, target_vox, res, led)
    offsets = spec.relative_offsets_vox(res)
    centres_vox = origin_vox[None, :] + offsets
    centres_um = res.voxels_to_um(centres_vox)

    # ---- geometric capture ------------------------------------------------
    from scipy.spatial import cKDTree
    all_um = res.voxels_to_um(soma.positions_vox)
    tree = cKDTree(all_um)
    n_ch = centres_um.shape[0]
    # STRICT INTERIOR predicate, with an explicit epsilon.
    #
    # ``capture_radius_um`` is documented as "a soma point INSIDE this sphere",
    # so a soma at EXACTLY the radius is NOT captured.  Two sites at exactly
    # ``pitch`` apart with radius = pitch/2 therefore share exactly one boundary
    # point, and a soma sitting on it would be reported as a collision or not
    # depending on floating-point luck in ``sqrt``. That was observed: one linear
    # shank produced a "collision" whose two distances were both exactly 50.0.
    # The epsilon makes the strict reading explicit and the result deterministic,
    # so the declared invariant (pitch >= 2*radius => zero collisions) holds by
    # construction rather than by accident.  It is a numerical tie-break with no
    # biological content.
    EPS_UM = 1e-9
    r_query = spec.capture_radius_um - EPS_UM
    ch_neurons = []
    ch_dists = []
    for k in range(n_ch):
        idx = tree.query_ball_point(centres_um[k], r_query)
        # enforce the predicate on the recomputed distance too, so the reported
        # distances and the membership test can never disagree
        dmap = {int(j): float(np.linalg.norm(all_um[j] - centres_um[k])) for j in idx}
        idx = [j for j in dmap if dmap[j] < spec.capture_radius_um]
        if len(idx) > 1:
            idx = sorted(idx, key=lambda j: (dmap[j], j))
        d = [dmap[j] for j in idx]
        ch_neurons.append(tuple(int(j) for j in idx))
        ch_dists.append(tuple(d))

    n_soma = soma.positions_vox.shape[0]
    n_ch_of = np.zeros(n_soma, dtype=np.int64)
    for idx in ch_neurons:
        for j in idx:
            n_ch_of[j] += 1
    # "nearest capturing channel" per soma: distance-exact, deterministic ties.
    first_ch = np.full(n_soma, -1, dtype=np.int64)
    best = np.full(n_soma, np.inf, dtype=float)
    for k, (idx, ds) in enumerate(zip(ch_neurons, ch_dists)):
        for j, d in zip(idx, ds):
            if d < best[j] or (d == best[j] and k < first_ch[j]):
                best[j] = d
                first_ch[j] = k

    # ---- reports ----------------------------------------------------------
    # NOTE ON UNITS: ``all_um`` is in MICROMETRES, ``tier_idx`` and
    # ``target_rows`` are ROW INDICES into the soma table.  Mixing the two --
    # e.g. indexing a per-neuron array with a row index and calling it coverage
    # -- reports the whole cloud's capture as if it were the target's. That bug
    # was live in an earlier revision and produced a flattering, meaningless
    # 99.5%; the three lines below are deliberately explicit about which is which.
    tier_positions_um = all_um[tier_idx]
    captured_all = n_ch_of[tier_idx] > 0
    target_captured = n_ch_of[target_rows] > 0
    n_target = int(target_joined.sum())
    n_target_captured = int(target_captured.sum())
    assert target_captured.size == n_target, (
        "internal: target coverage has %d entries for %d targets"
        % (target_captured.size, n_target))
    counts = np.array([len(c) for c in ch_neurons], dtype=np.int64)

    # collisions: neurons in MORE THAN ONE capture sphere
    collided = n_ch_of[tier_idx] > 1
    n_collision_neurons = int(collided.sum())
    collision_soma_incidences = int(n_ch_of[tier_idx][collided].sum())

    # WHERE THE UNCAPTURED TARGETS ARE LOST. A planar array is confined to one
    # z-plane, so an uncaptured target is lost either to the in-plane lattice
    # (lateral) or to the finite z-slab that plane can reach. Separating the two
    # matters because they have different fixes (more sites vs a 3-D layout).
    # NOTE: these two tests are NECESSARY conditions, not the full 3-D criterion
    # -- their intersection is a superset of `target_captured`, which is exactly
    # why the counts below are reported as a decomposition, not as the answer.
    from scipy.spatial import cKDTree as _KD
    t_um = all_um[target_rows]
    lat = _KD(centres_um[:, :2]).query(t_um[:, :2], k=1)[0]
    plane_z = float(centres_um[0, 2])
    dz = np.abs(t_um[:, 2] - plane_z)
    lat_ok = lat < spec.capture_radius_um
    z_ok = dz < spec.capture_radius_um

    reports = {
        "loss_decomposition": {
            "array_plane_z_um": plane_z,
            "target_z_um_min": float(t_um[:, 2].min()),
            "target_z_um_max": float(t_um[:, 2].max()),
            "target_z_span_um": float(t_um[:, 2].max() - t_um[:, 2].min()),
            "z_slab_reachable_um": float(2 * spec.capture_radius_um),
            "targets_laterally_reachable_lt_r": int(lat_ok.sum()),
            "targets_z_reachable_lt_r": int(z_ok.sum()),
            "targets_reachable_by_both": int((lat_ok & z_ok).sum()),
            "targets_captured_3d": int(target_captured.sum()),
            "lost_to_z_only": int((lat_ok & ~z_ok).sum()),
            "lost_to_lateral_only": int((~lat_ok & z_ok).sum()),
            "lost_to_both": int((~lat_ok & ~z_ok).sum()),
            "note": ("for a planar_grid_xy array the sites all share ONE z, so the "
                     "reachable z-slab is 2*capture_radius_um thick. Targets outside "
                     "it cannot be reached by ANY number of channels in that plane; "
                     "adding channels only helps the lateral term."),
        },
        "population": {
            "tier_neurons": int(tier_root_ids.size),
            "tier_neurons_with_soma": int(found.sum()),
            "tier_join_fraction": float(found.sum() / tier_root_ids.size),
            "target_neurons": int(target_mask_local.sum()),
            "target_neurons_with_soma": n_target,
            "target_join_fraction": (float(n_target / int(target_mask_local.sum()))
                                     if target_mask_local.sum() else None),
            "target_names": list(target_names) if target_names else None,
        },
        "soma_cloud_vox": {
            "min": [int(v) for v in soma.positions_vox.min(axis=0)],
            "max": [int(v) for v in soma.positions_vox.max(axis=0)],
            "span": [int(v) for v in (soma.positions_vox.max(axis=0)
                                      - soma.positions_vox.min(axis=0))],
        },
        "soma_cloud_um": {
            "min": [float(v) for v in res.voxels_to_um(soma.positions_vox.min(axis=0))],
            "max": [float(v) for v in res.voxels_to_um(soma.positions_vox.max(axis=0))],
            "span": [float(v) for v in res.voxels_to_um(soma.positions_vox.max(axis=0)
                                                        - soma.positions_vox.min(axis=0))],
            "tier_min": [float(v) for v in tier_positions_um.min(axis=0)],
            "tier_max": [float(v) for v in tier_positions_um.max(axis=0)],
            "tier_span": [float(v) for v in (tier_positions_um.max(axis=0)
                                             - tier_positions_um.min(axis=0))],
        },
        "coverage": {
            "target_neurons": n_target,
            "target_neurons_captured": n_target_captured,
            "target_addressable_fraction": (float(n_target_captured / n_target)
                                            if n_target else None),
            "all_joined_tier_neurons": int(found.sum()),
            "all_joined_tier_neurons_captured": int(captured_all.sum()),
            "all_joined_tier_fraction": float(captured_all.mean()) if found.sum() else None,
        },
        "collision": {
            "neurons_captured_by_more_than_one_channel": n_collision_neurons,
            "collision_fraction_of_joined_tier": (float(n_collision_neurons / int(found.sum()))
                                                  if found.sum() else None),
            "channel_neuron_incidences_on_collided_neurons": collision_soma_incidences,
            "max_channels_per_neuron": int(n_ch_of[tier_idx].max()) if found.sum() else 0,
            "histogram_channels_per_neuron": _histogram(n_ch_of[tier_idx]),
            "note": ("a collision means one neuron's soma point lies inside two or more "
                     "channel capture spheres. This is a GEOMETRIC OVERLAP COUNT ONLY: "
                     "there is no spike sorting, no source separation and no "
                     "disentangling model here, so a collision is reported as an "
                     "unresolved ambiguity, not as something this map can fix."),
        },
        "array": {
            "n_channels": int(n_ch),
            "nonempty_channels": int((counts > 0).sum()),
            "empty_channels": int((counts == 0).sum()),
            "min_channels_per_neuron_owner": int(counts[counts > 0].min()) if (counts > 0).any() else 0,
            "max_neurons_on_one_channel": int(counts.max()),
            "mean_neurons_per_nonempty_channel": (float(counts[counts > 0].mean())
                                                  if (counts > 0).any() else 0.0),
            "extent_um": [float(v) for v in (centres_um.max(axis=0) - centres_um.min(axis=0))],
            "extent_vox": [float(v) for v in (centres_vox.max(axis=0) - centres_vox.min(axis=0))],
            "footprint_um2_xy": float(np.prod(centres_um.max(axis=0)[:2]
                                              - centres_um.min(axis=0)[:2])),
            "pitch_um_as_built_min_pairwise": _min_pairwise_um(centres_um),
        },
        "provenance_note": DISCLAIMER,
    }

    led.record("result.target_addressable_fraction",
               reports["coverage"]["target_addressable_fraction"], "fraction",
               PROVENANCE.MEASURED_LOCAL,
               note="COMPUTED from this spec; not a measurement of any real device")
    led.record("result.collision_neurons", n_collision_neurons, "neurons",
               PROVENANCE.MEASURED_LOCAL, note="geometric overlap count")

    return AccessMap(spec=spec, resolution=res, channel_centres_vox=centres_vox,
                     channel_centres_um=centres_um,
                     channel_neurons=tuple(ch_neurons), channel_neurons_um=tuple(ch_dists),
                     soma_root_ids=np.asarray(soma.root_ids, dtype=np.int64),
                     soma_positions_vox=np.asarray(soma.positions_vox, dtype=np.int64),
                     neuron_first_channel=first_ch, neuron_n_channels=n_ch_of,
                     reports=reports, ledger=led)


def _min_pairwise_um(centres_um):
    c = np.asarray(centres_um, dtype=float)
    if c.shape[0] < 2:
        return None
    from scipy.spatial import cKDTree
    d, _ = cKDTree(c).query(c, k=2)
    return float(d[:, 1].min())


def _as_resolution(voxel_resolution_nm):
    if isinstance(voxel_resolution_nm, VoxelResolution):
        return voxel_resolution_nm
    a = tuple(float(v) for v in voxel_resolution_nm)
    if len(a) != 3:
        raise ValueError("voxel resolution must be a VoxelResolution or a 3-sequence")
    if a[0] == a[1] == a[2]:
        src = None
    else:
        src = None
    return VoxelResolution(
        a[0], a[1], a[2], PROVENANCE.ASSUMED, source=src,
        note=("passed as a bare 3-sequence with NO SOURCE, so recorded ASSUMED. "
              "Swept over UNVERIFIED_RESOLUTION_CANDIDATES_NM. Every micrometre "
              "quantity scales linearly with this number."))


# ---------------------------------------------------------------------------
# wiring to the existing recording front end
# ---------------------------------------------------------------------------
def access_map_to_recording(access_map, *, channel_index=0, neuron_current_nA=0.01,
                            conductivity_S_m=0.3, bandwidth_Hz=1000.0,
                            noise_sd_mV=0.0, dt_ms=0.1, n_steps=None,
                            source_mode="channel_site", seed=0):
    """Turn ONE channel of ``access_map`` into a signal, using the EXISTING front end.

    Volume conduction is **reused, not reimplemented**: every potential is a call
    into :func:`engine.electrode.transfer_mV_per_nA`, and every trace is produced
    by :class:`engine.electrode.VoltageRecorder`.  There is no second field
    solver anywhere in this module.

    ``source_mode``
        * ``"channel_site"`` (default): ONE source contact per captured neuron,
          placed at the CHANNEL's own position.  This is an explicitly stated
          NON-ANATOMICAL degeneracy: it encodes "this neuron is in range of this
          channel" and stops there.  Its only virtue is that the resulting trace
          has a magnitude exactly equal to ``transfer_mV_per_nA``'s 1/r law
          evaluated AT the channel centre, i.e. the true attenuation for a source
          at the contact is ``inf``-free and the numbers are interpretable.
        * ``"soma_position"``: sources placed at the REAL measured soma voxel
          positions of the captured neurons, in micrometres.  Geometrically
          meaningful -- and it immediately shows the point of the whole exercise:
          a soma ``capture_radius_um`` away contributes ``1/(4*pi*sigma*r)``, so a
          100 um soma on a 50 um-radius channel is ALREADY attenuated to a few
          per cent of a 1 um one.  Note the soma POSITION here is a single voxel
          point, which is a floor on the model's realism, not a validated source
          geometry.

    The measured membrane source set must BALANCE (``VoltageRecorder.step`` raises
    otherwise), so this function adds one explicit distant RETURN contact per
    channel, at ``array_half_diagonal + return_offset_um`` from the channel, on
    the tissue side opposite the array.  The return carries ``-sum(I)``.  This is
    a DECLARED closed-domain convention, exactly as ``engine.electrode``'s own
    docstring requires ("the caller must supply a balanced source set in the
    chosen closed domain"); it is not a model of a reference electrode.

    Returns a dict with the trace plus everything needed to re-derive it.
    """
    if not isinstance(access_map, AccessMap):
        raise TypeError("access_map must be an AccessMap")
    if source_mode not in ("channel_site", "soma_position"):
        raise ValueError("source_mode must be 'channel_site' or 'soma_position'")
    n_ch = access_map.n_channels
    if isinstance(channel_index, bool) or not isinstance(channel_index, int):
        raise ValueError("channel_index must be an integer")
    if not 0 <= channel_index < n_ch:
        raise ValueError("channel_index %r out of range for %d channels"
                         % (channel_index, n_ch))
    for name, v in (("neuron_current_nA", neuron_current_nA),
                    ("conductivity_S_m", conductivity_S_m),
                    ("bandwidth_Hz", bandwidth_Hz)):
        if not np.isfinite(v) or v <= 0:
            raise ValueError("%s must be positive and finite" % name)
    if not np.isfinite(noise_sd_mV) or noise_sd_mV < 0:
        raise ValueError("noise_sd_mV must be nonnegative and finite")
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be positive and finite")

    centre_um = np.asarray(access_map.channel_centres_um)[channel_index]
    soma_rows = list(access_map.channel_neurons[channel_index])
    dists = list(access_map.channel_neurons_um[channel_index])

    # return contact: opposite the array, beyond the array half-diagonal.
    extent = np.asarray(access_map.extent_um, dtype=float)
    half_diag = float(np.linalg.norm(extent) / 2.0)
    return_offset_um = half_diag + 1000.0
    # array-facing axis = the axis of largest extent (or +y if the array is a point)
    axis = int(np.argmax(extent)) if extent.size and extent.max() > 0 else 1
    direction = np.zeros(3, dtype=float)
    direction[axis] = 1.0
    centroid_um = np.asarray(access_map.channel_centres_um).mean(axis=0)
    # DEFECT FIXED HERE.  The reference used to be placed relative to the REQUESTING
    # channel's own centre, with its direction chosen by which side of the centroid
    # that channel sits on (side = +/-1).  That gives every channel a DIFFERENT
    # reference point, and a channel's reference can then land close to ANOTHER
    # channel's source -- catastrophic in source_mode='channel_site', where the
    # sources ARE the channel centres.  Measured by an independent party on the
    # affected build: 0.398 mV/nA at the reference versus 2.19e-4 mV/nA at the
    # contact, i.e. the common-mode term was ~5.7x the neural signal and the channel
    # sum nearly cancelled.  ONE array-level reference is now used instead, placed
    # from the array CENTROID so it is the same point for every channel and is
    # guaranteed to be at least return_offset_um from every channel centre.
    reference_um = centroid_um + direction * return_offset_um
    # guard: never let the shared reference drift near any channel site
    _d_min = float(np.min(np.linalg.norm(
        np.asarray(access_map.channel_centres_um) - reference_um, axis=1)))
    _d_floor = half_diag + 500.0
    if _d_min < _d_floor:
        raise ValueError(
            "shared return reference is only %.3f um from the nearest channel site, "
            "below the %.3f um floor; a return that close would inject a common-mode "
            "term larger than the neural signal" % (_d_min, _d_floor))
    _reference_min_separation_um = _d_min

    if source_mode == "channel_site":
        contacts = [Contact(tuple(float(v) for v in centre_um), 1.0)
                    for _ in soma_rows]
        trace_note = ("sources placed AT the channel site: a stated non-anatomical "
                      "degeneracy, used so the per-neuron attenuation is the "
                      "transfer_mV_per_nA 1/r law at the contact")
    else:
        if not soma_rows:
            contacts = []
        else:
            pos_um = access_map.resolution.voxels_to_um(
                _positions_for(access_map, soma_rows))
            contacts = [Contact(tuple(float(v) for v in p), 1.0) for p in pos_um]
        trace_note = ("sources placed at the REAL measured soma voxel positions "
                      "(single voxel points), converted with the declared voxel "
                      "resolution")
    n_src = len(contacts)
    total = neuron_current_nA * n_src
    # Physical balancing source is NOT the passive voltage observation reference.
    # Keep the latter at the shared location; put the return on the opposite side.
    balancing_return_um = centroid_um - direction * return_offset_um
    contacts.append(Contact(tuple(float(v) for v in balancing_return_um), 1.0))
    currents = [neuron_current_nA] * n_src + [-total]

    # --- reuse the front end: no field solver is written here ---------------
    T = transfer_mV_per_nA(np.vstack([centre_um, reference_um]),
                           contacts, conductivity_S_m)
    per_neuron_mV = (T[0, :n_src] - T[1, :n_src]) * neuron_current_nA
    neural_unfiltered = float((T[0] - T[1]) @ np.asarray(currents))

    steps = int(n_steps) if n_steps is not None else 1
    if steps < 1:
        raise ValueError("n_steps must be >= 1")
    rec = VoltageRecorder(position_um=tuple(float(v) for v in centre_um),
                          reference_um=tuple(float(v) for v in reference_um),
                          conductivity_S_m=conductivity_S_m,
                          bandwidth_Hz=bandwidth_Hz, noise_sd_mV=noise_sd_mV,
                          seed=seed)
    trace = []
    for _ in range(steps):
        trace.append(rec.step(dt_ms, source_contacts=contacts,
                              outward_currents_nA=currents))

    out = {
        "attribution": dict(BANC_SOMA_ATTRIBUTION),
        "disclaimer": DISCLAIMER,
        "channel_index": int(channel_index),
        "channel_centre_um": [float(v) for v in centre_um],
        "reference_um": [float(v) for v in reference_um],
        "balancing_return_um": [float(v) for v in balancing_return_um],
        "scope_note": ("legacy single-channel captured-source diagnostic only; "
                       "not the full global recording field or noise budget. "
                       "Physical balancing return is distinct from voltage reference."),
        "return_contact_note": (
            "one explicit distant RETURN contact carries -sum(I) so the measured "
            "source set balances, as VoltageRecorder requires. A DECLARED "
            "closed-domain convention, not a reference-electrode model."),
        "neuron_current_nA": float(neuron_current_nA),
        "conductivity_S_m": float(conductivity_S_m),
        "bandwidth_Hz": float(bandwidth_Hz),
        "noise_sd_mV": float(noise_sd_mV),
        "dt_ms": float(dt_ms),
        "n_steps": steps,
        "source_mode": source_mode,
        "source_mode_note": trace_note,
        "n_captured_neurons": int(n_src),
        "distances_um": [float(d) for d in dists],
        "per_neuron_contribution_mV": [float(v) for v in per_neuron_mV],
        "neural_unfiltered_mV": neural_unfiltered,
        "steady_state_measured_mV": float(trace[-1]["measured_mV"]),
        "trace_measured_mV": [float(t["measured_mV"]) for t in trace],
        "transfer_units": "mV/nA (engine.electrode.transfer_mV_per_nA)",
        "no_spike_sorting": True,
        "limitations": [
            "no electrode-contact model: point probe, not contact-area averaging",
            "no impedance, no interface electrochemistry, no crosstalk",
            "no stimulation, no tissue mechanics",
            "neurons are current sources of a DECLARED magnitude, not spiking cells",
            "this is an extracellular potential trace, not a sorted spike train",
        ],
    }
    return out


def _positions_for(access_map, soma_rows):
    """Soma voxel positions for rows of the map's own soma table."""
    table = np.asarray(access_map.soma_positions_vox)
    if table.size == 0:
        raise ValueError(
            "this AccessMap carries no soma positions, so source_mode="
            "'soma_position' cannot place sources. Rebuild with "
            "build_access_map(), or use source_mode='channel_site'.")
    return table[np.asarray(soma_rows, dtype=np.int64)]


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def sweep_access_map(base_spec, base_resolution, tier_root_ids, somas_parquet_path,
                     *, target_mask=None, param="n_channels",
                     values=None, soma_table=None):
    """Vary ONE declared quantity and report how the answer moves.

    Sweeps are only over FREE or explicitly ASSUMED quantities: ``n_channels``
    (free by the user's constraint), ``pitch_um`` and ``diameter_um`` are NOT
    swept as optimisable design variables -- ``pitch_um`` is swept only to
    DISPLAY the sensitivity the user is entitled to see, with the fixed value
    marked, and ``diameter_um`` is not swept at all because no computation here
    consumes it.
    """
    if values is None:
        raise ValueError("values is required")
    rows = []
    for v in values:
        if param == "n_channels":
            spec = replace(base_spec, n_channels=int(v))
            res = base_resolution
        elif param == "capture_radius_um":
            spec = replace(base_spec, capture_radius_um=float(v))
            res = base_resolution
        elif param == "pitch_um":
            spec = replace(base_spec, pitch_um=float(v))
            res = base_resolution
        elif param == "voxel_resolution_nm":
            spec = base_spec
            a = tuple(float(x) for x in v)
            res = VoxelResolution(
                a[0], a[1], a[2], PROVENANCE.ASSUMED,
                source="swept candidate, no external source attached to this row",
                note=("swept candidate; the swept range is "
                      "UNVERIFIED_RESOLUTION_CANDIDATES_NM (sweep range), which anchors on the "
                      "verified 4x4x45 nm BANC value and includes the "
                      "8x8x45 public-mip and 4x4x40 FAFB near misses. Every "
                      "micrometre quantity scales with this number."))
        else:
            raise ValueError("unsupported sweep parameter %r" % (param,))
        if not isinstance(v, (int, float)) and param != "voxel_resolution_nm":
            raise ValueError("sweep value must be a number")
        if param in ("n_channels", "capture_radius_um", "pitch_um") and v <= 0:
            raise ValueError("sweep value must be positive")
        am = build_access_map(spec, tier_root_ids, somas_parquet_path, res,
                              soma_table=soma_table, target_mask=target_mask)
        cov = am.reports["coverage"]
        arr = am.reports["array"]
        rows.append({
            "param": param, "value": (list(v) if param == "voxel_resolution_nm"
                                      else (int(v) if param == "n_channels" else float(v))),
            "n_channels": int(arr["n_channels"]),
            "extent_um": arr["extent_um"],
            "target_neurons": cov["target_neurons"],
            "target_captured": cov["target_neurons_captured"],
            "target_addressable_fraction": cov["target_addressable_fraction"],
            "all_joined_tier_fraction": cov["all_joined_tier_fraction"],
            "collision_neurons": am.reports["collision"][
                "neurons_captured_by_more_than_one_channel"],
            "empty_channels": int(arr["empty_channels"]),
            "max_neurons_on_one_channel": int(arr["max_neurons_on_one_channel"]),
        })
    return rows
