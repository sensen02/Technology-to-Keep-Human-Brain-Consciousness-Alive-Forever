"""engine.profile -- organism/scale profiles that encode verified constraints.

WHY THIS EXISTS
---------------
Several of this project's hard-won findings are *negative* constraints: they say
what must NOT be modelled a certain way.  Recorded in `MEASURED_ANCHORS.md` and
`ORGANISM_SCALE_PLAN.md`:

  * **Drosophila has no vasculature.**  Insects exchange gas through tracheae
    and tracheoles, not capillary beds.  A Krogh cylinder model is the wrong
    category for a fly.  (Verified: the locust tracheole measurements in the
    records are tracheole radius 0.5 um, muscle-cylinder radius 6.5 um,
    spacing ~13 um, i.e. ~150x denser than mammalian capillaries -- and the
    records explicitly note Drosophila itself is NOT in that data set.)
  * **Hydra has no vasculature either**; transport is diffusive.
  * **Stages must not be mixed.**  The notum anchors span 12-13.5 hAPF,
    18-26 hAPF and embryo stage 5.

These are exactly the errors that are easy to make and hard to notice.  So an
organism profile carries them as machine-enforced guards: a pipeline that tries
to attach a vessel source to a profile without vasculature is refused, and the
profile's caveats are injected into the run record so they cannot be dropped
from the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["OrganismProfile", "ProfileViolation",
           "DROSOPHILA_NOTUM", "HYDRA", "MAMMALIAN_TISSUE", "PROFILES"]

# port names that only make sense if the organism actually has a circulation
VESSEL_PORTS = ("vessel_source", "perfusion", "blood_flow", "capillary_o2",
                "krogh_source", "vascular_network", "vessel_geometry")

TRACHEAL_PORTS = ("tracheole_source", "tracheal_flux", "spiracle_input")


class ProfileViolation(RuntimeError):
    pass


@dataclass
class OrganismProfile:
    name: str
    species: str
    stage: str
    dimensionality: int
    has_vasculature: bool
    gas_exchange: str                     # "tracheal" | "capillary" | "diffusive"
    cell_area_um2: float | None = None
    length_scale_um: float | None = None
    notes: list = field(default_factory=list)
    cross_species_notes: list = field(default_factory=list)
    caveats: list = field(default_factory=list)   # MUST appear in any report

    # ------------------------------------------------------------------
    def forbidden_ports(self):
        bad = []
        if not self.has_vasculature:
            bad += list(VESSEL_PORTS)
        if self.gas_exchange != "tracheal":
            bad += list(TRACHEAL_PORTS)
        return bad

    def validate_pipeline(self, pipeline):
        """Refuse a pipeline that contradicts this organism's physiology.

        Reads the layers' DECLARED `provides` directly rather than asking the
        pipeline for its wired ports: the wired port table is only populated by
        `wire()`, so querying it made this guard silently pass on an unwired
        pipeline.  A guard that fails open is worse than no guard.
        """
        provided = set()
        for layer in pipeline.layers:
            if getattr(layer, "enabled", True):
                provided |= set(layer.provides)
        bad = sorted(provided & set(self.forbidden_ports()))
        if bad:
            why = ("this organism has NO vasculature" if not self.has_vasculature
                   else f"gas exchange is {self.gas_exchange}, not tracheal")
            raise ProfileViolation(
                f"profile {self.name!r} ({self.species}, {self.stage}) cannot "
                f"attach layers providing {bad}: {why}. "
                f"{' '.join(self.caveats)}")
        return True

    def required_caveats(self):
        return list(self.caveats)

    def to_dict(self):
        return {
            "name": self.name, "species": self.species, "stage": self.stage,
            "dimensionality": self.dimensionality,
            "has_vasculature": self.has_vasculature,
            "gas_exchange": self.gas_exchange,
            "cell_area_um2": self.cell_area_um2,
            "length_scale_um": self.length_scale_um,
            "notes": self.notes, "cross_species_notes": self.cross_species_notes,
            "caveats": self.caveats, "forbidden_ports": self.forbidden_ports(),
        }


# ----------------------------------------------------------------------
# The organism this project actually started from.
# `PROTOCOL.md` line 1: "400-cell Drosophila epithelial wound calcium prototype"
# Source: Stevens et al., Mol Biol Cell 2023 (PMC10208100);
#         code: github.com/mshutson/wound-calcium-LRCa (MIT)
# ----------------------------------------------------------------------
DROSOPHILA_NOTUM = OrganismProfile(
    name="drosophila_notum",
    species="Drosophila melanogaster",
    stage="pupal notum, 12-13.5 hAPF",
    dimensionality=2,
    has_vasculature=False,
    gas_exchange="tracheal",
    cell_area_um2=55.6,
    length_scale_um=7.5,
    notes=[
        "Mean area of a 4-cell group 222.3 um^2 -> 55.6 um^2 per cell "
        "(notum, 12-13.5 hAPF; Curran 2017).",
        "Epithelial thickness 7.5-10 um (notum 15-18 hAPF; Pinheiro 2017): "
        "this is a thin monolayer, so a 2D apical model is a defensible "
        "approximation, but it is NOT a 3D tissue.",
        "T1 rate 8.5 +/- 1.5e-4 per min per junction (notum 12-13.5 hAPF).",
        "No planar myosin anisotropy at 12-13.5 hAPF (measured null: "
        "Spearman -0.0155, p=0.6875, n=688).",
    ],
    cross_species_notes=[
        "The tracheole geometry in the records (radius 0.5 um, muscle cylinder "
        "6.5 um, spacing ~13 um) is from LOCUST, not Drosophila; the records "
        "state Drosophila itself is not in that data set.",
        "Absolute junction tension 44 +/- 22 pN is from EMBRYO STAGE 5, not "
        "the notum; the notum has no absolute tension measurement at all.",
        "Bending modulus 1.9e-13 N m is from MDCK (mammalian) monolayers.",
    ],
    caveats=[
        "果蝇没有血管系统，气体交换靠气管/气管小管；不得对其使用毛细血管 Krogh 模型。",
        "notum 上没有任何绝对张力(nN)实测值，因此不得声称绝对力。",
        "所有实测锚点来自不同发育阶段（12-13.5 hAPF / 18-26 hAPF / 胚胎 stage 5），"
        "每次比较都必须声明阶段。",
    ],
)

DROSOPHILA_CNS = OrganismProfile(
    name="drosophila_cns",
    species="Drosophila melanogaster",
    stage="adult (FlyWire FAFB v783 brain; MANC/BANC add the nerve cord)",
    dimensionality=3,
    has_vasculature=False,
    gas_exchange="tracheal",
    cell_area_um2=None,
    length_scale_um=None,
    notes=[
        "FlyWire FAFB v783: 139,255 neurons, 3,732,460 unique connections, "
        "50,666,648 chemical synapses (measured in this project from the "
        "downloaded connections_princeton product).",
        "nt_type is the presynaptic neuron's transmitter and is CONSISTENT per "
        "neuron in this snapshot (verified: 0 presynaptic neurons show more "
        "than one label), so each edge's excitatory/inhibitory sign comes from "
        "data, not from a modelling choice.",
        "The adult fly CNS is supplied by the tracheal system, not by blood "
        "vessels; there is no blood-brain barrier of the vertebrate kind and "
        "the haemolymph is an open circulation.",
    ],
    cross_species_notes=[
        "Vertebrate cortical parameters (E/I ratio, firing rates, membrane "
        "time constants, conduction velocities) must NOT be transferred to the "
        "fly without saying so.",
        "The connectome gives STRUCTURE ONLY: no synaptic weights, no release "
        "probabilities, no channel kinetics, no neuromodulation.  'Structure is "
        "not function' -- this is the single most important limit here.",
    ],
    caveats=[
        "连接组只给「谁连谁」与递质类型；**不给突触权重、释放概率、离子通道、"
        "神经调质**。因此接上真连接组**不等于**得到一个会工作的脑。",
        "权重标度是 illustrative 自由参数，它直接决定网络是静默还是活跃。",
        "LIF 不是神经元：无树突、无通道动力学、无适应。",
        "果蝇用气管供气，没有血管；不得套用脊椎动物的血脑屏障或神经血管耦合。",
    ],
)

HYDRA = OrganismProfile(
    name="hydra",
    species="Hydra vulgaris",
    stage="adult",
    dimensionality=3,
    has_vasculature=False,
    gas_exchange="diffusive",
    cell_area_um2=None,
    length_scale_um=None,
    notes=["Has a single-cell atlas (~25k cells whole animal) but the records "
           "note a transcriptome is not a set of physical parameters.",
           "Tissue stiffness has a ~100x conflict in the literature "
           "(whole tissue 440 +/- 330 Pa vs mesoglea only 20-120 kPa); the two "
           "are different quantities and must not be mixed."],
    caveats=["水螅没有血管系统，物质靠扩散；不得声称模拟了其循环。"],
)

MAMMALIAN_TISSUE = OrganismProfile(
    name="mammalian_tissue",
    species="Mammalia (generic)",
    stage="unspecified",
    dimensionality=3,
    has_vasculature=True,
    gas_exchange="capillary",
    cell_area_um2=None,
    length_scale_um=None,
    notes=["The only profile in this project for which a Krogh-type capillary "
           "oxygen model is the right category.  It is also the least anchored: "
           "no mammalian tissue parameters have been verified in this project."],
    caveats=["哺乳动物组织的参数在本项目中几乎没有核实过；使用它是类别正确但数值未验证。"],
)

PROFILES = {p.name: p for p in (DROSOPHILA_NOTUM, DROSOPHILA_CNS,
                                HYDRA, MAMMALIAN_TISSUE)}
