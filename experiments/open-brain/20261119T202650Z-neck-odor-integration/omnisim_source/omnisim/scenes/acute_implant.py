"""Create or reload a post-insertion acute tissue state for recovery experiments."""

import argparse
from dataclasses import asdict, replace
import json
from math import ceil, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from ..composite_world import CompositeGpuWorld
from ..engine import FieldSpec, PorousDiffusion
from ..modules.array_damage import RadialInsertion
from ..modules.electrode_observation import AxonProbe, ElectrodeObservation
from ..modules.glial_remodeling import GlialRemodeling, GlialRemodelingConfig, GlialSurface
from ..modules.ion_feedback import IonFeedbackConfig
from ..modules.synaptic_plasticity import PlasticityConfig
from ..modules.tissue import Anatomy, BATH, ImplantLayout, Synapses, TISSUE, TissueFluid, build_anatomy, tissue_grid
from ..modules.tissue_dynamics import CHLORIDE, POTASSIUM, TissueDynamics
from ..native_transport import process_vram_bytes
from ..provenance import save_source_bundle, snapshot_digest, source_version
from ..world import Grid
from . import acute_checkpoint
from .hybrid_implant import BASE_M, INTRACELLULAR, LEVELS, MARKER, _NearArray, _rasterizer
from .implant_damage import measure

SHAFT = 2
FIRST_AXON = 3
ACUTE_DAMAGE = "implant.acute_damage"
FORMAT = "omnisim.acute.v1"
BLOCK_CELLS = 4


class _AcuteGeometry(_NearArray):
    def __init__(self, axes, anatomy, mechanic, tip_depth_m):
        super().__init__(axes, anatomy, fine_tip_depth_m=tip_depth_m,
                         near_m=max(mechanic.radius_m, mechanic.spatial_threshold_radius_m(mechanic.onset_Pa)))
        self.mechanic = mechanic
        self.reach_m = mechanic.spatial_threshold_radius_m(mechanic.onset_Pa)
        self.radius_m = max(self.reach_m, mechanic.radius_m)
        self.shaft_kernel = self._rasterize or _rasterizer()
        delta = axes[:, 1] - axes[:, 0]
        self.lengths = np.linalg.norm(delta, axis=1)
        self.direction = delta / self.lengths[:, None]
        self.damage = {}
        self.counts = {}

    def materials(self, grid, labels):
        labels = super().materials(grid, labels)
        labels[labels >= SHAFT] += 1
        if grid.spacing_m > 0.81e-6:
            return labels
        shaft_id = self.guides.rasterize(grid, self.radius_m, self.shaft_kernel)
        candidates = np.flatnonzero((shaft_id >= 0) & (labels != BATH))
        if not len(candidates):
            return labels
        ids = shaft_id.ravel()[candidates]
        indices = np.asarray(np.unravel_index(candidates, grid.shape)).T
        points = np.asarray(grid.origin_m) + (indices + 0.5) * grid.spacing_m
        rel = points - self.axes[ids, 0]
        depth = np.einsum("ij,ij->i", rel, self.direction[ids])
        radial2 = np.square(rel - depth[:, None] * self.direction[ids]).sum(axis=1)
        behind_tip = self.lengths[ids] - depth
        contact = (depth >= 0) & (behind_tip >= 0)
        tip_section = np.clip(behind_tip, 0, self.mechanic.radius_m)
        solid = contact & (radial2 <= tip_section * (2 * self.mechanic.radius_m - tip_section))
        energy = self.mechanic.spatial_strain_energy(np.sqrt(radial2), behind_tip)
        damage = np.clip((energy - self.mechanic.onset_Pa)
                         / (self.mechanic.failure_Pa - self.mechanic.onset_Pa), 0, 1)
        flat = labels.ravel()
        axon = flat[candidates] >= FIRST_AXON
        injured = contact & (damage > 0)
        failures = injured & (damage >= 1)
        remaining = injured & ~solid
        if remaining.any():
            self.damage[grid] = (candidates[remaining].astype(np.int32), damage[remaining])
        for name, mask in (("shaft_voxels", solid), ("shaft_replaced_axon_voxels", solid & axon),
                           ("nonshaft_injured_voxels", remaining),
                           ("nonshaft_failed_voxels", failures & ~solid)):
            count = int(mask.sum())
            tag = f"{grid.spacing_m*1e6:g}"
            totals = self.counts.setdefault(tag, {})
            totals[name] = totals.get(name, 0) + count
        flat[candidates[solid]] = SHAFT
        return labels


class AcuteWorld(CompositeGpuWorld):
    """Composite tissue initialized with an immutable, sparse mechanical history field."""

    def set_acute_history(self, blocks):
        self._acute = blocks
        self._specs[ACUTE_DAMAGE] = FieldSpec()

    def _pinned_tile(self, key):
        return key in getattr(self, '_acute', {})

    def _initial_block(self, name, block):
        archive = getattr(self, '_resume_archive', None)
        if archive is None or name in block.fields:
            amounts = super()._initial_block(name, block)
            release = getattr(self, '_injury_salt_density', 0.0)
            if (release and name in (POTASSIUM, CHLORIDE) and name not in block.fields
                    and block.key in self._acute):
                indices, injury = self._acute[block.key]
                in_fluid = block.materials.ravel()[indices] == TISSUE
                amounts.ravel()[indices[in_fluid]] += release * self._volume(block) * injury[in_fluid]
            return amounts
        with archive.open(f'{self._resume_indices[block.key]}/{self._resume_fields[name]}.npy') as entry:
            amounts = np.load(entry, allow_pickle=False)
        if (amounts.shape != block.materials.shape or amounts.dtype != np.float64 or not np.isfinite(amounts).all()
                or (self.field_specs[name].nonnegative and np.any(amounts < 0))):
            raise ValueError(f'invalid checkpoint amounts for {name}')
        return amounts

    def close(self):
        archive = getattr(self, '_resume_archive', None)
        if archive is not None:
            archive.close()
            self._resume_archive = None
        super().close()

    def _device_field_specs(self):
        return {name: spec for name, spec in super()._device_field_specs().items() if name != ACUTE_DAMAGE}

    def block_densities(self, name, key):
        if name != ACUTE_DAMAGE:
            return super().block_densities(name, key)
        self._check_open()
        values = np.zeros(self._blocks[key].materials.shape)
        if key in self._acute:
            index, damage = self._acute[key]
            values.ravel()[index] = damage
        return values

    def block_amounts(self, name, key):
        if name == ACUTE_DAMAGE:
            return self.block_densities(name, key) * self._volume(self._blocks[key])
        return super().block_amounts(name, key)

    def sample(self, name, point):
        if name != ACUTE_DAMAGE:
            return super().sample(name, point)
        self._check_open()
        block, cell = self._cell_at(point)
        if block.key not in self._acute:
            return 0.0
        ids, damage = self._acute[block.key]
        flat = np.ravel_multi_index(cell, block.materials.shape)
        i = np.searchsorted(ids, flat)
        return float(damage[i]) if i < len(ids) and ids[i] == flat else 0.0

    def total(self, name):
        if name == ACUTE_DAMAGE:
            self._check_open()
            return sum(float(values.sum()) * self._volume(self._blocks[key])
                       for key, (_, values) in self._acute.items())
        return super().total(name)

    def acute_debris_sites(self):
        """Reduce the injury voxels to one representative tissue site per damaged tile."""
        for key, (cells, damage) in self._acute.items():
            grid = self.block_grid(key)
            material = self._blocks[key].materials.ravel()[cells] == TISSUE
            if not material.any():
                continue
            available = np.where(material, damage, 0)
            cell = np.unravel_index(cells[available.argmax()], grid.shape)
            position = tuple(x + (i + .5) * grid.spacing_m for x, i in zip(grid.origin_m, cell))
            yield (*position, float(available.sum()) * grid.cell_volume_m3 * .02)

    def set_block_densities(self, name, key, density):
        if name == ACUTE_DAMAGE:
            raise ValueError("the acute history is read-only")
        super().set_block_densities(name, key, density)

    def add_amounts(self, position_m, changes):
        if ACUTE_DAMAGE in changes:
            raise ValueError("the acute history is read-only")
        super().add_amounts(position_m, changes)


def _surface_patches(axes, radius):
    """Stable sector IDs and area elements over cylindrical sides and hemispherical tips."""
    centers, normals, areas, owners = [], [], [], []
    azimuths = (np.arange(8) + 0.5) * np.pi / 4
    for shaft, (a, b) in enumerate(axes):
        length = np.linalg.norm(b - a)
        u = (b - a) / length
        cross = np.cross(u, (0, 1, 0)) if abs(u[1]) < .9 else np.cross(u, (1, 0, 0))
        x = cross / np.linalg.norm(cross)
        y = np.cross(u, x)
        radial = np.cos(azimuths)[:, None] * x + np.sin(azimuths)[:, None] * y
        stations = np.linspace(0, length - radius, 17)
        for left, right in zip(stations[:-1], stations[1:]):
            centers.extend(a + (left + right) * u / 2 + radius * radial)
            normals.extend(radial)
            areas.extend([radius * (right - left) * np.pi / 4] * 8)
            owners.extend([shaft] * 8)
        for cosine in (.25, .75):
            n = cosine * u + np.sqrt(1 - cosine**2) * radial
            centers.extend(b - radius * u + radius * n)
            normals.extend(n)
            areas.extend([radius**2 * np.pi / 8] * 8)
            owners.extend([shaft] * 8)
    return dict(surface_xyz_m=np.asarray(centers), surface_normal=np.asarray(normals),
                surface_area_m2=np.asarray(areas), surface_shaft_id=np.asarray(owners),
                surface_coverage=np.zeros(len(owners)))


def _pack_anatomy(anatomy):
    return {name: getattr(anatomy, name) for name in
            ("position_m", "cell_type", "layer", "process_tips_m", "axon_branch_m")} | {
                f"synapse_{name}": getattr(anatomy.synapses, name)
                for name in ("pre", "post", "position_m", "conductance_S", "delay_s")}


def _parse_anatomy(state):
    return Anatomy(*(state[name] for name in ("position_m", "cell_type", "layer", "process_tips_m", "axon_branch_m")),
                   Synapses(*(state[f"synapse_{name}"] for name in
                              ("pre", "post", "position_m", "conductance_S", "delay_s"))))


def save_acute_state(directory, *, tip_depth_um=157.6, max_cells=335_000_000,
                     implant_depth_um=320, injury_release_mm=0.0):
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError(directory)
    if not isfinite(injury_release_mm) or injury_release_mm < 0:
        raise ValueError('injury release concentration must be finite and nonnegative')
    if not 0 < tip_depth_um <= implant_depth_um:
        raise ValueError('the fine tip window must fit the implant depth')
    vertical = ImplantLayout(normal_depth_m=implant_depth_um * 1e-6)
    grid, root_material = tissue_grid((vertical, replace(vertical, tilt_deg=30)), spacing_m=BASE_M)
    anatomy = build_anatomy(grid)
    model = RadialInsertion()
    axes = vertical.axes_m()
    report, masks = measure(anatomy, vertical, model)
    geometry = _AcuteGeometry(axes, anatomy, model, tip_depth_um * 1e-6)
    world = AcuteWorld(grid, root_material, levels=LEVELS, select=geometry.refine,
                       materialize=geometry.materials, max_cells=max_cells, block_cells=BLOCK_CELLS,
                       leaf_limit=128, stream_initial=True)
    try:
        n = anatomy.neuron_count
        synapses = anatomy.synapses
        displaced = np.fromiter((world.material_at(point) == SHAFT for point in anatomy.position_m), bool)
        unavailable = masks["neurons"] | displaced[:n]
        endpoints_occupied = np.fromiter((world.material_at(point) == SHAFT for point in synapses.position_m), bool)
        unavailable_links = masks["connections"] | endpoints_occupied | unavailable[synapses.pre] | unavailable[synapses.post]
        glial_unavailable = masks["glia"] | displaced[n:]
        fluid = TissueFluid()
        biology = TissueDynamics(anatomy, fluid)
        biology.apply_lesion(neurons=unavailable, synapses=unavailable_links, glia=glial_unavailable)
        patches = _surface_patches(axes, model.radius_m)
        state = _pack_anatomy(anatomy) | patches | dict(
            axes_m=axes, root_material=root_material, soma_in_shaft=displaced, synapse_in_shaft=endpoints_occupied,
            neuron_unavailable=unavailable, glia_unavailable=glial_unavailable, connection_unavailable=unavailable_links,
            voltage_V=biology.voltage_V, glial_activation=biology.microglia_activation,
            neuron_first_failure_m=masks["soma_first_m"], glia_first_failure_m=masks["glia_first_m"],
            synapse_first_failure_m=masks["synapse_first_m"], connection_first_failure_m=masks["connection_first_m"],
        )
        by_resolution = {}
        rows, damaged, offsets = [], [], [0]
        with TemporaryDirectory(prefix=".acute-", dir=directory.parent) as scratch:
            target = Path(scratch)
            with ZipFile(target / "materials.zip", "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
                for i, (key, block) in enumerate(world._blocks.items()):
                    leaf = world.block_grid(key)
                    rows.append((block.level, *block.lower, *leaf.shape))
                    tag = f"{leaf.spacing_m * 1e6:g}"
                    by_resolution[tag] = by_resolution.get(tag, 0) + block.materials.size
                    with archive.open(f"{i}.npy", "w") as entry:
                        np.save(entry, block.materials, allow_pickle=False)
                    index, values = geometry.damage.get(leaf, (np.empty(0, dtype=np.int32), np.empty(0)))
                    damaged.append((index, values))
                    offsets.append(offsets[-1] + len(index))
            np.savez_compressed(target / "state.npz", **state, tile_locations=np.asarray(rows, dtype=np.int32),
                                damage_offsets=np.asarray(offsets, dtype=np.int64),
                                damage_cells=np.concatenate([a for a, _ in damaged]),
                                damage_values=np.concatenate([a for _, a in damaged]))
            manifest = {
                "format":FORMAT, "time_s":0, "layout":asdict(vertical), "mechanics":asdict(model),
                "damage_coordinates":"deformed-spatial-inverse-radial-map",
                "source":source_version(world.gpu_build),
                "source_bundle_sha256":save_source_bundle(target / 'source.tar.gz'),
                "max_cells":max_cells,
                "grid":asdict(grid), "levels":LEVELS, "block_cells":BLOCK_CELLS, "fluid":asdict(fluid),
                "solutes":[[name, concentration, diffusivity, fluid.COMPOSITION.get(name, {name:1})]
                           for name, concentration, diffusivity in fluid.SOLUTES],
                "active_cells":world.active_cells, "voxels_by_spacing_um":by_resolution,
                "acutely_damaged_tiles":len(geometry.damage), "voxel_damage":geometry.counts,
                "shaft_material_id":SHAFT, "first_axon_material_id":FIRST_AXON,
                "surface_patch_count":len(patches["surface_shaft_id"]),
                "neuron_unavailable":int(unavailable.sum()), "glia_unavailable":int(glial_unavailable.sum()),
                "connection_unavailable":int(unavailable_links.sum()),
                "injury_release_mM_per_damage":injury_release_mm,
                "resting_initialization":"pre-insertion resting membrane voltage; baseline ions with optional local paired salt release",
            }
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (target / "damage-history.json").write_text(json.dumps(report, indent=2) + "\n")
            target.rename(directory)
        return manifest
    finally:
        world.close()


def load_acute_state(directory, *, budget_mib=14500, reserve_mib=768,
                     plasticity: PlasticityConfig | None = None, ion_feedback: IonFeedbackConfig | None = None,
                     glial_remodeling: GlialRemodelingConfig | None = None, resume=None):
    """Rebuild a HIP world, severed network and surface geometry at implant age zero."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["format"] != FORMAT or manifest["time_s"] != 0
            or manifest["shaft_material_id"] != SHAFT or manifest["first_axon_material_id"] != FIRST_AXON):
        raise ValueError("unsupported acute snapshot")
    with np.load(directory / "state.npz", allow_pickle=False) as archive:
        state = dict(archive.items())
    grid = Grid(**manifest["grid"])
    restored, runtime = acute_checkpoint.read(resume, directory) if resume is not None else (None, None)
    if restored is not None:
        saved = acute_checkpoint.configs(restored)
        for name, config in (('plasticity', plasticity), ('ion_feedback', ion_feedback), ('glial_remodeling', glial_remodeling)):
            if config is not None and config != saved[name]:
                raise ValueError(f'checkpoint {name} differs from the requested configuration')
        plasticity, ion_feedback, glial_remodeling = (saved[name] for name in ('plasticity', 'ion_feedback', 'glial_remodeling'))
    tile_locations = state['tile_locations'] if restored is None else runtime['tile_locations']
    active_cells = manifest['active_cells'] if restored is None else restored['active_cells']

    def tiles():
        with ZipFile(directory / 'materials.zip' if resume is None else Path(resume) / 'tiles.zip') as archive:
            for i, row in enumerate(tile_locations):
                with archive.open(f'{i}.npy' if resume is None else f'{i}/material.npy') as entry:
                    values = np.load(entry, allow_pickle=False)
                    if values.shape != tuple(row[4:]):
                        raise ValueError("tile dimensions differ from the manifest")
                    yield int(row[0]), tuple(map(int, row[1:4])), values

    world = AcuteWorld(grid, state["root_material"], levels=tuple(manifest["levels"]), archived_tiles=tiles(),
                       max_cells=manifest.get('max_cells', max(manifest['active_cells'], active_cells)),
                       block_cells=manifest["block_cells"], stream_initial=True,
                       budget_bytes=budget_mib * 2**20, reserve_bytes=reserve_mib * 2**20)
    try:
        if world.active_cells != active_cells:
            raise ValueError("reloaded cell count differs from the manifest")
        offsets = state["damage_offsets"]
        damage_ids, damages = state["damage_cells"], state["damage_values"]
        if (offsets.shape != (len(world.block_keys) + 1,) or offsets[0] != 0
                or offsets[-1] != len(damage_ids) or len(damage_ids) != len(damages)
                or np.any(np.diff(offsets) < 0)
                or not np.all(np.isfinite(damages) & (damages > 0) & (damages <= 1))):
            raise ValueError("invalid archived damage field")
        damage_ids.setflags(write=False)
        damages.setflags(write=False)
        history = {}
        for key, a, b in zip(world.block_keys, offsets[:-1], offsets[1:]):
            if b == a:
                continue
            indices = damage_ids[a:b]
            if indices[0] < 0 or indices[-1] >= world._blocks[key].materials.size or np.any(np.diff(indices) <= 0):
                raise ValueError("invalid damaged cell indices")
            history[key] = indices, damages[a:b]
        world.set_acute_history(history)
        fluid = TissueFluid(**manifest["fluid"])
        world._injury_salt_density = manifest.get('injury_release_mM_per_damage', 0.0) * fluid.volume_fraction
        medium = PorousDiffusion(TISSUE, fluid.volume_fraction, fluid.tortuosity, SHAFT)
        for solute, concentration, d, formula in manifest["solutes"]:
            name = fluid.field(solute)
            if solute in ("K", "Cl"):
                world.add_field(name, FieldSpec(diffusivity_m2_s=d, porous=medium, elements=formula),
                                initial_density=fluid.fraction(state["root_material"]) * concentration)
            else:
                world.add_equilibrium_field(name, FieldSpec(porous=medium, elements=formula), concentration)
        world.add_field(MARKER, FieldSpec(diffusivity_m2_s=1.96e-9, porous=medium))
        world.add_field(INTRACELLULAR, FieldSpec(diffusivity_m2_s=.2e-9, isolated_materials_from=FIRST_AXON))
        if restored is not None:
            if set(restored['fields']) != world._device_field_specs().keys():
                raise ValueError('checkpoint fields do not match the acute scenario')
            world._resume_indices = {key: i for i, key in enumerate(world.block_keys)}
            world._resume_fields = {name: i for i, name in enumerate(restored['fields'])}
            world._resume_archive = ZipFile(Path(resume) / 'tiles.zip')
        anatomy = _parse_anatomy(state)
        cut_axons = None
        if plasticity is not None:
            mechanics = RadialInsertion(**manifest["mechanics"])
            cut_axons = np.isfinite(mechanics.first_depths(anatomy.position_m[:anatomy.neuron_count],
                                                           anatomy.axon_branch_m, state["axes_m"], mechanics.failure_Pa))
        biology = TissueDynamics(anatomy, fluid, damage_field=ACUTE_DAMAGE, plasticity=plasticity,
                                 first_axon_material=FIRST_AXON, disconnected_axons=cut_axons,
                                 ion_feedback=ion_feedback, blocked_axon_material=SHAFT)
        biology.apply_lesion(neurons=state["neuron_unavailable"], synapses=state["connection_unavailable"],
                             glia=state["glia_unavailable"])
        biology.voltage_V[:] = state["voltage_V"]
        biology.microglia_activation[:] = state["glial_activation"]
        world.add_module("tissue", biology)
        if glial_remodeling is not None:
            surface = GlialSurface(state["surface_xyz_m"], state["surface_normal"],
                                   state["surface_area_m2"], state["surface_coverage"])
            debris = list(world.acute_debris_sites()) if restored is None else runtime['glia_debris_seeds']
            world.add_module("glia", GlialRemodeling(biology, glial_remodeling, surface=surface, debris=debris))
        if restored is not None:
            acute_checkpoint.restore(world, biology, state, restored, runtime)
            if 'electrode_setup' in restored:
                electrode = build_observation(world, state, restored['electrode_setup'])
                electrode.events = restored['electrode_pending']
        return world, biology, manifest, state
    except BaseException:
        world.close()
        raise


def verify_reload(directory, *, budget_mib=14500, reserve_mib=768, duration_s=20e-6,
                  plasticity=None, ion_feedback=None, glial_remodeling=None):
    # Exercise registration and two slow updates within this brief device diagnostic.
    if glial_remodeling is not None:
        glial_remodeling = replace(glial_remodeling, update_interval_s=duration_s / 2)
    world, biology, manifest, state = load_acute_state(directory, budget_mib=budget_mib, reserve_mib=reserve_mib,
                                                       plasticity=plasticity, ion_feedback=ion_feedback,
                                                       glial_remodeling=glial_remodeling)
    try:
        world.initialize_device()
        axes = state["axes_m"]
        tip = axes[0, 1]
        shaft_sample = tip + (0, 0, -6e-6)
        if world.material_at(shaft_sample) != SHAFT or world.sample_concentration(POTASSIUM, shaft_sample):
            raise AssertionError("shaft was not restored as a closed material boundary")
        offsets = np.linspace(2.05e-6, 3.5e-6, 30)
        probe = next(tuple(shaft_sample + (r, 0, 0)) for r in offsets
                     if world.material_at(shaft_sample + (r, 0, 0)) == TISSUE)
        dose = 1e-20
        world.add_amounts(probe, {MARKER:dose})
        step = world.block_grid(world.block_at(probe)).cell_volume_m3
        before = world.sample(MARKER, probe) * step
        if ion_feedback is not None:
            live = np.flatnonzero(~state["neuron_unavailable"])
            position = biology.anatomy.position_m
            driven = int(live[np.argmin(np.square(position[live] - probe).sum(axis=1))])
            reference = {name: world.total(name) for name in (POTASSIUM, CHLORIDE)}
            # Force one diagnostic spike inside the short capacity check.
            biology.set_current(driven, 2 * 120e-12 * .015 / duration_s)
        world.step(duration_s)
        after = world.sample(MARKER, probe) * step
        conservation = abs(world.total(MARKER) / dose - 1)
        if not 0 < after < before or conservation > 1e-12:
            raise AssertionError("the restored material network did not transport and conserve the verification tracer")
        report = dict(active_cells=world.active_cells, runtime_fields=world.gpu_field_buffers,
                    lesion_connections=int(state["connection_unavailable"].sum()),
                    shaft_material=int(world.material_at(shaft_sample)),
                    acute_damage_integral_m3=world.total(ACUTE_DAMAGE), simulated_s=world.time_s,
                    marker_fraction_outside_seed=1-after/before, relative_marker_balance_error=conservation,
                    microglia_responding=int(np.count_nonzero(biology.microglia_activation)),
                    gpu_peak_bytes=world.peak_gpu_bytes, process_vram_bytes=process_vram_bytes())
        if biology.plasticity is not None:
            contacts = biology.plasticity.contacts
            if any((c.phase == "lesioned") != cut for c, cut in zip(contacts, state["connection_unavailable"])):
                raise AssertionError("plastic synapse graph did not preserve the archived lesion")
            report["plastic_graph_edges"] = len(contacts)
            report["plastic_graph_active_edges"] = sum(c.phase == "active" for c in contacts)
        if ion_feedback is not None:
            balance = {name: (world.total(name) + biology.intracellular_salt_change_mol - value) / value
                       for name, value in reference.items()}
            released = biology.ion_exchange.released_mol
            if released <= 0 or max(map(abs, balance.values())) > 1e-11:
                raise AssertionError("the diagnostic neuronal ion exchange failed its material balance")
            report.update(driven_neuron=driven, ion_release_mol=released, relative_ion_balance_error=balance,
                          driven_neuron_K_mol_m3=world.sample_concentration(POTASSIUM, position[driven]))
        if biology.glial_remodeling is not None:
            glia = biology.glial_remodeling
            if glia.ticks != 2 or not np.any(glia.activation > 0):
                raise AssertionError("slow glia did not consume the archived injury")
            report["glial_remodeling"] = glia.summary() | {"diagnostic_interval_s": glial_remodeling.update_interval_s}
        return report
    finally:
        world.close()


def build_observation(world, state, setup):
    observation = ElectrodeObservation(state['surface_xyz_m'], [AxonProbe(**route) for route in setup['routes']],
                                       damage_field=ACUTE_DAMAGE, **setup.get('parameters', {}))
    observation.setup = setup
    world.add_module('electrode', observation)
    return observation


def run_acute_state(directory, *, duration_s, step_s, observe=None, stimulus=None, focus=None,
                    checkpoint=None, electrode_protocol=None, adapt_regions=None, **options):
    """Continue from the implant snapshot and optionally observe completed world steps."""
    if not all(isfinite(value) and value > 0 for value in (duration_s, step_s)):
        raise ValueError('run duration and step must be finite and positive')
    if stimulus is not None:
        neuron, current_A, until_s = stimulus
        if not isfinite(current_A) or not isfinite(until_s) or not 0 < until_s <= duration_s:
            raise ValueError('stimulus must end inside the recorded duration')
    started = perf_counter()
    world, tissue, manifest, state = load_acute_state(directory, **options)
    try:
        alive = np.flatnonzero(~state['neuron_unavailable'])
        if focus is None:
            focus = int(alive[np.argmin(np.square(tissue.anatomy.position_m[alive] - state['axes_m'][0, 1]).sum(axis=1))])
        if focus not in alive or (stimulus is not None and neuron not in alive):
            raise ValueError('the focus and stimulated neuron must be live')
        world.initialize_device()
        ready = perf_counter()
        start_s = world.time_s
        if electrode_protocol is not None:
            setup = {key: value for key, value in electrode_protocol.items() if key != 'pulses'}
            electrode = world._modules.get('electrode')
            if electrode is None:
                electrode = build_observation(world, state, setup)
            elif electrode.setup != setup:
                raise ValueError('electrode configuration differs from the checkpoint')
            for pulse in electrode_protocol.get('pulses', ()):
                electrode.pulse(world, pulse['source'], pulse['sink'], pulse['current_A'],
                                start_s + pulse['time_s'], first_axon_material=FIRST_AXON)
        solver_seconds = capture_seconds = 0.0
        if observe is not None:
            tick = perf_counter()
            observe(world, tissue, state, focus, 0, False, ())
            capture_seconds += perf_counter() - tick
        steps = ceil(duration_s / step_s)
        endpoints = {duration_s * i / steps for i in range(1, steps)} | {duration_s}
        if stimulus is not None and not any(abs(value - until_s) <= 1e-12 * duration_s for value in endpoints):
            endpoints.add(until_s)
        spike_events, electrode_events = [], []
        for index, end_s in enumerate(sorted(endpoints), 1):
            if stimulus is not None:
                active = world.time_s - start_s < until_s - 1e-12 * duration_s
                tissue.set_current(neuron, current_A if active else 0.0)
            tick = perf_counter()
            world.step(start_s + end_s - world.time_s)
            if adapt_regions is not None:
                regions = adapt_regions(world, tissue)
                if regions is not None:
                    world.adapt(regions)
            solver_seconds += perf_counter() - tick
            if stimulus is not None and end_s >= until_s:
                tissue.set_current(neuron, 0.0)
            events = world.drain_events()
            electrode = world._modules.get('electrode')
            if electrode is not None:
                for event in events:
                    if event.kind == 'spike' and event.source == 'tissue':
                        electrode.observe_spike(world, event.data['neuron'], event.time_s, FIRST_AXON)
                electrode.step(world, 'electrode', 0.0)
                events.extend(world.drain_events())
            spike_events.extend(dict(time_s=e.time_s, neuron=e.data['neuron']) for e in events if e.kind == 'spike')
            electrode_events.extend(dict(time_s=e.time_s, kind=e.kind, **e.data) for e in events if e.source == 'electrode')
            if observe is not None:
                tick = perf_counter()
                observe(world, tissue, state, focus, index, end_s == duration_s, events)
                capture_seconds += perf_counter() - tick
        advancing = perf_counter() - ready
        checkpoint_started = perf_counter()
        if checkpoint is not None:
            acute_checkpoint.save(checkpoint, directory, world, tissue, state)
        return dict(simulated_s=world.time_s - start_s, world_time_s=world.time_s, steps=len(endpoints),
                    stimulus=stimulus, focus_neuron=focus,
                    shafts=manifest['layout']['count'], active_cells=world.active_cells, spike_events=spike_events,
                    runtime_fields=world.gpu_field_buffers, gpu_peak_bytes=world.peak_gpu_bytes,
                    process_vram_bytes=process_vram_bytes(), setup_s=ready - started, advance_s=advancing,
                    solver_s=solver_seconds, capture_s=capture_seconds, source=source_version(world.gpu_build),
                    checkpoint_s=perf_counter() - checkpoint_started if checkpoint is not None else 0.0,
                    electrode_events=electrode_events, mesh_revision=world.mesh_revision,
                    acute_input=snapshot_digest(directory), initial_source=manifest.get('source'),
                    plasticity=tissue.plasticity is not None, ion_feedback=tissue.ion_exchange is not None,
                    glial_remodeling=None if tissue.glial_remodeling is None else tissue.glial_remodeling.summary())
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("save", "verify", "run"))
    parser.add_argument("state", type=Path)
    parser.add_argument("--budget-mib", type=int, default=14500)
    parser.add_argument("--reserve-mib", type=int, default=768)
    parser.add_argument("--plasticity", action="store_true", help="enable the plastic synapse graph")
    parser.add_argument("--ion-feedback", action="store_true", help="enable exchange with the existing ion fields")
    parser.add_argument("--glia-remodeling", action="store_true", help="enable structural glial dynamics")
    parser.add_argument('--duration-us', type=float, default=120, help='simulated duration in run mode')
    parser.add_argument('--step-us', type=float, default=10, help='maximum outer world step in run mode')
    parser.add_argument('--implant-depth-um', type=float, default=320, help='synthetic normal insertion depth in save mode')
    parser.add_argument('--fine-tip-depth-um', type=float, default=157.6, help='fine mesh window behind the tips in save mode')
    parser.add_argument('--injury-release-mm', type=float, default=0, help='optional effective local KCl release per unit damage in save mode')
    parser.add_argument('--focus-neuron', type=int)
    parser.add_argument('--stimulus-neuron', type=int, help='optionally apply a current clamp in run mode')
    parser.add_argument('--stimulus-nA', type=float, default=0)
    parser.add_argument('--stimulus-us', type=float, default=20)
    parser.add_argument('--record-dir', type=Path, help='write the completed solver frames and an MP4 in run mode')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--checkpoint', type=Path, help='save an exact end-of-run checkpoint')
    parser.add_argument('--resume', type=Path, help='continue from a checkpoint for this acute snapshot')
    parser.add_argument('--electrode-protocol', type=Path, help='JSON routes and surface-patch pulses')
    args = parser.parse_args()
    options = dict(budget_mib=args.budget_mib, reserve_mib=args.reserve_mib,
                   plasticity=PlasticityConfig() if args.plasticity else None,
                   ion_feedback=IonFeedbackConfig() if args.ion_feedback else None,
                   glial_remodeling=GlialRemodelingConfig() if args.glia_remodeling else None)
    if args.mode == 'run':
        stimulus = None if args.stimulus_neuron is None else (
            args.stimulus_neuron, args.stimulus_nA * 1e-9, args.stimulus_us * 1e-6)
        if args.record_dir is not None:
            if min(args.fps, args.frame_stride) < 1:
                parser.error('recording fps and stride must be positive')
            from .acute_record_render import check_encoder, render
            from .acute_recording import AcuteRecording

            check_encoder()
            observer = AcuteRecording(args.record_dir, stride=args.frame_stride)
        else:
            observer = None
        result = run_acute_state(args.state, duration_s=args.duration_us * 1e-6, step_s=args.step_us * 1e-6,
                                 observe=observer, stimulus=stimulus, focus=args.focus_neuron,
                                 checkpoint=args.checkpoint, resume=args.resume,
                                 electrode_protocol=None if args.electrode_protocol is None else
                                 json.loads(args.electrode_protocol.read_text()), **options)
        if observer is not None:
            path = observer.write()
            result['recorded_frames'] = len(observer.frames)
            report_path = args.record_dir / 'implant-run.json'
            result['render_status'] = 'pending'
            report_path.write_text(json.dumps(result, indent=2) + '\n')
            result = render(path, fps=args.fps)
    elif args.record_dir is not None:
        parser.error('recording requires run mode')
    elif args.mode == 'save':
        result = save_acute_state(args.state, implant_depth_um=args.implant_depth_um,
                                  tip_depth_um=args.fine_tip_depth_um, injury_release_mm=args.injury_release_mm)
    else:
        result = verify_reload(args.state, **options)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
