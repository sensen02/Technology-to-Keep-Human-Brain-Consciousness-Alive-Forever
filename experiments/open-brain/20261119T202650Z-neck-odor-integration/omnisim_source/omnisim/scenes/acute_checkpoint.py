"""Save and restore a completed step of the implant scenario without pickling modules."""

from dataclasses import asdict, fields, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from ..modules.glial_remodeling import GlialRemodelingConfig
from ..modules.ion_feedback import IonFeedbackConfig
from ..modules.synaptic_plasticity import PlasticityConfig, SynapticContact
from ..provenance import snapshot_digest, source_version

FORMAT = 'omnisim.acute.checkpoint.v1'
TISSUE_ARRAYS = ('voltage_V', 'excitatory_S', 'inhibitory_S', '_rest_V', '_inhibitory_reversal_V',
                 '_drive_A', '_refractory_until', '_glial_K_mol', '_glial_volume_fraction',
                 'microglia_activation', '_inactive_glia', '_inactive_neurons', '_disconnected_synapses', 'deliveries')
GLIA_ARRAYS = ('tip_m', 'activation', 'engulfed_m3', 'parent_ids', 'birth_time_s',
               '_cycle', '_exposure_s', 'debris_m3')
ANATOMY_ARRAYS = ('position_m', 'process_tips_m', 'cell_type', 'layer')
CONTACT_OPTIONALS = ('weak_since_s', 'stable_since_s')


def save(directory, base, world, tissue, state):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    world._check_ready()
    directory.parent.mkdir(parents=True, exist_ok=True)
    p, ion, glia = tissue.plasticity, tissue.ion_exchange, tissue.glial_remodeling
    metadata = dict(format=FORMAT, time_s=world.time_s, mesh_revision=world.mesh_revision,
                    base=snapshot_digest(base), source=source_version(world.gpu_build),
                    fields=list(world._device_field_specs()), active_cells=world.active_cells,
                    plasticity=asdict(p.config) if p is not None else None,
                    ion_feedback=asdict(ion._config) if ion is not None else None,
                    glial_remodeling=asdict(glia.config) if glia is not None else None)
    if 'electrode' in world._modules:
        electrode = world._modules['electrode']
        metadata.update(electrode_setup=electrode.setup, electrode_pending=electrode.events)
    arrays = {name: getattr(tissue, name) for name in TISSUE_ARRAYS}
    arrays.update({name: getattr(tissue.anatomy, name) for name in ANATOMY_ARRAYS})
    arrays['surface_coverage'] = state['surface_coverage']
    arrays['pending_times'] = np.asarray([entry[0] for entry in tissue._pending])
    arrays['pending_edges'] = np.asarray([entry[1] for entry in tissue._pending], dtype=int)
    if p is not None:
        for field in fields(SynapticContact):
            values = [getattr(c, field.name) for c in p.contacts]
            arrays['contact_' + field.name] = np.asarray(values, dtype=float if field.name in CONTACT_OPTIONALS else None)
        arrays['plastic_last_post'] = p._last_post_s
        arrays['plastic_activity'] = p._activity
        arrays['plastic_mobile'] = np.asarray(list(p._mobile), dtype=int)
        arrays['plastic_weak'] = np.asarray(list(p._weak), dtype=int)
        arrays['plastic_changing'] = np.asarray(list(p._changing), dtype=int)
        arrays['plastic_active_order'] = np.asarray([edge for edges in p._outgoing for edge in edges], dtype=int)
        arrays['plastic_incoming_order'] = np.asarray([edge for edges in p._incoming for edge in edges], dtype=int)
        metadata['plastic_next_s'] = p.next_update_s
    if ion is not None:
        arrays['ion_withdrawal'] = ion._withdrawal
        arrays['ion_recovering'] = np.asarray(sorted(ion._recovering), dtype=int)
        metadata['ion_state'] = [ion.released_mol, ion.recovered_mol, ion._last_recovery_s]
    if glia is not None:
        arrays.update({'glia_' + name: getattr(glia, name) for name in GLIA_ARRAYS})
        # The same debris targets can be retained even when nearby tiles are retiled.
        arrays['glia_debris_seeds'] = np.column_stack((glia._debris.points, glia._original_debris_m3))
        metadata['glia_state'] = [glia.ticks, glia.migrations, glia.births, glia._start_s]
    with TemporaryDirectory(prefix='.acute-checkpoint-', dir=directory.parent) as scratch:
        target = Path(scratch)
        with ZipFile(target / 'tiles.zip', 'w', compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for i, (key, tile) in enumerate(world._blocks.items()):
                with archive.open(f'{i}/material.npy', 'w', force_zip64=True) as entry:
                    np.save(entry, tile.materials, allow_pickle=False)
                for j, name in enumerate(metadata['fields']):
                    with archive.open(f'{i}/{j}.npy', 'w', force_zip64=True) as entry:
                        np.save(entry, world.block_amounts(name, key), allow_pickle=False)
        arrays['tile_locations'] = np.array([(tile.level, *tile.lower, *tile.materials.shape)
                                             for tile in world._blocks.values()], dtype=np.int32)
        np.savez_compressed(target / 'runtime.npz', **arrays)
        (target / 'checkpoint.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        target.rename(directory)


def read(directory, base):
    path = Path(directory)
    metadata = json.loads((path / 'checkpoint.json').read_text())
    if metadata['format'] != FORMAT or metadata['base'] != snapshot_digest(base):
        raise ValueError('checkpoint does not belong to this acute snapshot')
    with np.load(path / 'runtime.npz', allow_pickle=False) as data:
        arrays = dict(data.items())
    return metadata, arrays


def configs(metadata):
    return {name: cls(**metadata[name]) if metadata[name] is not None else None for name, cls in (
        ('plasticity', PlasticityConfig), ('ion_feedback', IonFeedbackConfig), ('glial_remodeling', GlialRemodelingConfig))}


def restore(world, tissue, state, metadata, arrays):
    for name in TISSUE_ARRAYS:
        setattr(tissue, name, arrays[name])
    tissue.anatomy = replace(tissue.anatomy, **{name: arrays[name] for name in ANATOMY_ARRAYS})
    tissue._pending = list(zip(arrays['pending_times'].tolist(), arrays['pending_edges'].tolist()))
    tissue._revision = -1
    state['surface_coverage'][:] = arrays['surface_coverage']
    p, ion, glia = tissue.plasticity, tissue.ion_exchange, tissue.glial_remodeling
    if p is not None:
        p.contacts.clear()
        for group in (*p._outgoing, *p._incoming):
            group.clear()
        p._pairs.clear()
        p._axon_contacts[:] = 0
        p._changing.clear()
        p._weak.clear()
        p._mobile.clear()
        for i in range(len(arrays['contact_phase'])):
            values = {}
            for field in fields(SynapticContact):
                value = arrays['contact_' + field.name][i]
                values[field.name] = (None if field.name in CONTACT_OPTIONALS and np.isnan(value)
                                      else value.copy() if isinstance(value, np.ndarray) else value.item())
            c = SynapticContact(**values)
            edge = p._append(c)
            if c.phase in ('removed', 'lesioned'):
                p._retire(edge, c.phase)
        for order, groups, endpoint in (('plastic_active_order', p._outgoing, 'pre'),
                                          ('plastic_incoming_order', p._incoming, 'post')):
            for group in groups:
                group.clear()
            for edge in arrays[order].tolist():
                groups[getattr(p.contacts[edge], endpoint)][edge] = None
        p._last_post_s = arrays['plastic_last_post']
        p._activity = arrays['plastic_activity']
        p._inactive[:] = tissue._inactive_neurons
        p._mobile = dict.fromkeys(arrays['plastic_mobile'].tolist())
        p._weak = dict.fromkeys(arrays['plastic_weak'].tolist())
        p._changing = dict.fromkeys(arrays['plastic_changing'].tolist())
        p.next_update_s = metadata['plastic_next_s']
    if ion is not None:
        ion._withdrawal = arrays['ion_withdrawal']
        ion._recovering = set(arrays['ion_recovering'].tolist())
        ion.released_mol, ion.recovered_mol, ion._last_recovery_s = metadata['ion_state']
    if glia is not None:
        for name in GLIA_ARRAYS:
            setattr(glia, name, arrays['glia_' + name])
        glia.ticks, glia.migrations, glia.births, glia._start_s = metadata['glia_state']
    world._time_s = metadata['time_s']
    world._mesh_revision = metadata['mesh_revision']
