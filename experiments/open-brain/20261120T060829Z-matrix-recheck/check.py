"""Independent material-frame slice and electrode endpoint checks of the supplied profile."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = ROOT/'experiments/open-brain'
SOURCE = Path('/run/media/sensen/Data1/omnisim')
sys.path[:0] = [str(SOURCE), str(BASE/'20261119T202650Z-neck-odor-integration')]
from neck_profile import configure
from omnisim.modules.array_damage import RadialInsertion
from omnisim.modules.banc_volume import SectionExposure
from omnisim.modules.banc_neck import load_fixed_neck
from omnisim.modules.electrode_observation import ElectrodeObservation, AxonProbe


def main(folder):
    matrix = folder/'matrix.json'
    data = json.loads(matrix.read_text())
    archive = Path('/run/media/sensen/Data1/omnisim-results/implant-site/banc-neck/synapse-amr/audit.json')
    assert matrix.read_bytes() == archive.read_bytes()
    before = json.loads((BASE/'20261120T053412Z-matrix-acceptance/matrix.json').read_text())
    axes = np.array(data['array']['axes_um'])
    assert np.array_equal(axes, before['array']['axes_um'])
    with np.load(data['segmentation']['cache']) as pack:
        vol, lower, roots = pack['labels'], pack['lower'], pack['roots'].tolist()
    ijk = np.column_stack(np.nonzero(vol))
    owner = vol[tuple(ijk.T)] - 1
    xyz = (ijk+lower+.5)*.2
    damage = np.zeros(len(xyz))
    in_metal = np.zeros(len(xyz),bool)
    model = RadialInsertion()
    # Direct evaluation at each tracked voxel; no adaptive tiles or path rasterizer.
    for start,end in axes:
        shift = end-start
        length = np.linalg.norm(shift)
        u = shift/length
        rel = xyz-start
        forward = rel @ u
        near = (forward>=0) & (forward<=length)
        dist2 = np.sum((rel-forward[:,None]*u)**2, axis=1)
        behind = (length-forward)*1e-6
        offset = np.clip(behind,0,model.radius_m)
        in_metal |= near & (dist2*1e-12 < offset*(2*model.radius_m-offset))
        energy = model.strain_energy(np.sqrt(dist2)*1e-6,behind)
        damage = np.maximum(damage, np.where(near,np.clip((energy-model.onset_Pa)/(model.failure_Pa-model.onset_Pa),0,1),0))
    paths = {p.root:p for p in load_fixed_neck(ROOT)[0]}
    ports = {r['root']:r for r in data['routes']}
    with np.load(BASE/'20261120T043500Z-neck-recording/recordings/11701_intact_reference.npz') as pack:
        ground = set(pack['roots'].tolist())
    rows = []
    for j,root in enumerate(roots):
        use = owner==j
        r = ports[root]
        count = np.bincount(ijk[use,1],minlength=vol.shape[1])
        dose = np.bincount(ijk[use,1],weights=damage[use],minlength=vol.shape[1])
        profile = SectionExposure(lower[1],count,dose)
        path = paths[root]
        computed = profile.effect(path.points_m)
        assert int(in_metal[use].sum())==r['overlap_voxels']
        np.testing.assert_allclose(computed.gain,r['gain'],atol=1e-10,rtol=0)
        # Standard electrical observation API, single source event, original threshold, no mixing.
        sensor = ElectrodeObservation(axes[:,1]*1e-6,[AxonProbe(root,path.points_m,speed_m_s=path.speed_m_s,owner=j)],
                                     section_exposure={root:profile})
        sensor.attach(None,'electrode')
        sensor.observe_spike(None,j,0.,1000)
        marks=[e for e in sensor.events if e['kind'] in ('recorded','missed')]
        assert len(marks)==64 and len({e['patch'] for e in marks})==64
        rows.append(dict(root=root,cell=r['cell_type'],direction=r['direction'],ground=root in ground,
                         metal_voxels=r['overlap_voxels'],fully_loaded_sections=int(((count>0)&(dose==count)).sum()),
                         unresolved_um=r['unresolved_arc_m']*1e6, gain=r['gain'],
                         above_threshold=sum(e['kind']=='recorded' for e in marks),
                         max_peak_uv=max(abs(e['peak_V']) for e in marks)*1e6))
    try:
        configure(SimpleNamespace(_tick=0),matrix,'shaft')
    except KeyError as exc:
        integration = repr(exc)
    else:
        raise AssertionError('reader behavior differs from inspected schema')
    summary=dict(archive_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),same_geometry=True,
                 healthy_receivers={k:sum(bool(r['arrival_ticks']) for r in rs.values()) for k,rs in data['frozen_runtime'].items()},
                 threshold_uv=sensor.detection*1e6, integration_exception=integration,
                 synapses={k:data['synaptic_damage'][k] for k in ('contacts','contacts_in_local_world','affected_contacts')},
                 sweep_counts=dict(Counter(e['kind'] for e in data['selected']['electrode']['sampling'][0]['events'])),
                 rows=rows)
    (folder/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    for group in (rows,[r for r in rows if r['ground']]):
        print(len(group),'overlap',sum(r['metal_voxels']>0 for r in group),'fullslice',sum(r['fully_loaded_sections']>0 for r in group),
              'unresolved',sum(r['unresolved_um']>1e-6 for r in group),'detect',sum(r['above_threshold']>0 for r in group),
              'max_uv',max(r['max_peak_uv'] for r in group))
    print(summary['healthy_receivers'],integration)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
