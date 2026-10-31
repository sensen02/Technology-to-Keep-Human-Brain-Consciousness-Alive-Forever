"""S121/P1/S2-S4: passive S115 reproduction with complete constraint rows.

Import is inert. main delegates the unmodified S115 runner in an isolated module.
No new dynamics calls, neural ticks or MuJoCo state/model writes are introduced.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / 'run_extended_initialization_probe.py'
S115 = Path('/home/sensen/Desktop/engine修复验收/s115-extended-initialization/20261024T062500Z')
MINIMUM = Path('/home/sensen/Desktop/engine修复验收/s120-constraint-identifiability/20261024T071500Z/MINIMUM_NEXT_RECORDING.json')
OUTPUT = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z')


def constraint_record(mj, m, d, sample_index):
    """Pure copies after original com_contact_record; never forward/step/write state."""
    n, nv, nc = int(d.nefc), int(m.nv), int(d.ncon)
    sparse = bool(mj.mj_isSparse(m))
    row = dict(time=np.asarray(float(d.time)), sample_index=np.asarray(sample_index),
               nv=np.asarray(nv), nefc=np.asarray(n), ncon=np.asarray(nc),
               is_sparse=np.asarray(sparse), nJ=np.asarray(int(d.nJ)),
               physical_step=np.asarray(sample_index),
               physical_tick=np.asarray(-1 if sample_index == 0 else (sample_index-1)//20),
               source_start_s=np.asarray(.08 if sample_index == 0 else .08+((sample_index-1)//20)*.0005),
               source_end_s=np.asarray(.08 if sample_index == 0 else .08+((sample_index-1)//20+1)*.0005))
    for name in ('efc_type', 'efc_id', 'efc_force'):
        value = np.asarray(getattr(d, name)).reshape(-1).copy()
        assert value.size == n, (name, value.size, n)
        row[name] = value
    raw = np.asarray(d.efc_J).reshape(-1).copy()
    assert raw.size == int(d.nJ)
    row['efc_J_raw'] = raw
    if sparse:
        for name in ('efc_J_rownnz', 'efc_J_rowadr', 'efc_J_rowsuper'):
            value = np.asarray(getattr(d, name)).reshape(-1).copy()
            assert value.size == n
            row[name] = value
        cols = np.asarray(d.efc_J_colind).reshape(-1).copy()
        assert cols.size == raw.size
        row['efc_J_colind'] = cols
        dense = np.zeros((n, nv), dtype=raw.dtype)
        active_indices = []
        for r, (adr, nnz) in enumerate(zip(row['efc_J_rowadr'], row['efc_J_rownnz'])):
            adr, nnz = int(adr), int(nnz)
            assert 0 <= adr <= raw.size and 0 <= nnz and adr+nnz <= raw.size
            idx = np.arange(adr, adr+nnz, dtype=np.int64)
            assert np.all((cols[idx] >= 0) & (cols[idx] < nv))
            np.add.at(dense[r], cols[idx], raw[idx])
            active_indices.append(idx)
        row['efc_J_active_indices'] = np.concatenate(active_indices) if n else np.empty(0, dtype=np.int64)
        row['efc_J_active_nnz'] = np.asarray(int(row['efc_J_rownnz'].sum()))
    else:
        assert raw.size == n*nv
        dense = raw.reshape(n, nv).copy()
        for name in ('efc_J_rownnz', 'efc_J_rowadr', 'efc_J_rowsuper', 'efc_J_colind', 'efc_J_active_indices'):
            row[name] = np.empty(0, dtype=np.int64)  # dense mode: CSR not applicable
        row['efc_J_active_nnz'] = np.asarray(np.count_nonzero(dense))
    row['efc_J_dense'] = dense
    contacts = list(d.contact)
    assert len(contacts) == nc
    for name in ('efc_address', 'dim', 'exclude', 'geom1', 'geom2'):
        row['contact_'+name] = np.asarray([int(getattr(c, name)) for c in contacts], dtype=np.int64)
    row['contact_geom'] = np.column_stack((row['contact_geom1'], row['contact_geom2']))
    row['contact_body'] = np.full((nc, 2), -1, dtype=np.int64)
    for i in range(nc):
        for side in range(2):
            geom = int(row['contact_geom'][i, side])
            if geom >= 0: row['contact_body'][i, side] = int(m.geom_bodyid[geom])
    row['contact_pos'] = np.asarray([np.asarray(c.pos).copy() for c in contacts], dtype=float).reshape(nc, 3)
    row['contact_frame'] = np.asarray([np.asarray(c.frame).copy() for c in contacts], dtype=float).reshape(nc, 3, 3)
    row['contact_valid_mask'] = np.ones(nc, dtype=bool)
    # Actual row IDs, not a guessed condim/cone expansion or slice by ncon.
    contact_types = [int(v) for k, v in mj.mjtConstraint.__members__.items() if k.startswith('mjCNSTR_CONTACT_')]
    row['row_contact_index'] = np.where(np.isin(row['efc_type'], contact_types), row['efc_id'], -1).astype(np.int64)
    row['contact_row_count'] = np.asarray([np.count_nonzero(row['row_contact_index'] == ci) for ci in range(nc)], dtype=np.int64)
    return row


def constraint_metadata(mj, m):
    names = lambda kind, n: [mj.mj_id2name(m, kind, i) for i in range(n)]
    return dict(schema_version=1, mujoco_version=mj.__version__,
        constraint_enum={k:int(v) for k,v in mj.mjtConstraint.__members__.items()},
        cone=int(m.opt.cone), cone_enum={k:int(v) for k,v in mj.mjtCone.__members__.items()},
        nv=int(m.nv), njnt=int(m.njnt), ntendon=int(m.ntendon), neq=int(m.neq), ngeom=int(m.ngeom),
        is_sparse=bool(mj.mj_isSparse(m)),
        joint_names=names(mj.mjtObj.mjOBJ_JOINT,m.njnt),
        tendon_names=names(mj.mjtObj.mjOBJ_TENDON,m.ntendon),
        equality_names=names(mj.mjtObj.mjOBJ_EQUALITY,m.neq),
        geom_names=names(mj.mjtObj.mjOBJ_GEOM,m.ngeom),
        body_names=names(mj.mjtObj.mjOBJ_BODY,m.nbody),
        geom_bodyid=m.geom_bodyid.tolist(), jnt_type=m.jnt_type.tolist(),
        jnt_qposadr=m.jnt_qposadr.tolist(), jnt_dofadr=m.jnt_dofadr.tolist(),
        jnt_limited=m.jnt_limited.tolist(), jnt_range=m.jnt_range.tolist(), dof_jntid=m.dof_jntid.tolist(),
        eq_type=m.eq_type.tolist(), eq_obj1id=m.eq_obj1id.tolist(), eq_obj2id=m.eq_obj2id.tolist(),
        tendon_limited=m.tendon_limited.tolist(), tendon_range=m.tendon_range.tolist(),
        id_namespaces={'mjCNSTR_EQUALITY':'equality','mjCNSTR_FRICTION_DOF':'dof (dof_jntid maps to joint)',
            'mjCNSTR_FRICTION_TENDON':'tendon','mjCNSTR_LIMIT_JOINT':'joint','mjCNSTR_LIMIT_TENDON':'tendon',
            'mjCNSTR_CONTACT_FRICTIONLESS':'contact index','mjCNSTR_CONTACT_PYRAMIDAL':'contact index',
            'mjCNSTR_CONTACT_ELLIPTIC':'contact index','unknown':'preserved raw type/id; never classified by residual'},
        schema='global time/nefc/ncon/nv/is_sparse/nJ/sample_index/physical_step/physical_tick/source_start_s/source_end_s; row_offsets+flattened efc_type/id/force; sample000_ through sample120_ every record field',
        sparse_storage='efc_J_raw and colind copy complete actual nJ arrays; rowadr/rownnz/rowsuper copy nefc; active_indices concatenate each row address range; active_nnz=sum(rownnz); dense independently expanded with np.add.at; no truncation',
        dense_storage='efc_J_raw.reshape(nefc,nv) ONLY when mj_isSparse false; CSR keys empty not applicable',
        contact_storage='all actual ncon contacts including excluded; row_contact_index from actual contact efc_type and efc_id, -1 for noncontact; contact_row_count counts actual rows',
        sampling='original sample expression calls original com_contact_record first then pure copies at same after-forward state; no new forward/step/neural/model/state writes',
        clocks='sample0 release before first physical command: physical_tick=-1 and source_start=end=.08; later ticks 0..5 identify held physical command, source interval .08+tick*.0005 to .08+(tick+1)*.0005',
        units='native; physical units NOT_ESTABLISHED', selection=False, deployment=False)


def pack_records(records):
    assert len(records) == 121
    packed = {name:np.asarray([r[name] for r in records]) for name in
        ('time','sample_index','physical_step','physical_tick','source_start_s','source_end_s','nv','nefc','ncon','is_sparse','nJ')}
    packed['row_offsets'] = np.r_[0, np.cumsum(packed['nefc'], dtype=np.int64)]
    for name in ('efc_type','efc_id','efc_force'):
        packed[name] = np.concatenate([r[name] for r in records])
    for i, row in enumerate(records):
        packed.update({f'sample{i:03d}_{name}':value for name,value in row.items()})
    return packed


def configure(original):
    """Install only serialization/capture hooks on the private runner module."""
    original.OUTPUT_PARENT = OUTPUT
    sha, save_json, save_npz = original.sha, original.atomic_json, original.atomic_npz
    baseline_inputs = json.loads((S115/'INPUT_HASHES.json').read_text())
    baseline_result = json.loads((S115/'init80ms/RESULT.json').read_text())
    for name, digest in baseline_inputs.items():
        if sha(Path(name)) != digest: raise ValueError('S115 input changed: '+name)
    for name, digest in baseline_result['hashes'].items():
        if sha(Path(name)) != digest: raise ValueError('S115 artifact changed: '+name)
    if sha(S115/'init80ms/TRAJECTORY.npz') != baseline_result['trajectory_sha256']:
        raise ValueError('S115 trajectory commitment changed')
    import mujoco as mj
    headers = Path(mj.__file__).resolve().parent/'include/mujoco'
    extras = [Path(__file__).resolve(), MINIMUM, *sorted(p for p in S115.rglob('*') if p.is_file()),
              headers/'mujoco.h', headers/'mjdata.h', headers/'mjtype.h', *map(Path,baseline_inputs)]
    commitments = {str(p):sha(p) for p in dict.fromkeys(extras)}
    records, metadata = [], {}
    original_capture = original.com_contact_record
    def capture(mj, m, d, floor_ids):
        result = original_capture(mj, m, d, floor_ids)
        records.append(constraint_record(mj, m, d, len(records)))
        if not metadata: metadata.update(constraint_metadata(mj, m))
        return result
    def json_hook(path, value):
        path = Path(path)
        if path.name == 'INPUT_HASHES.json': value = dict(value, **commitments)
        elif path.name == 'PREREG.json':
            value = dict(value, source_hashes=dict(value['source_hashes'], **commitments),
                selection_rule='NONE: passive S115 reproduction only; no W selection or tuning',
                passive_reproduction_only=True, retained_s115=str(S115), constraint_row_schema_version=1,
                constraint_recording='all121 after-original-com-capture states; read-only copies; no extra physics/neural calls',
                initialization_duration_intervention_ms=0,
                minimum_next_recording=str(MINIMUM), minimum_next_recording_sha256=sha(MINIMUM))
        elif path.name == 'COMPLETE.json':
            assert commitments == {name:sha(Path(name)) for name in commitments}
            value = dict(value, source_chain_hashes=dict(value['source_chain_hashes'], **commitments),
                         passive_reproduction_only=True, constraint_samples=len(records))
        return save_json(path,value)
    def npz_hook(path, **arrays):
        path = Path(path)
        if path.name == 'TRAJECTORY.npz':
            assert np.array_equal(arrays['time'],np.asarray([float(r['time']) for r in records]))
            save_npz(path.parent/'CONSTRAINT_ROWS.npz', **pack_records(records))
            save_json(path.parent/'CONSTRAINT_ROW_METADATA.json', metadata)
        return save_npz(path, **arrays)
    original.com_contact_record = capture
    original.atomic_json = json_hook
    original.atomic_npz = npz_hook
    return records, commitments


def main():
    spec = importlib.util.spec_from_file_location('_s121_original_s115', ORIGINAL)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    configure(original)
    original.main()


if __name__ == '__main__':
    main()
