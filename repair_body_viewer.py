"""Bounded body-only snapshot repair. No neural run, renderer, or full export.
Run with venv_body/bin/python repair_body_viewer.py [--apply].
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, resource
import numpy as np
from export_integrated_viewer import ROOT, export_body, canonical_body_poses
from engine.embodied.body_backend import BodyBackend, BodyConfig

EPISODE=ROOT/'outputs/embodied_body/loop_episode_neural_modulated_seed0.npz'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def load_payload(text):
    prefix='window.VIEWER_DATA='
    if not text.startswith(prefix) or not text.endswith(';\n'):
        raise RuntimeError('unexpected snapshot format')
    return json.loads(text[len(prefix):-2])

def geometry_regression(be,body):
    """Compare exported mesh -> body pose against MuJoCo geom world pose.

    Resets to initial qpos then forwards kinematics only (no episode replay).
    Also tests the warmed initial backend state and an arbitrary rigid rotation.
    """
    import mujoco
    m,d=be.model,be.data
    maxima=[]
    for state in ('warmed_initial','reset_initial','rigid_rotation'):
        if state=='reset_initial':
            mujoco.mj_resetData(m,d);mujoco.mj_forward(m,d)
        elif state=='rigid_rotation':
            d.qpos[3:7]=[np.cos(.37),0,np.sin(.37),0];mujoco.mj_forward(m,d)
        p,q,_=canonical_body_poses(be,be.sim.get_body_positions(be.fly.name)[None],be.sim.get_body_rotations(be.fly.name)[None])
        assert np.allclose(p[0],d.xpos,atol=1e-12)
        assert np.allclose(q[0],d.xquat,atol=1e-12)
        worst=0.
        for mesh in body['meshes']:
            gi=m.geom(mesh['name']).id;mid=int(m.geom_dataid[gi]);bid=mesh['body_index']
            assert bid==int(m.geom_bodyid[gi])
            adr=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid])
            expected=m.mesh_vert[adr:adr+nv]@d.geom_xmat[gi].reshape(3,3).T+d.geom_xpos[gi]
            mat=np.empty(9);mujoco.mju_quat2Mat(mat,q[0,bid])
            actual=np.asarray(mesh['local_vertices_mm'])@mat.reshape(3,3).T+p[0,bid]
            worst=max(worst,float(np.max(np.linalg.norm(actual-expected,axis=1))))
        assert worst<9e-7,(state,worst)
        maxima.append({'state':state,'max_vertex_error_mm':worst})
    # Deliberately malformed/ambiguous recordings must fail closed.
    p=be.sim.get_body_positions(be.fly.name)[None].copy();q=be.sim.get_body_rotations(be.fly.name)[None].copy()
    rejected=0
    for pp,qq in ((p[:,:-1],q[:,:-1]),(np.roll(p,1,axis=1),np.roll(q,1,axis=1))):
        try: canonical_body_poses(be,pp,qq)
        except RuntimeError: rejected+=1
    assert rejected==2
    assert all(a!=b for a,b in body['links'])
    return {'world_mesh_comparisons':maxima,'vertices_per_state':sum(len(x['local_vertices_mm']) for x in body['meshes']),
            'ambiguous_inputs_rejected':rejected,'no_self_links':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
    path=ROOT/'viewer/data.js';original=path.read_text();payload=load_payload(original)
    protected={k:digest(v) for k,v in payload.items() if k!='body'}
    before=digest(payload['body'])
    be=BodyBackend(BodyConfig(),gl_backend=None)
    try:
        with np.load(EPISODE,allow_pickle=False) as ep:
            body,vertices=export_body(be,ep)
            # Existing snapshot must be the same recorded sample, not a different episode.
            if body['times_s']!=payload['body']['times_s']:
                raise RuntimeError('snapshot is not the expected recorded sample')
        tests=geometry_regression(be,body)
    finally: be.close()
    payload['body']=body
    # Replace ONLY the raw body JSON region. Preserve all other sections byte-for-byte.
    decoder=json.JSONDecoder();start=original.index('"body":')+len('"body":')
    _,end=decoder.raw_decode(original,start)
    text=original[:start]+json.dumps(body,separators=(',',':'),allow_nan=False)+original[end:]
    verified=load_payload(text)
    assert {k:digest(v) for k,v in verified.items() if k!='body'}==protected
    assert verified==payload
    report_path=ROOT/'viewer/export_report.json';report=json.loads(report_path.read_text())
    report.update(payload_bytes=len(text.encode()),body_vertices=vertices)
    report['body_repair']={'applied':args.apply,'source_episode':str(EPISODE),
        'episode_sha256':hashlib.sha256(EPISODE.read_bytes()).hexdigest(),
        'body_before_sha256':before,'body_after_sha256':digest(body),
        'unchanged_non_body_sha256':protected,'non_body_raw_bytes_preserved':True,
        'validation':tests,'row_mapping':body['row_mapping_audit'],
        'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'execution':'model construction + built-in 0.05 s warmup; forward kinematics only; no full export or neural simulation'}
    if args.apply:
        path.write_text(text)
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        assert {k:digest(v) for k,v in load_payload(path.read_text()).items() if k!='body'}==protected
    print(json.dumps(report['body_repair'],indent=2))

if __name__=='__main__':main()
