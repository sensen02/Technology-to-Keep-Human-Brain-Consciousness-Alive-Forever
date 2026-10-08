"""Fast repaired-snapshot regression; no model construction or simulation.
Analytical live MuJoCo checks: venv_body/bin/python repair_body_viewer.py (dry-run).
"""
import hashlib, json
import numpy as np
from repair_body_viewer import ROOT, EPISODE, digest, load_payload
from export_integrated_viewer import samples, rounded

def main():
    p=load_payload((ROOT/'viewer/data.js').read_text());b=p['body']
    report=json.loads((ROOT/'viewer/export_report.json').read_text())['body_repair']
    assert report['applied'] and report['non_body_raw_bytes_preserved']
    assert hashlib.sha256(EPISODE.read_bytes()).hexdigest()==report['episode_sha256']
    for key,sha in report['unchanged_non_body_sha256'].items(): assert digest(p[key])==sha,key
    assert digest(b)==report['body_after_sha256']
    pos=np.asarray(b['frames']);q=np.asarray(b['rotations_wxyz'])
    assert np.array_equal(pos[:,0],np.zeros_like(pos[:,0]))
    assert np.array_equal(q[:,0],np.tile([1,0,0,0],(len(q),1)))
    assert b['nodes'][1]['name']=='nmf/c_thorax'
    assert len(b['nodes'])==69 and len(b['links'])==68
    assert all(a!=c and 0<=a<69 and 0<c<69 for a,c in b['links'])
    with np.load(EPISODE,allow_pickle=False) as ep:
        ix=samples(len(ep['truth/time_s']))
        assert np.array_equal(ep['truth/body_positions_mm'][:,1],ep['truth/body_positions_mm'][:,68])
        assert np.array_equal(ep['truth/body_rotations_wxyz'][:,1],ep['truth/body_rotations_wxyz'][:,68])
        assert np.array_equal(pos[:,1],rounded(ep['truth/body_positions_mm'][ix,0]))
        assert np.array_equal(q[:,1],rounded(ep['truth/body_rotations_wxyz'][ix,0],7))
        assert np.array_equal(pos[:,2:],rounded(ep['truth/body_positions_mm'][ix,2:]))
        assert np.array_equal(q[:,2:],rounded(ep['truth/body_rotations_wxyz'][ix,2:],7))
        assert np.array_equal(pos[:,1],rounded(ep['truth/thorax_mm'][ix]))
    head=next(x for x in b['meshes'] if x['name']=='nmf/c_head')
    assert head['body_index']==1
    assert sum(len(x['local_vertices_mm']) for x in b['meshes'])==54905
    print('PASS: canonical recorded mapping, fused-head exclusion, meshes/links, all non-body SHA256, unchanged NPZ truth alias evidence')

if __name__=='__main__':main()
