"""Independent static geometry only: no force solve, LP, body or neural steps."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z/MODEL.xml')
INPUTS=[('previous_lp',Path('/home/sensen/Desktop/engine修复验收/s5-stance-lp-feasibility/20261101T205257Z/BEST_LP.npz'),'points'),('soft_best_k09',Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-wider-stance/20261101T205257Z/SEED_0.9_FINAL_STATE.npz'),'contact_rows')]
O=Path('/home/sensen/Desktop/engine修复验收/s5-stance-support-polygon/20261101T205953Z')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
def main():
    O.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),S]+[p for _,p,_ in INPUTS];hashes={str(p):sha(p) for p in files}
    dump(O/'PREREGISTRATION.json',dict(source_sha256=hashes,scope='S1/S2/S5 independent geometry for fixed poses, no force solve or LP',method='One compile, two kinematics/comPos; mass-weighted all-body inertial COM independently compared with Thorax subtreeCOM. Convex hull of saved force application points xy; signed edge distances, explicit three-point nonnegative barycentric certificate if inside.',limits='Support polygon containment concerns gravity-only whole-body wrench with point normal forces. Does not establish joint/muscle balance, compliant realization, geometry without penetration, stability, or neural recruitment. No >=4feet gate.'))
    assert sha(S)=='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
    m=_load_mjcf(str(S)).compile();rows=[]
    for name,p,key in INPUTS:
        z=np.load(p);q=z['qpos'];points=z[key] if key=='points' else z[key][:,3:6]
        d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        com=(m.body_mass[:,None]*d.xipos).sum(0)/m.body_mass.sum()
        np.testing.assert_allclose(com,d.subtree_com[m.body('Thorax').id],rtol=0,atol=1e-12)
        if 'com_world' in z.files:np.testing.assert_allclose(com,z['com_world'],rtol=0,atol=1e-12)
        xy=np.unique(points[:,:2],axis=0);h=ConvexHull(xy);vertices=xy[h.vertices]
        signed=h.equations[:,:2]@com[:2]+h.equations[:,2];inside=bool(signed.max()<=1e-12);certificate=None
        if inside:
            for j in range(1,len(vertices)-1):
                tri=vertices[[0,j,j+1]];weights=np.linalg.solve(np.vstack([tri.T,np.ones(3)]),np.r_[com[:2],1.])
                if weights.min()>=-1e-12:
                    certificate=dict(triangle=tri,normal_force_fractions=weights,reconstructed_com_xy=weights@tri,error=float(abs(weights@tri-com[:2]).max()));break
            assert certificate is not None
        rows.append(dict(name=name,input=str(p),mass_weighted_COM=com,support_points_xy=xy,hull_vertices=vertices,hull_equations=h.equations,COM_inside=inside,minimum_signed_inside_margin=float(-signed.max()),vertical_force_certificate=certificate))
    dump(O/'RESULT.json',dict(rows=rows,counts=dict(model_load=1,kinematics=2,comPos=2,forward=0,LP=0,body_step=0,neural_step=0),source_unchanged=hashes=={str(p):sha(p) for p in files}))
    print(json.dumps([{k:v for k,v in r.items() if k in ('name','COM_inside','minimum_signed_inside_margin')} for r in rows],indent=2))
if __name__=='__main__':main()
