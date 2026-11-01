"""Offline pose coordinate search with an inner conservative contact-force LP.
Not a runtime controller; no physical or neural integration.
"""
from pathlib import Path
import os,json,hashlib,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import mujoco
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
SEED=Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-wider-stance/20261101T205257Z/SEED_0.9_FINAL_STATE.npz')
O=Path('/home/sensen/Desktop/engine修复验收/s5-stance-lp-feasibility/20261101T205257Z')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
def main():
    O.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),S/'MODEL.xml',S/'MAPPING.json',SEED]
    hashes={str(p):sha(p) for p in sources}
    dump(O/'PREREGISTRATION.json',dict(source_sha256=hashes,task='S2/S5 offline pose search, inner all48 force balance LP',pose_seed=str(SEED),budget='185 inner LPs: baseline plus two fixed coordinate sweeps of46pose variables, two signs per coordinate',pose_variables='42 hinges, rootz, relative body rotation vector3; rootxy fixed by previously tested horizontal invariance',steps='sweep1 hinges .02rad/z .002length/rot .01rad; sweep2 half these steps; accept strictly lower objective; bounds native hinges, seedz +/- .1, rotations +/- .15',contact='Nearest compiled mesh vertex to world floor for each of30tarsus meshes, only if signed vertex gap<=.001; projected onto plane; rigid-contact relaxation permits existing penetration and positive margin gaps. No forces from unsupported raised feet.',friction='Conservative diamond |fx|+|fy|<=mu*fz inside circular Coulomb cone; equal-priority geom pairs use max sliding friction, otherwise higher-priority geom; no torsional/rolling friction or joint-limit reaction variables',muscles='66 shared groups a=floor+.9999*G*x, x in[0,1],16unmapped floor; actual zero/unit force forwards at qvel0 give A=moment.T*deltaforce, passive+joint passive separately',LP='L1 slack48 minimization of normalized all48 (A*.9999*G)x+J.T*f = bias-jointpassive-musclebaseline; fn>=0. No contact constraint forces double counted.',selection='LP L1 residual + 100*sum(nonfoot clearance deficits below .02) +100*sum(distal body-side deficits below .02)',candidate='LP residual infinity norm/W<.001, actual circular friction verified, nonfoot clearance>=.02, distal body-side>=.02, then fresh actual softcontact forward and full48qacc<=.1, Fz/W .99..1.01, actdot/forces/bounds/geometry gates; static only',limitations='Rigid force LP cannot certify realized softcontact force distribution, contact complementarity, stability or neural availability. Local coordinate failure is not global infeasibility; native units NOT_ESTABLISHED. No integration or live use.'))
    assert hashes[str(S/'MODEL.xml')]=='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
    m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m)
    hinges=np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE);qa=m.jnt_qposadr[hinges]
    qseed=np.load(SEED)['qpos'].copy();rotseed=Rotation.from_quat(qseed[[4,5,6,3]])
    groups={}
    for row in json.loads((S/'MAPPING.json').read_text())['rows']:
        if row['local_indices']:groups.setdefault(tuple(row['local_indices']),[]).append(row['actuator_index'])
    G=np.zeros((90,66))
    assert len(groups)==66
    for k,ix in enumerate(groups.values()):G[ix,k]=1
    foot=[g for g in range(m.ngeom) if 'Tars' in (m.geom(g).name or '')];assert len(foot)==30
    floor=m.geom('floor').id;nonfoot=[g for g in range(m.ngeom) if g!=floor and g not in foot]
    sideb=[b for b in range(m.nbody) if any(t in (m.body(b).name or '') for t in ('Tibia','Tarsus'))]
    thorax=m.body('Thorax').id;W=float(m.body_mass.sum()*abs(m.opt.gravity[2]));floor_z=float(m.geom_pos[floor,2])
    assert np.allclose(m.geom_quat[floor],[1,0,0,0])
    lower=np.r_[m.jnt_range[hinges,0],qseed[2]-.1,[-.15]*3];upper=np.r_[m.jnt_range[hinges,1],qseed[2]+.1,[.15]*3]
    x=np.clip(np.r_[qseed[qa],qseed[2],np.zeros(3)],lower,upper)
    counts=dict(model_load=1,forward=0,LP=0,geomDistance=0,body_step=0,neural_step=0)
    best=None;log=[];t0=time.time()
    def forward(q,act):
        mujoco.mj_resetData(m,d);d.qpos[:]=q;d.qvel[:]=0;d.act[:]=act;d.ctrl[:]=act;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
        mujoco.mj_forward(m,d);counts['forward']+=1
    def evaluate(x):
        nonlocal best
        q=qseed.copy();q[qa]=x[:42];q[2]=x[42];quat=(rotseed*Rotation.from_rotvec(x[43:])).as_quat();q[3:7]=quat[[3,0,1,2]]
        forward(q,np.zeros(90));f0=d.actuator_force.copy();bias=d.qfrc_bias.copy();jp=d.qfrc_passive.copy()
        moment=np.zeros((90,48))
        for i in range(90):
            a=int(d.moment_rowadr[i]);n=int(d.moment_rownnz[i]);moment[i,d.moment_colind[a:a+n]]=d.actuator_moment[a:a+n]
        forward(q,np.ones(90));f1=d.actuator_force.copy();A=moment.T*(f1-f0)[None,:];base=moment.T@f0+A@np.full(90,.0001)
        error=0.
        if not log:
            probe=np.linspace(.01,.99,90);forward(q,probe);error=float(abs(base+A@(probe-.0001)-d.qfrc_actuator).max());assert error<1e-8
        gaps=[];points=[];geomids=[];mus=[];J=[]
        for g in foot:
            mesh=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
            verts=m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];point=verts[np.argmin(verts[:,2])].copy();gap=float(point[2]-floor_z)
            if gap>.001:continue
            point[2]=floor_z;j=np.zeros((3,48));jr=np.zeros_like(j);mujoco.mj_jac(m,d,j,jr,point,int(m.geom_bodyid[g]))
            priority=m.geom_priority[[floor,g]];fr=m.geom_friction[[floor,g],0]
            mu=float(max(fr) if priority[0]==priority[1] else fr[int(priority[1]>priority[0])])
            gaps.append(gap);points.append(point);geomids.append(g);mus.append(mu);J.append(j.T)
        nc=len(J);B=np.column_stack([.9999*A@G]+J)/W if J else .9999*A@G/W;y=(bias-jp-base)/W;nvar=B.shape[1]
        C=np.vstack([np.c_[B,-np.eye(48)],np.c_[-B,-np.eye(48)]]);rhs=np.r_[y,-y]
        if nc:
            friction=np.zeros((4*nc,nvar+48))
            for k,mu in enumerate(mus):
                for j,(sx,sy) in enumerate(((1,1),(1,-1),(-1,1),(-1,-1))):friction[4*k+j,66+3*k:69+3*k]=[sx,sy,-mu]
            C=np.vstack([C,friction]);rhs=np.r_[rhs,np.zeros(4*nc)]
        lb=np.r_[np.zeros(66),np.tile([-np.inf,-np.inf,0.],nc),np.zeros(48)]
        ub=np.r_[np.ones(66),np.full(3*nc+48,np.inf)]
        objective=np.r_[np.zeros(nvar),np.ones(48)]
        sol=linprog(objective,A_ub=C,b_ub=rhs,bounds=list(zip(lb,ub)),method='highs');counts['LP']+=1;assert sol.success,sol.message
        residual=B@sol.x[:nvar]-y
        dist=np.array([mujoco.mj_geomDistance(m,d,g,floor,1000,None) for g in nonfoot]);counts['geomDistance']+=len(nonfoot)
        side=np.array([(1 if m.body(b).name.startswith('L') else -1)*((d.xpos[b]-d.xpos[thorax])@d.xmat[thorax].reshape(3,3)[:,1]) for b in sideb])
        score=float(sol.fun+100*np.maximum(.02-dist,0).sum()+100*np.maximum(.02-side,0).sum())
        force=sol.x[66:nvar].reshape(nc,3)
        row=dict(index=len(log),score=score,lp_l1=float(sol.fun),lp_inf=float(abs(residual).max()),contacts=nc,clearance_min=float(dist.min()),side_min=float(side.min()),reconstruction_error=error)
        arrays=dict(pose_x=x.copy(),qpos=q,A=A,baseline=base,bias=bias,joint_passive=jp,B=B,y=y,C=C,rhs=rhs,objective=objective,solution=sol.x,lower=lb,upper=ub,residual=residual,activation=.0001+.9999*(G@sol.x[:66]),points=np.array(points).reshape(-1,3),contact_geoms=np.array(geomids),mu=np.array(mus),contact_force=force,gaps=np.array(gaps),nonfoot_distance=dist,side=side,group_matrix=G)
        log.append(row)
        if best is None or score<best[0]['score']:
            best=(row,arrays);np.savez_compressed(O/'BEST_LP.npz',**arrays);dump(O/'BEST.json',row)
        return score
    current=evaluate(x)
    for step in (1.,.5):
        for j in range(46):
            bestx=x.copy();bestscore=current
            for sign in (-1,1):
                candidate=x.copy();candidate[j]=np.clip(candidate[j]+sign*step*(.02 if j<42 else .002 if j==42 else .01),lower[j],upper[j]);score=evaluate(candidate)
                if score<bestscore:bestscore=score;bestx=candidate
            x=bestx;current=bestscore
    row,a=best;candidate=bool(row['lp_inf']<.001 and row['clearance_min']>=.02 and row['side_min']>=.02)
    soft=None
    if candidate:
        forward(a['qpos'],a['activation']);soft=dict(max_qacc=float(abs(d.qacc).max()),act_dot=float(abs(d.act_dot).max()),passed_acceleration=bool(abs(d.qacc).max()<=.1))
        np.savez_compressed(O/'SOFT_STATE.npz',qpos=d.qpos,qacc=d.qacc,activation=d.act,act_dot=d.act_dot)
    dump(O/'RESULT.json',dict(best=row,LP_candidate=candidate,soft_check=soft,history=log,counts=counts,wall_seconds=time.time()-t0,source_unchanged=all(sha(Path(p))==h for p,h in hashes.items()),interpretation='Local pose search only. LP permits force distributions not necessarily realized by compliant contact. No full candidate certification from a partial soft check.'))
    assert all(sha(Path(p))==h for p,h in hashes.items())
    (O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(best=row,LP_candidate=candidate,soft_check=soft,counts=counts),indent=2))
if __name__=='__main__':main()
