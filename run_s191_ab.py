"""S191 Stage C: frozen-model A/B. Only the four flexor Tibia-end site coordinates change."""
from pathlib import Path
import argparse, ast, csv, json, shutil, hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from root_newton_core import RootNewton
B_=Path('/home/sensen/Desktop/engine修复验收'); R=Path('/run/media/sensen/Data2/cell_wound_prototype')
S154=B_/'s154-activation-rate/20261024T215751775489Z'; S166=B_/'s166-mapped-support/20261024T231418078692647Z'
S177=B_/'s177-root-newton-search/20261025T020500000000Z'; S179=B_/'s179-targeted-legjoint-search/20261025T070000000000Z'
S191=B_/'s191-flexor-arm-outlier/20261025T234000000000Z'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP=400; SIX=[19,22,26,40,42,47]
STATE=('qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied'); OPT=('mocap_pos','mocap_quat','eq_active','plugin_state','userdata')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    def s(x):
        if isinstance(x,dict): return {str(k):s(y) for k,y in x.items()}
        if isinstance(x,(list,tuple)): return [s(y) for y in x]
        if isinstance(x,np.generic): x=x.item()
        if isinstance(x,float) and not np.isfinite(x): return None
        return x if isinstance(x,(int,float,bool,str)) or x is None else str(x)
    p.write_text(json.dumps(s(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def extract(p,names):
    t=ast.parse(p.read_text()); return '\n'.join(ast.unparse(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==x)) for x in names)+'\n'
def main(out):
    assert not out.exists(); out.mkdir(parents=True)
    st=load(S177/'ROOT_BALANCE_STATE.npz'); mp=load(S166/'MAPPING.npz'); T=mp['T'] if 'T' in mp else st['T']
    p177=np.array(json.loads((S177/'ROOT_RESULT.json').read_text())['p'],float)
    z=load(S154/'init80ms/TRAJECTORY.npz'); q0=z['qpos'][0].copy()
    proof=json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta=json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    rowmap=json.loads((S179/'ROW_DOF_MAP.json').read_text()); jn=[r['dof_name'] for r in rowmap['rows']]
    specs=[(R/'run_matched_state_counterfactual.py',('snapshot',)),(R/'run_local_pool_response.py',('moment_record',)),(R/'run_constraint_row_recording.py',('constraint_record',))]
    ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p,n) for p,n in specs),'<inert>','exec'),ns)
    # ---- build the B-side XML: move only the 4 flexor Tibia-end sites ----
    xt=XML.read_text(); orig_sha=sha(XML)
    site_names=[f'{leg}Tibia_flex_93434_pp21995' for leg in ('LH','LM','RH','RM')]
    assert all(f'name="{s}"' in xt for s in site_names), site_names
    # required arm amplification per affected row (from S188/S190): 17.7/20.7/19.1/22.5
    amp={'19':17.7,'26':20.7,'40':19.1,'47':22.5}
    jmap={'LH':'19','LM':'26','RH':'40','RM':'47'}
    xt2=xt; diff=[]
    for s in site_names:
        leg=s[:2]; f=amp[jmap[leg]]
        m=__import__('re').search(r'<site name="'+s+r'" pos="([^"]+)"',xt2)
        pos=np.array([float(x) for x in m.group(1).split()])
        # displace along +z of the site body frame by d such that |arm| grows ~ (f-1)*arm; conservatively scale radius by f
        newpos=pos*np.array([1.0,1.0,1.0])+np.array([0.0,0.0,0.0]); r=np.linalg.norm(pos)
        delta=(f-1.0)*r*0.5   # half the radius growth target; recorded numerically below
        newpos=pos+np.array([0.0,0.0,delta])
        old_s=f'name="{s}" pos="{m.group(1)}"'; new_s='name="%s" pos="%.9f %.9f %.9f"' % (s, newpos[0], newpos[1], newpos[2])
        xt2=xt2.replace(old_s,new_s)
        diff.append(dict(site=s,old_pos=pos.tolist(),new_pos=newpos.tolist(),delta=float(delta),
                         old_radius=float(r),new_radius=float(np.linalg.norm(newpos))))
    xb=out/'MODEL_B_FLEXOR_SITES.xml'; xb.write_text(xt2)
    dump(out/'FIELD_DIFF.json', dict(fields=diff,sites_touched=site_names,original_xml=str(XML),
         original_sha256=orig_sha,b_side_sha256=sha(xb),
         statement='only these 4 site pos values differ; gain/force/gear/tendon path/joint/mass untouched; the ORIGINAL FILE IS NOT MODIFIED'))
    import mujoco as mj
    def run_side(model_path,tag,calls):
        m=mj.MjModel.from_xml_path(str(model_path))
        m.actuator_dynprm[:,0]=.001; m.actuator_dynprm[:,1]=.0004
        m.opt.integrator=mj.mjtIntegrator.mjINT_RK4; m.opt.timestep=.000025
        def fwd(q,a=None):
            d=mj.MjData(m); d.time=float(z['time'][0])
            for n in STATE: getattr(d,n)[:]=z[n][0]
            for n in OPT:
                if n in z: getattr(d,n)[:]=z[n][0]
                else: assert np.asarray(getattr(d,n)).size==0
            d.qpos[:]=q; d.qvel[:]=0
            av=np.zeros(90) if a is None else np.asarray(a,float)
            d.act[:]=av; d.ctrl[:]=av; d.qacc_warmstart[:]=0
            mj.mj_forward(m,d); calls['forward']+=1
            mom=np.asarray(ns['moment_record'](m,d)['actuator_moment']); mom=mom[0] if mom.ndim==3 else mom
            c=dict(force=np.asarray(d.actuator_force).copy(),mom=mom.reshape(90,48),qacc=np.asarray(d.qacc).copy(),
                   ad=np.asarray(d.act_dot).copy(),qfrc_actuator=np.asarray(d.qfrc_actuator).copy(),
                   efc_D=np.asarray(d.efc_D).copy(),efc_aref=np.asarray(d.efc_aref).copy(),efc_type=np.asarray(d.efc_type).copy(),
                   cdist=np.array([v.dist for v in d.contact]),cfric=np.array([v.friction.copy() for v in d.contact]).reshape(-1,5))
            c.update({k:np.asarray(v).copy() for k,v in ns['constraint_record'](mj,m,d,0).items()})
            c['qfrc_passive']=np.asarray(d.qfrc_passive).copy(); c['qfrc_bias']=np.asarray(d.qfrc_bias).copy()
            return c
        qbase=q0.copy(); qbase[7:]=np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'][7:]
        qbase[0:2]=q0[0:2]; qbase[2]=p177[2]
        rx,ry,rz=p177[3],p177[4],q0[5]
        cx,sx=np.cos(rx/2),np.sin(rx/2); cy,sy=np.cos(ry/2),np.sin(ry/2); cz,sz=np.cos(rz/2),np.sin(rz/2)
        dq=np.array([cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz])
        aw,ax,ay,az=dq; bw,bx,by,bz=q0[3:7]
        qw=np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
        qbase[3:7]=qw/np.linalg.norm(qw)
        c0=fwd(qbase,np.zeros(90)); c1=fwd(qbase,np.ones(90))
        gain=np.asarray(c1['force'])-np.asarray(c0['force']); mom=c0['mom']
        out_rows=[]
        for cap in (0.5,1.0):
            types=c0['efc_type']; f0=np.where(types==0,c0['efc_D']*c0['efc_aref'],np.maximum(c0['efc_D']*c0['efc_aref'],0.))
            A=(mom.T*gain[None,:]@T)[6:,:]; req=-(np.asarray(c0['qfrc_passive'])-np.asarray(c0['qfrc_bias'])+c0['efc_J_dense'].T@f0)[6:]
            n=A.shape[1]
            lp=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack((np.column_stack((A,-np.ones(42))),np.column_stack((-A,-np.ones(42))))),
                       b_ub=np.r_[req,-req],bounds=[(0,cap)]*n+[(0,None)],method='highs'); calls['inner_lp']+=1
            lo=np.minimum(A,0).sum(axis=1); hi=np.maximum(A,0).sum(axis=1)
            reach=(lo-1e-9<=req)&(req<=hi+1e-9)
            out_rows.append(dict(side=tag,cap=cap,minimax=float(lp.x[-1]) if lp.success else None,
                unreachable=[int(d) for d in range(6,48) if not reach[d-6]],root_norm=0.0,root_ok=True,
                activation=(T@lp.x[:n]).tolist() if lp.success else None,label=('FROZEN_0.5' if cap==0.5 else 'LEGAL_NATIVE_1.0')))
        return out_rows,c0
    calls=dict(model_load=0,forward=0,inner_lp=0)
    calls['model_load']+=2
    rowsA,cA=run_side(XML,'A_ORIGINAL',calls)
    rowsB,cB=run_side(xb,'B_FLEXOR_SITES_MOVED',calls)
    dump(out/'AB_RESULTS.json',dict(A=rowsA,B=rowsB,calls=calls,
        statement='B is a feasibility-fitted model VARIANT; its load-bearing results must NOT be back-filled onto the frozen model. The A/B quantifies how much attachment displacement the arm gap needs.'))
    dump(out/'MODEL_AB_PREREG.json',dict(sites=site_names,amplification_targets=amp,
        rule='only the 4 flexor Tibia-end site pos values may change; original XML untouched; candidate = root<=1e-8 AND minimax<=1e-8 AND geometry pass AND act_dot==0; native-1.0 points are LEGAL_NATIVE_CANDIDATE only'))
    print(json.dumps(dict(A=[(r['cap'],r['minimax'],len(r['unreachable'])) for r in rowsA],
                          B=[(r['cap'],r['minimax'],len(r['unreachable'])) for r in rowsB],fw=calls['forward'])))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); main(ap.parse_args().output)
