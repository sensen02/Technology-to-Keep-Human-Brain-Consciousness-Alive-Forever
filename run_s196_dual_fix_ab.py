"""S196: dual fix A/B - gradient-placed tibia-end sites (variant C) + slot-47 mapping (variant D).
Canonical XML never modified. No mj_step / neural / training."""
from pathlib import Path
import argparse, ast, json, shutil, hashlib, re, csv
import numpy as np
from scipy.optimize import linprog
B_=Path('/home/sensen/Desktop/engine修复验收'); R=Path('/run/media/sensen/Data2/cell_wound_prototype')
S154=B_/'s154-activation-rate/20261024T215751775489Z'; S166=B_/'s166-mapped-support/20261024T231418078692647Z'
S177=B_/'s177-root-newton-search/20261025T020500000000Z'; S179=B_/'s179-targeted-legjoint-search/20261025T070000000000Z'
S195=B_/'s195-flexor-path-attribution-and-b2/20261025T284000000000Z'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
BASE=3.308620867791373; SIX=[19,22,26,40,42,47]; SLOT47=47; TARGET_ARM=1.8e-2
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
    z=load(S154/'init80ms/TRAJECTORY.npz'); st=load(S177/'ROOT_BALANCE_STATE.npz'); qbase=st['qpos'].copy()
    mp=load(S166/'MAPPING.npz'); T=mp['T']; pool=mp['actuator_pool']
    meta=json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    rowmap=json.loads((S179/'ROW_DOF_MAP.json').read_text()); jn=[r['dof_name'] for r in rowmap['rows']]
    W=float(load(S154/'init80ms/NATIVE_MODEL_ARRAYS.npz')['body_mass'].sum()*9801.)
    specs=[(R/'run_matched_state_counterfactual.py',('snapshot',)),(R/'run_local_pool_response.py',('moment_record',)),(R/'run_constraint_row_recording.py',('constraint_record',))]
    ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p,n) for p,n in specs),'<inert>','exec'),ns)
    import mujoco as mj
    calls=dict(model_load=0,forward=0,inner_lp=0,kinematics=0)
    xt=XML.read_text(); sx=(R/'venv_body/lib/python3.12/site-packages/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml').read_text()
    srcsites={}
    for nm in ('bifemlh_r-P2','pp21995'):
        mm=re.search(r'<site name="LFTibia_flex_93434_'+re.escape(nm)+r'" pos="([^"]+)"',sx)
        srcsites[nm]=np.array([float(v) for v in mm.group(1).split()])
    # ---- variant C: B2 then gradient displacement of the tibia-end sites ----
    m=mj.MjModel.from_xml_path(str(XML)); d=mj.MjData(m); d.qpos[:]=qbase; mj.mj_kinematics(m,d); calls['model_load']+=1; calls['kinematics']+=1
    def arm(leg):
        nm=f'{leg}Tibia_flex_93434'; jt=m.joint(f'joint_{leg}Tibia_pitch')
        anchor=np.asarray(d.xanchor[jt.id]); axis=np.asarray(d.xaxis[jt.id])
        pt=np.asarray(d.site_xpos[m.site(nm+'_pp21995').id]); pa=np.asarray(d.site_xpos[m.site(nm+'_bifemlh_r-P2').id])
        r=pt-anchor; u=(pa-pt)/max(np.linalg.norm(pa-pt),1e-30)
        return float(-np.dot(np.cross(r,u),axis))
    # first put the source offsets in the model, then compute gradients there
    editsC={}
    for leg in ('LH','LM','RH','RM'):
        sign=1.0 if leg.startswith('L') else -1.0
        for nm in ('bifemlh_r-P2','pp21995'):
            sid=m.site(f'{leg}Tibia_flex_93434_{nm}').id
            p=srcsites[nm].copy(); p[1]*=sign; m.site_pos[sid]=p; editsC[f'{leg}Tibia_flex_93434_{nm}']=p.copy()
    mj.mj_kinematics(m,d); calls['kinematics']+=1
    disp=[]
    for leg in ('LH','LM','RH','RM'):
        base_arm=abs(arm(leg))
        best=None
        for nm in ('bifemlh_r-P2','pp21995'):
            sid=m.site(f'{leg}Tibia_flex_93434_{nm}').id; p0=np.array(m.site_pos[sid]).copy()
            g=np.zeros(3); eps=1e-6
            for k in range(3):
                for sgn in (+1,-1):
                    p=p0.copy(); p[k]+=sgn*eps; m.site_pos[sid]=p; mj.mj_kinematics(m,d); calls['kinematics']+=1
                    g[k]+=sgn*abs(arm(leg))/(2*eps); m.site_pos[sid]=p0
            mj.mj_kinematics(m,d)
            k=int(np.argmax(np.abs(g)))
            delta=(TARGET_ARM-base_arm)/g[k]
            if best is None or abs(delta)<abs(best['delta']): best=dict(site=nm,axis='xyz'[k],grad=float(g[k]),delta=float(delta),base_arm=base_arm)
        sid=m.site(f'{leg}Tibia_flex_93434_{best["site"]}').id
        p=np.array(m.site_pos[sid]).copy(); p['xyz'.index(best['axis'])]+=best['delta']; m.site_pos[sid]=p
        editsC[f'{leg}Tibia_flex_93434_{best["site"]}']=p.copy()
        mj.mj_kinematics(m,d); calls['kinematics']+=1
        disp.append(dict(leg=leg,site=best['site'],axis=best['axis'],delta=best['delta'],
                         frac_of_segment=float(abs(best['delta'])/0.545),arm_before=best['base_arm'],arm_after=abs(arm(leg))))
    with (out/'PLACEMENT_DISPLACEMENT.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(disp[0])); w.writeheader(); w.writerows(disp)
    # write variant C xml (B2 offsets + displacement)
    t=xt; fields=[]
    for leg in ('LH','LM','RH','RM'):
        sign=1.0 if leg.startswith('L') else -1.0
        for nm in ('bifemlh_r-P2','pp21995'):
            sn=f'{leg}Tibia_flex_93434_{nm}'
            mm=re.search(r'<site name="'+re.escape(sn)+r'" pos="([^"]+)"',t)
            new=editsC[sn]
            t=t.replace(f'<site name="{sn}" pos="{mm.group(1)}"','<site name="%s" pos="%.9f %.9f %.9f"'%(sn,new[0],new[1],new[2]))
            fields.append(dict(site=sn,old=[float(v) for v in mm.group(1).split()],new=list(map(float,new))))
    xc=out/'MODEL_C_GRADIENT_PLACEMENT.xml'; xc.write_text(t)
    dump(out/'FIELD_DIFF_C.json',dict(fields=fields,original_sha256=sha(XML),c_sha256=sha(xc),
        statement='variant C = B2 (tibia-end sites at FlyGym source offsets) + gradient minimal displacement to reach |arm|=1.8e-2; P1 untouched; no other fields changed'))
    # ---- evaluator with optional extra slot activation ----
    def evaluate(model_path,tag,box,extra_slot=False):
        mm_=mj.MjModel.from_xml_path(str(model_path)); calls['model_load']+=1
        mm_.actuator_dynprm[:,0]=.001; mm_.actuator_dynprm[:,1]=.0004
        mm_.opt.integrator=mj.mjtIntegrator.mjINT_RK4; mm_.opt.timestep=.000025
        def fwd(a=None):
            dv=mj.MjData(mm_); dv.time=float(z['time'][0])
            for n in STATE: getattr(dv,n)[:]=z[n][0]
            for n in OPT:
                if n in z: getattr(dv,n)[:]=z[n][0]
                else: assert np.asarray(getattr(dv,n)).size==0
            dv.qpos[:]=qbase; dv.qvel[:]=0
            av=np.zeros(90) if a is None else np.asarray(a,float)
            dv.act[:]=av; dv.ctrl[:]=av; dv.qacc_warmstart[:]=0
            mj.mj_forward(mm_,dv); calls['forward']+=1
            mom=np.asarray(ns['moment_record'](mm_,dv)['actuator_moment']); mom=mom[0] if mom.ndim==3 else mom
            c=dict(force=np.asarray(dv.actuator_force).copy(),mom=mom.reshape(90,48),qacc=np.asarray(dv.qacc).copy(),
                   ad=np.asarray(dv.act_dot).copy(),efc_D=np.asarray(dv.efc_D).copy(),efc_aref=np.asarray(dv.efc_aref).copy(),
                   efc_type=np.asarray(dv.efc_type).copy(),cdist=np.array([v.dist for v in dv.contact]))
            c.update({k:np.asarray(v).copy() for k,v in ns['constraint_record'](mm_,dv,0 and 0 or 0,).items()} if False else {})
            c.update({k:np.asarray(v).copy() for k,v in ns['constraint_record'](mj,mm_,dv,0).items()})
            c['qfrc_passive']=np.asarray(dv.qfrc_passive).copy(); c['qfrc_bias']=np.asarray(dv.qfrc_bias).copy()
            return c
        c0=fwd(np.zeros(90)); c1=fwd(np.ones(90))
        gain=np.asarray(c1['force'])-np.asarray(c0['force']); momT=np.asarray(c0['mom']).T
        types=c0['efc_type']; f0=np.where(types==0,c0['efc_D']*c0['efc_aref'],np.maximum(c0['efc_D']*c0['efc_aref'],0.))
        Tex = T.copy()
        if extra_slot:
            Tex = np.column_stack((T, 0.5*np.eye(90)[:,SLOT47]))
        A=(momT*gain[None,:]@Tex)[6:,:]
        req=-(momT@np.asarray(c0['force'])+np.asarray(c0['qfrc_passive'])-np.asarray(c0['qfrc_bias'])+c0['efc_J_dense'].T@f0)[6:]
        n=A.shape[1]
        lp=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack((np.column_stack((A,-np.ones(42))),np.column_stack((-A,-np.ones(42))))),
                   b_ub=np.r_[req,-req],bounds=[(0,box)]*n+[(0,None)],method='highs'); calls['inner_lp']+=1
        lo=np.minimum(A,0).sum(axis=1); hi=np.maximum(A,0).sum(axis=1)
        gap=np.maximum.reduce([lo-req,np.zeros(42),req-hi])
        rows=[]
        for i in range(42):
            mn=(float(req[i]/hi[i]) if (req[i]>0 and hi[i]>1e-12) else (float(req[i]/lo[i]) if (req[i]<0 and lo[i]<-1e-12) else None))
            rows.append(dict(version=tag,box=box,joint_row=i,dof=6+i,dof_name=jn[i],leg=jn[i][6:8],joint_name=jn[i],
                required=float(req[i]),Sigma_min=float(lo[i]),Sigma_max=float(hi[i]),gap=float(gap[i]),m_needed=mn))
        arms={}
        for leg in ('LF','RF','LH','RH','LM','RM'):
            try:
                jd=int(mm_.jnt_dofadr[mm_.joint(f'joint_{leg}Tibia_pitch').id])
                arms[leg]=float(c0['mom'][mm_.actuator(f'{leg}Tibia_flex_93434').id,jd])
            except Exception: pass
        return dict(version=tag,box=box,extra_slot=extra_slot,n_pools=A.shape[1],
            minimax=float(lp.x[-1]) if lp.success else None,
            lower_bound=float(max(gap)),unreachable=[int(x) for x in range(6,48) if not (lo[x-6]-1e-9<=req[x-6]<=hi[x-6]+1e-9)],
            rows=rows,flexor_arms=arms,
            sum_normal_over_weight=float(sum(float(f0[int(c0['contact_efc_address'][ci])] if int(c0['contact_dim'][ci])==1 else f0[int(c0['contact_efc_address'][ci]):int(c0['contact_efc_address'][ci])+4].sum()) for ci in range(len(c0['contact_geom'])) if int(c0['contact_efc_address'][ci])>=0)/W),
            max_abs_qacc=float(np.abs(c0['qacc']).max()))
    res=[]
    res.append(evaluate(XML,'A_frozen',1.0)); res.append(evaluate(XML,'A_frozen',2.0))
    res.append(evaluate(xc,'C_placement',1.0)); res.append(evaluate(xc,'C_placement',2.0))
    res.append(evaluate(xc,'D_placement+mapping47',1.0,extra_slot=True)); res.append(evaluate(xc,'D_placement+mapping47',2.0,extra_slot=True))
    dump(out/'AB_RESULTS_V3.json',dict(results=[{k:v for k,v in r.items() if k!='rows'} for r in res],
        statement='C/D are fitted model variants; their load-bearing numbers must not be back-filled onto A or history'))
    for r in res:
        with (out/f'ROWS_{r["version"]}_box{r["box"]}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(r['rows'][0])); w.writeheader(); w.writerows(r['rows'])
    dump(out/'SUMMARY.json',dict(baseline_gate=dict(measured=res[0]['minimax'],expected=BASE,passed=bool(abs(res[0]['minimax']-BASE)<=1e-9)),
        results=[{k:v for k,v in r.items() if k!='rows'} for r in res],displacements=disp,calls=calls,
        canonical_xml_sha256=sha(XML),no_step=True,no_neural=True,no_training=True,units='native; SI NOT_ESTABLISHED'))
    i42=42-6
    print(json.dumps(dict(gate=abs(res[0]['minimax']-BASE)<=1e-9,A=res[0]['minimax'],A2=res[1]['minimax'],
        C=res[2]['minimax'],C2=res[3]['minimax'],D=res[4]['minimax'],D2=res[5]['minimax'],
        dof42_gap_A=res[0]['rows'][i42]['gap'],dof42_gap_C=res[2]['rows'][i42]['gap'],dof42_gap_D=res[4]['rows'][i42]['gap'],
        lb_C=res[2]['lower_bound'],lb_D=res[4]['lower_bound'],arms=res[2]['flexor_arms'],disp=[(d['leg'],round(d['frac_of_segment'],3)) for d in disp])))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); main(ap.parse_args().output)
