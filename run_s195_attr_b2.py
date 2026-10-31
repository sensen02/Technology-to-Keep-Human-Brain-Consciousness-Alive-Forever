"""S195: baseline repro gate -> per-row attribution -> DOF42 mechanism -> B2 -> gradient minimal B3 -> A/B v2.
Never overwrites the canonical XML. No mj_step / neural / training."""
from pathlib import Path
import argparse, ast, json, shutil, hashlib, re, csv
import numpy as np
from scipy.optimize import linprog
B_=Path('/home/sensen/Desktop/engine修复验收'); R=Path('/run/media/sensen/Data2/cell_wound_prototype')
S154=B_/'s154-activation-rate/20261024T215751775489Z'; S166=B_/'s166-mapped-support/20261024T231418078692647Z'
S177=B_/'s177-root-newton-search/20261025T020500000000Z'; S179=B_/'s179-targeted-legjoint-search/20261025T070000000000Z'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
SRC=R/'venv_body/lib/python3.12/site-packages/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml'
BASE_LP=3.308620867791373; SIX=[19,22,26,40,42,47]
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
    z=load(S154/'init80ms/TRAJECTORY.npz'); st=load(S177/'ROOT_BALANCE_STATE.npz')
    qbase=st['qpos'].copy()                     # FIX: use the frozen qpos verbatim
    proof=json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta=json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    mp=load(S166/'MAPPING.npz'); T=mp['T']; pool=mp['actuator_pool']
    rowmap=json.loads((S179/'ROW_DOF_MAP.json').read_text()); jn=[r['dof_name'] for r in rowmap['rows']]
    W=float(load(S154/'init80ms/NATIVE_MODEL_ARRAYS.npz')['body_mass'].sum()*9801.)
    specs=[(R/'run_matched_state_counterfactual.py',('snapshot',)),(R/'run_local_pool_response.py',('moment_record',)),(R/'run_constraint_row_recording.py',('constraint_record',))]
    ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p,n) for p,n in specs),'<inert>','exec'),ns)
    import mujoco as mj
    calls=dict(model_load=0,forward=0,inner_lp=0,kinematics=0)
    def evaluate(model_path,tag,want):
        cap=want  # box upper bound in T-space: T already carries the 0.5 gain, so frozen=1.0 and native=2.0
        m=mj.MjModel.from_xml_path(str(model_path)); calls['model_load']+=1
        m.actuator_dynprm[:,0]=.001; m.actuator_dynprm[:,1]=.0004
        m.opt.integrator=mj.mjtIntegrator.mjINT_RK4; m.opt.timestep=.000025
        def fwd(a=None):
            d=mj.MjData(m); d.time=float(z['time'][0])
            for n in STATE: getattr(d,n)[:]=z[n][0]
            for n in OPT:
                if n in z: getattr(d,n)[:]=z[n][0]
                else: assert np.asarray(getattr(d,n)).size==0
            d.qpos[:]=qbase; d.qvel[:]=0
            av=np.zeros(90) if a is None else np.asarray(a,float)
            d.act[:]=av; d.ctrl[:]=av; d.qacc_warmstart[:]=0
            mj.mj_forward(m,d); calls['forward']+=1
            mom=np.asarray(ns['moment_record'](m,d)['actuator_moment']); mom=mom[0] if mom.ndim==3 else mom
            c=dict(force=np.asarray(d.actuator_force).copy(),mom=mom.reshape(90,48),qacc=np.asarray(d.qacc).copy(),
                   ad=np.asarray(d.act_dot).copy(),efc_D=np.asarray(d.efc_D).copy(),efc_aref=np.asarray(d.efc_aref).copy(),
                   efc_type=np.asarray(d.efc_type).copy(),cdist=np.array([v.dist for v in d.contact]),
                   cfric=np.array([v.friction.copy() for v in d.contact]).reshape(-1,5))
            c.update({k:np.asarray(v).copy() for k,v in ns['constraint_record'](mj,m,d,0).items()})
            c['qfrc_passive']=np.asarray(d.qfrc_passive).copy(); c['qfrc_bias']=np.asarray(d.qfrc_bias).copy()
            return c
        c0=fwd(np.zeros(90)); c1=fwd(np.ones(90))
        gain=np.asarray(c1['force'])-np.asarray(c0['force'])
        types=c0['efc_type']; f0=np.where(types==0,c0['efc_D']*c0['efc_aref'],np.maximum(c0['efc_D']*c0['efc_aref'],0.))
        momT=np.asarray(c0['mom']).T
        bias_force=momT@np.asarray(c0['force'])          # actuator bias force at act=0 (MUST be included)
        A=(momT*gain[None,:]@T)[6:,:]
        req=-(bias_force+np.asarray(c0['qfrc_passive'])-np.asarray(c0['qfrc_bias'])+c0['efc_J_dense'].T@f0)[6:]
        n=A.shape[1]
        lp=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack((np.column_stack((A,-np.ones(42))),np.column_stack((-A,-np.ones(42))))),
                   b_ub=np.r_[req,-req],bounds=[(0,cap)]*n+[(0,None)],method='highs'); calls['inner_lp']+=1
        lo=np.minimum(A,0).sum(axis=1); hi=np.maximum(A,0).sum(axis=1)
        gap=np.maximum.reduce([lo-req,np.zeros(42),req-hi])
        arms={}
        for leg in ('LF','RF','LH','RH','LM','RM'):
            try:
                jd=int(m.jnt_dofadr[m.joint(f'joint_{leg}Tibia_pitch').id])
                k=m.actuator(f'{leg}Tibia_flex_93434').id
                arms[leg]=float(c0['mom'][k,jd])
            except Exception: pass
        floors=set(meta['floor_geom_ids']); gl={int(k):v for k,v in meta['geom_map'].items()}
        dep=0.;nf=0.;bear=set();mar=[]
        for ci,(g1,g2) in enumerate(c0['contact_geom']):
            g1=int(g1);g2=int(g2);leg=gl.get(g2 if g1 in floors else g1) if ((g1 in floors)!=(g2 in floors)) else None
            pen=max(0.,-float(c0['cdist'][ci])); addr=int(c0['contact_efc_address'][ci]); dim=int(c0['contact_dim'][ci])
            if leg is None: nf=max(nf,pen)
            else: dep=max(dep,pen)
            if addr<0: continue
            e=f0[addr:addr+4] if dim!=1 else np.r_[f0[addr],0,0,0]
            normal=float(f0[addr] if dim==1 else e.sum())
            mar.append(normal)
            if leg is not None and normal>0 and c0['cdist'][ci]<=0: bear.add(leg)
        nsum=sum(mar)
        root_res=float(np.abs(-(momT@np.asarray(c0['force'])+np.asarray(c0['qfrc_passive'])-np.asarray(c0['qfrc_bias'])+c0['efc_J_dense'].T@f0)[:6]).max())
        rows=[]
        for i in range(42):
            d=6+i
            mn=(float(req[i]/hi[i]) if (req[i]>0 and hi[i]>1e-12) else (float(req[i]/lo[i]) if (req[i]<0 and lo[i]<-1e-12) else None))
            ft=('OK' if gap[i]<=1e-9 else ('AMPLITUDE' if (mn is not None and mn<=1e9) else 'CHANNEL'))
            rows.append(dict(version=tag,cap=cap,joint_row=i,dof=d,dof_name=jn[i],leg=jn[i][6:8],joint_name=jn[i],
                required=float(req[i]),Sigma_min=float(lo[i]),Sigma_max=float(hi[i]),gap=float(gap[i]),
                m_needed=mn,failure_type=ft))
        with (out/f'ROWS_{tag}_cap{cap}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        return dict(version=tag,cap=cap,minimax=float(lp.x[-1]) if lp.success else None,
            unreachable=[int(d) for d in range(6,48) if not (lo[d-6]-1e-9<=req[d-6]<=hi[d-6]+1e-9)],
            lower_bound=float(max(gap)),rows=rows,flexor_arms=arms,root_residual=root_res,
            sum_normal_over_weight=float(nsum/W),max_abs_qacc=float(np.abs(c0['qacc']).max()),
            geometry=dict(foot_depth=dep,nonfoot=nf,bearing=len(bear)),
            geom_ok=bool(dep<=0.03442910871489635 and nf<=1e-12 and len(bear)>=3))
    # ---------- build variants ----------
    xt=XML.read_text(); sx=SRC.read_text(); orig=sha(XML)
    srcsites={}
    for nm in ('bifemlh_r-P1','bifemlh_r-P2','pp21995'):
        mm=re.search(r'<site name="LFTibia_flex_93434_'+re.escape(nm)+r'" pos="([^"]+)"',sx)
        srcsites[nm]=np.array([float(v) for v in mm.group(1).split()])
    def build(variant, edits):
        t=xt; field=[]
        for sn,new in edits:
            mm=re.search(r'<site name="'+re.escape(sn)+r'" pos="([^"]+)"',t)
            old=[float(v) for v in mm.group(1).split()]
            t=t.replace(f'<site name="{sn}" pos="{mm.group(1)}"','<site name="%s" pos="%.9f %.9f %.9f"'%(sn,new[0],new[1],new[2]))
            field.append(dict(site=sn,old=old,new=list(map(float,new)),delta=float(np.linalg.norm(np.array(new)-np.array(old)))))
        p=out/f'MODEL_{variant}.xml'; p.write_text(t)
        return p,field
    editsB=[]; editsB2=[]
    for leg in ('LH','LM','RH','RM'):
        sign=1.0 if leg.startswith('L') else -1.0
        for nm in ('bifemlh_r-P1','bifemlh_r-P2','pp21995'):
            sn=f'{leg}Tibia_flex_93434_{nm}'; new=srcsites[nm].copy(); new[1]*=sign
            editsB.append((sn,new))
            if nm!='bifemlh_r-P1': editsB2.append((sn,new))
    xb,fieldB=build('B_SOURCE_OFFSETS',editsB)
    xb2,fieldB2=build('B2_TIBIA_END_ONLY',editsB2)
    dump(out/'FIELD_DIFF_B2.json',dict(fields=fieldB2,
        all_B_fields=fieldB,p1_sites=[f['site'] for f in fieldB if f['site'].endswith('bifemlh_r-P1')],
        note='P1 sits on the Thorax in the source model? (checked below); the FlyGym source has only LF, so B P1 values are not inheritable and B2 isolates the two tibia-end sites'))
    # ---------- Stage A.1 baseline gate ----------
    a05=evaluate(XML,'A_ORIGINAL',1.0); a10=evaluate(XML,'A_ORIGINAL',2.0)
    gate=dict(measured=a05['minimax'],expected=BASE_LP,abs_diff=float(abs(a05['minimax']-BASE_LP)),
              tol=1e-9,passed=bool(abs(a05['minimax']-BASE_LP)<=1e-9),root_residual=a05['root_residual'],
              qpos_source='ROOT_BALANCE_STATE.npz[qpos] used verbatim (S194 reconstructed it and got 3.3975)',
              cap10=dict(minimax=a10['minimax'],improvement=float((a05['minimax']-a10['minimax'])/a05['minimax']),box='x in [0,2] = native ctrl 1.0'),
              s189_reference='S188 reported m=2 gives 3.126 (5.5% better than 3.3086)')
    dump(out/'BASELINE_REPRO_GATE.json',gate)
    if not gate['passed']:
        dump(out/'SUMMARY.json',dict(gate=gate,stopped='BASELINE_GATE_FAILED'))
        print(json.dumps(dict(gate='FAIL',measured=a05['minimax'],diff=gate['abs_diff']))); return
    # ---------- Stage B: all variants ----------
    res=[a05,a10]
    b05=evaluate(xb,'B_SOURCE_OFFSETS',1.0); b10=evaluate(xb,'B_SOURCE_OFFSETS',2.0); res+=[b05,b10]
    b205=evaluate(xb2,'B2_TIBIA_END_ONLY',1.0); b210=evaluate(xb2,'B2_TIBIA_END_ONLY',2.0); res+=[b205,b210]
    dump(out/'AB_RESULTS_V2.json',dict(results=[{k:v for k,v in r.items() if k!='rows'} for r in res],
        lower_bounds=[dict(version=r['version'],cap=r['cap'],lower_bound=r['lower_bound'],minimax=r['minimax']) for r in res],
        statement='B/B2 are fitted model variants; their load-bearing numbers must not be back-filled onto A or history'))
    # ---------- Stage A.2 attribution ----------
    attr=[]
    for r_a,r_b in ((a05,b05),(a10,b10)):
        for i in range(42):
            ra=r_a['rows'][i]; rb=r_b['rows'][i]
            attr.append(dict(cap=r_a['cap'],joint_row=i,dof=ra['dof'],dof_name=ra['dof_name'],
                gap_A=ra['gap'],gap_B=rb['gap'],delta_gap=rb['gap']-ra['gap'],
                required_A=ra['required'],required_B=rb['required'],delta_required=rb['required']-ra['required'],
                hi_A=ra['Sigma_max'],hi_B=rb['Sigma_max'],lo_A=ra['Sigma_min'],lo_B=rb['Sigma_min']))
    with (out/'ROW_GAP_A_VS_B.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(attr[0])); w.writeheader(); w.writerows(attr)
    # ---------- Stage A.3 DOF42 mechanism ----------
    i42=42-6
    dof42=dict(A=dict(required=a05['rows'][i42]['required'],Sigma_min=a05['rows'][i42]['Sigma_min'],Sigma_max=a05['rows'][i42]['Sigma_max'],gap=a05['rows'][i42]['gap']),
               B=dict(required=b05['rows'][i42]['required'],Sigma_min=b05['rows'][i42]['Sigma_min'],Sigma_max=b05['rows'][i42]['Sigma_max'],gap=b05['rows'][i42]['gap']),
               B2=dict(required=b205['rows'][i42]['required'],Sigma_min=b205['rows'][i42]['Sigma_min'],Sigma_max=b205['rows'][i42]['Sigma_max'],gap=b205['rows'][i42]['gap']))
    dof42['mechanism']=('unchanged' if abs(dof42['A']['gap']-dof42['B']['gap'])<1e-12 and abs(dof42['A']['Sigma_max']-dof42['B']['Sigma_max'])<1e-12
        else 'changed: compare required vs Sigma and note the flexor acts on the NEGATIVE side')
    dump(out/'DOF42_MECHANISM.json',dof42)
    dump(out/'MINIMAX_LOWER_BOUND.json',dict(rows=[dict(version=r['version'],cap=r['cap'],lower_bound=r['lower_bound'],
        minimax=r['minimax'],dof42_row_gap=r['rows'][i42]['gap']) for r in res],
        note='lower_bound = max over rows of the single-row interval gap (coupling ignored)'))
    # ---------- Stage A.6 analytic gradient minimal displacement ----------
    m=mj.MjModel.from_xml_path(str(XML)); d=mj.MjData(m); d.qpos[:]=qbase; mj.mj_kinematics(m,d); calls['kinematics']+=1
    def arm_of(leg):
        nm=f'{leg}Tibia_flex_93434'
        jt=m.joint(f'joint_{leg}Tibia_pitch')
        anchor=np.asarray(d.xanchor[jt.id]); axis=np.asarray(d.xaxis[jt.id])
        pt=np.asarray(d.site_xpos[m.site(nm+'_pp21995').id]); pa=np.asarray(d.site_xpos[m.site(nm+'_bifemlh_r-P2').id])
        r=pt-anchor; u=(pa-pt)/max(np.linalg.norm(pa-pt),1e-30)
        return float(-np.dot(np.cross(r,u),axis))
    grad=[]
    TARGET=1.8e-2
    for leg in ('LH','LM','RH','RM'):
        for sname in (f'{leg}Tibia_flex_93434_bifemlh_r-P2',f'{leg}Tibia_flex_93434_pp21995'):
            sid=m.site(sname).id; base_pos=np.array(m.site_pos[sid]).copy()
            J=np.zeros(3); eps=1e-6
            for k in range(3):
                for sgn in (+1,-1):
                    p=base_pos.copy(); p[k]+=sgn*eps; m.site_pos[sid]=p
                    mj.mj_kinematics(m,d); calls['kinematics']+=1
                    v=arm_of(leg)
                    m.site_pos[sid]=base_pos
                    J[k]+=sgn*v/(2*eps)
            mj.mj_kinematics(m,d)
            for k in range(3):
                if abs(J[k])>1e-9:
                    delta=(TARGET-abs(arm_of(leg)))/J[k]
                    grad.append(dict(leg=leg,site=sname,axis='xyz'[k],d_arm_d_site=float(J[k]),delta_needed=float(delta),
                        delta_frac_of_segment=float(abs(delta)/0.545),within_10pct=bool(abs(delta)<=0.0545)))
    with (out/'GRADIENT_MINIMAL_DISPLACEMENT.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(grad[0])); w.writeheader(); w.writerows(grad)
    dump(out/'SUMMARY.json',dict(gate=gate,variants=[{k:v for k,v in r.items() if k!='rows'} for r in res],
        dof42=dof42,gradient_candidates=len(grad),
        gradient_best=min(grad,key=lambda g:abs(g['delta_frac_of_segment'])),
        calls=calls,units='native; SI NOT_ESTABLISHED',no_step=True,no_neural=True,no_training=True,xml_original_sha256=orig))
    print(json.dumps(dict(gate=gate['passed'],A05=a05['minimax'],A10=a10['minimax'],B05=b05['minimax'],B10=b10['minimax'],
        B205=b205['minimax'],B210=b210['minimax'],dof42_A05=a05['rows'][i42]['gap'],dof42_B05=b05['rows'][i42]['gap'],
        lower_B05=b05['lower_bound'],grad_best=min(grad,key=lambda g:abs(g['delta_frac_of_segment'])))))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); main(ap.parse_args().output)
