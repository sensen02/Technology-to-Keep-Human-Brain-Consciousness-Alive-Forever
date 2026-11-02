"""Bounded source phaseuniform33 static support diagnostic; never neural control.
prepare freezes source before science; capture once; solve gated once; verify pure arrays.
Parent review must precede seal. No defaults/XML/tracker changes.
"""
from pathlib import Path
import sys,json,hashlib,shutil,traceback,math
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
D=Path('/home/sensen/Desktop/engine修复验收')
O=D/'s2-reference-support-candidate/20261102T182109Z'
B=D/'s5-banc888-physical-handover/20261101T220018Z'
C=D/'s2-reference-muscle-operating-range/20261102T155412Z'
S=D/'s2-mapped-support-capacity/20261102T170741Z'
N=D/'s2-native-support-constraints/20261102T172633Z'
FLOOR=1e-4

def ar(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,v):
 (O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def err(a,b=0):return float(np.max(abs(np.asarray(a)-np.asarray(b)),initial=0))
def save(n,p):np.savez_compressed(O/n,**p)
def prepare():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'PREPARATION.json').exists()
 paths=[Path(__file__),R/'AGENTS.md',R/'GOAL_ROADMAP.zh-CN.md',R/'GOAL_TRACKER.zh-CN.md',R/'tools/complete_native_support_constraints.py',B/'RUN/MODEL.xml',B/'RUN/BODY.npz',B/'RUN/MAPPING.json',B/'IDENTITIES.npz',C/'AUDIT.npz',C/'SOURCE_HASHES_BEFORE.json',S/'GROUP_MASKS.npz',S/'AUTHORITATIVE_MAPPING.json',S/'POSE_1.npz',N/'NATIVE_1.npz',N/'REACTIONS_1.npz',N/'CONTACTS_1.json',N/'RESULT.json',N/'FARKAS_1.npz',N/'COMPILED_MODEL.npz',S/'HISTORY_DUPLICATE_CHECK.json']+[N/f'LP_1_{arm}_{mode}.npz' for arm in ['independent90','mapped66'] for mode in ['native','outerbox']]
 hashes={str(p):sha(p) for p in paths};dump('SOURCE_HASHES_BEFORE.json',hashes)
 b=ar(B/'RUN/BODY.npz');ix=np.rint(np.linspace(2200,2600,33)).astype(int)
 frozen={k:v[ix] for k,v in b.items()};frozen['source_indices']=ix;frozen['phase_rad']=np.maximum(0,b['time'][ix]-.15)*10*np.pi%(2*np.pi)
 compiled=ar(N/'COMPILED_MODEL.npz');jids=np.flatnonzero(compiled['jnt_type']==3);qa=compiled['jnt_qposadr'][jids];bounds=compiled['jnt_range'][jids]
 frozen.update(joint_ids=jids,joint_qpos_addresses=qa,joint_bounds=bounds,joint_within_limits=(frozen['qpos'][:,qa]>=bounds[:,0])&(frozen['qpos'][:,qa]<=bounds[:,1]))
 save('SOURCE_FROZEN.npz',frozen)
 for old,new in [(S/'GROUP_MASKS.npz','GROUP_MASKS.npz'),(S/'AUTHORITATIVE_MAPPING.json','AUTHORITATIVE_MAPPING.json')]:shutil.copyfile(old,O/new)
 dump('HISTORY_DUPLICATE_CHECK.json',dict(scope='bounded named prior packages and tracker only',three_prior='same3470 sameq/static complete native + outer90 already solved at original0/1.3/slow1.25; no repeats',endpoint_reuse=dict(source_index=2600,native=str(N/'NATIVE_1.npz'),reactions=str(N/'REACTIONS_1.npz'),solver=str(N/'RESULT.json'),certificate=str(N/'FARKAS_1.npz')),other_history=js(S/'HISTORY_DUPLICATE_CHECK.json'),not_new='old081/083 pose search001/phasecapacity/IK differ q/parameters or constraints; not claimed new methods',other_candidates='32 distinct actual saved q of original engineering reference; no denser followup if none'))
 dump('PREREGISTRATION.json',dict(head='6d0a42f9986e98851ea1b153bb38e6c4d578f03e',stage='S2/S4',candidate_schema='phaseuniform33',source=str(B/'RUN/BODY.npz'),source_indices=ix,order='ascending source index',selection='firstsourceindex among native66 witnesses; only if none66, firstsourceindex native90 upper mechanical bound',source_scope='Actual simulated tracking of processed real-derived dynamic trajectory with engineering amplitudes/phase retained; NOT measured animal poses. Original reference source hashes retained. q unchanged including root/hinges; offline qvel0 is explicit stationary calibration, not continuous handover.',static=dict(compiles=1,new_forward=32,cache_endpoint=1,actualgain_and_bias='current q native mju functions; endpoint staticcache reused',constraints='reuse complete_native_support_constraints.py classification verbatim via frozen extracted source; all foot-floor/native internal types with actualineqbounds; no nonfoot-floor bearing or external root force',activation=[FLOOR,1],compiled_arrays_exact=469),LP=dict(first='independent90 outerbox feasibility once per new pose; cached endpointstatus2',then='only successful outer witness gate permits native90 and mapped66, each once; no extra optimization or Farkas LP',objective='zero exact feasibility',tolerance=dict(raw_eq=1e-8,friction=1e-8,bounds=1e-10),no_search='no random/IK/Fmax/geometry expansion or dense repeat'),body=dict(label='DIAGNOSTIC_NOT_NEURAL',gate='only selected valid native witness, prefer66 otherwise90 upper bound',duration_s=.05,dt_s=.0000125,sample_s=.0005,initial='once q=savedcandidate,qvel0,act=ctrl=witness; zero assist/warmstart; no continuous handover claim',policy='constantctrl, no feedback/onlineLP/PD/rootforces/qposewrites',strong_support=dict(no_nonfoot=True,meanfoot_over_weight=.8,minCOM_over_fixedH0=.8,H0=1.207252822479,boundary_nonfoot_excludes=True,any_prone_fails=True),native='each12.5us reaction/root/COM/firstevent; BODY each.5ms'),neural=0,network=0,training=0,plasticity=0,STDP='hypothesis deferred',parent_review_before_seal=True))
 # Exact original native classification block, no altered taxonomy or bounds.
 lines=(R/'tools/complete_native_support_constraints.py').read_text().splitlines();start=next(i for i,l in enumerate(lines) if l.lstrip().startswith('raw=d.efc_J.copy()'));end=next(i for i,l in enumerate(lines) if l.lstrip().startswith('feet=[c for c in contacts'))
 baseindent=len(lines[start])-len(lines[start].lstrip());selected=lines[start:end];ds=next(i for i,l in enumerate(selected) if l.lstrip().startswith('diffs='));de=next(i for i,l in enumerate(selected) if '# Every active row' in l);selected=selected[:ds]+selected[de:]
 block='\n'.join(selected).replace("np.savez_compressed(O/f'NATIVE_{pi}.npz',**state)","pass # caller saves augmented state").replace("dump(f'CONTACTS_{pi}.json',contacts)","pass # caller saves contacts")
 block='def classify(m,d,mj,cone,sparse):\n'+''.join(' '+line[baseindent:]+'\n' for line in block.splitlines())+' return state,contacts,nativefoot,internal,ibounds,limitinfo,excluded\n'
 (O/'NATIVE_CLASSIFIER.py').write_text(block)
 shutil.copyfile(__file__,O/'EXECUTION_SOURCE.py')
 dump('PREPARATION.json',dict(passed=True,source_frozen_sha=sha(O/'SOURCE_FROZEN.npz'),classifier_sha=sha(O/'NATIVE_CLASSIFIER.py'),calls=dict(model_compile=0,forward=0,LP=0)))

def capture():
 import mujoco as mj
 from flygym.compose.fly.musculoskeletal import _load_mjcf
 from tools.complete_native_support_constraints import moment
 assert not (O/'CAPTURE_RESULT.json').exists()
 calls=dict(model_compile=0,forward=0,muscleGain=0,muscleBias=0,contact_jacobian=0,body_step=0,neural=0,network=0,training=0,plasticity=0,LP=0)
 shutil.copyfile(__file__,O/'CAPTURE_EXECUTION_SOURCE.py')
 dump('CAPTURE_PREREG.json',dict(prereg_sha=sha(O/'PREREGISTRATION.json'),classifier_sha=sha(O/'NATIVE_CLASSIFIER.py'),frozen_sha=sha(O/'SOURCE_FROZEN.npz'),execution_sha=sha(Path(__file__))))
 try:
  scope=dict(np=np,moment=moment,err=err);exec((O/'NATIVE_CLASSIFIER.py').read_text(),scope)
  m=_load_mjcf(str(B/'RUN/MODEL.xml')).compile();calls['model_compile']+=1
  compiled={k:getattr(m,k).copy() for k in dir(m) if isinstance(getattr(m,k),np.ndarray)};old=ar(N/'COMPILED_MODEL.npz');assert len(compiled)==469 and set(old)==set(compiled) and all(np.array_equal(v,old[k]) for k,v in compiled.items());save('COMPILED_MODEL.npz',compiled)
  f=ar(O/'SOURCE_FROZEN.npz');rows=[]
  for pi,i in enumerate(f['source_indices']):
   if i==2600:
    for a,b in [('NATIVE_1.npz',f'NATIVE_{pi}.npz'),('REACTIONS_1.npz',f'REACTIONS_{pi}.npz'),('CONTACTS_1.json',f'CONTACTS_{pi}.json')]:shutil.copyfile(N/a,O/b)
    p=ar(S/'POSE_1.npz');save(f'MUSCLE_{pi}.npz',{k:p[k] for k in ['gain','bias','A90','base','J','mu']});st=ar(O/f'NATIVE_{pi}.npz');assert np.array_equal(st['qpos'],f['qpos'][pi]);rows.append(dict(pose=pi,source_index=int(i),cache=True,nefc=len(st['efc_type']),all_rows_accounted=True));continue
   d=mj.MjData(m);d.qpos[:]=f['qpos'][pi];d.qvel[:]=0;d.act[:]=f['activation'][pi];d.ctrl[:]=f['command'][pi];d.time=f['time'][pi]
   mj.mj_forward(m,d);calls['forward']+=1
   st,contacts,nf,internal,ib,limits,excluded=scope['classify'](m,d,mj,int(m.opt.cone),bool(mj.mj_isSparse(m)))
   assert np.array_equal(d.qpos,f['qpos'][pi]) and not d.qvel.any() and not d.qfrc_applied.any() and not d.xfrc_applied.any()
   gain=np.array([mj.mju_muscleGain(d.actuator_length[u],0,m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_gainprm[u,:9]) for u in range(90)]);calls['muscleGain']+=90
   bias=np.array([mj.mju_muscleBias(d.actuator_length[u],m.actuator_lengthrange[u],m.actuator_acc0[u],m.actuator_biasprm[u,:9]) for u in range(90)]);calls['muscleBias']+=90
   A=st['moment'].T*gain;base=d.qfrc_passive+st['moment'].T@bias-d.qfrc_bias
   feet=[c for c in contacts if c['foot_floor']];world=[]
   for c in feet:
    jp=np.zeros((3,48));jr=np.zeros_like(jp);plane=m.geom_type[c['geom1']]==mj.mjtGeom.mjGEOM_PLANE;g=c['geom2'] if plane else c['geom1'];mj.mj_jac(m,d,jp,jr,c['pos'],int(m.geom_bodyid[g]));calls['contact_jacobian']+=1;world.append(jp)
   save(f'NATIVE_{pi}.npz',st);dump(f'CONTACTS_{pi}.json',contacts);save(f'REACTIONS_{pi}.npz',dict(native_foot_columns=nf,internal_columns=internal,internal_bounds=ib));save(f'MUSCLE_{pi}.npz',dict(gain=gain,bias=bias,A90=A,base=base,J=np.array(world).reshape(-1,3,48),mu=np.array([c['friction'][0] for c in feet])))
   assert err(st['J'].T@st['efc_force'],st['qfrc_constraint'])<1e-8 and err(gain*d.act+bias,d.actuator_force)<1e-8
   rows.append(dict(pose=pi,source_index=int(i),cache=False,nefc=int(d.nefc),ncon=int(d.ncon),limits=limits,excluded_nonfoot_rows=excluded,all_rows_accounted=True,withinlimits=f['joint_within_limits'][pi]))
   dump('CAPTURE_PROGRESS.json',dict(calls=calls,rows=rows))
  dump('CAPTURE_RESULT.json',dict(passed=True,calls=calls,rows=rows,compiled_exact=469))
 except Exception:
  dump('CAPTURE_FAILURE.json',dict(calls=calls,error=traceback.format_exc()));raise

def matrix(pi,arm,mode):
 a=ar(O/f'MUSCLE_{pi}.npz');r=ar(O/f'REACTIONS_{pi}.npz');P=np.eye(90) if arm=='independent90' else ar(O/'GROUP_MASKS.npz')['G'];n=P.shape[1]
 contact=r['native_foot_columns'] if mode=='native' else a['J'].transpose(2,0,1).reshape(48,-1)
 E=np.column_stack([a['A90']@P,contact,r['internal_columns']]);b=-a['base']-a['A90']@np.full(90,FLOOR)
 cb=np.array([[0,np.inf]]*contact.shape[1]).reshape(-1,2) if mode=='native' else np.array([[-np.inf,np.inf],[-np.inf,np.inf],[0,np.inf]]*len(a['J'])).reshape(-1,2)
 bounds=np.concatenate([np.array([[0,1-FLOOR]]*n),cb,r['internal_bounds']]);U=np.zeros((0 if mode=='native' else 4*len(a['J']),E.shape[1]))
 if mode=='outerbox':
  for c,mu in enumerate(a['mu']):
   for k,(sx,sy) in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):U[4*c+k,n+3*c:n+3*c+3]=[sx,sy,-mu]
 return dict(E=E,b=b,U=U,v=np.zeros(len(U)),bounds=bounds,P=P,row_scale=np.maximum(1,np.maximum(abs(E).max(1),abs(b))),ineq_scale=np.maximum(1,abs(U).max(1)) if len(U) else np.ones(0),muscle_count=n,contact_count=contact.shape[1])

def solve():
 from scipy.optimize import linprog
 assert not (O/'RESULT.json').exists();rows=[];calls=0
 for pi in range(33):
  outer_ok=False
  for arm,mode in [('independent90','outerbox'),('independent90','native'),('mapped66','native')]:
   name=f'{pi}_{arm}_{mode}'
   if pi==32:
    cached=next(r for r in js(N/'RESULT.json')['rows'] if r['pose']==1 and r['arm']==arm and r['contact_model']==mode);rows.append(dict(**{**cached,'pose':pi},execution='cached',source=str(N/'RESULT.json')));continue
   if mode=='native' and not outer_ok:
    rows.append(dict(pose=pi,arm=arm,contact_model=mode,execution='skipped',reason='outer90 no valid witness',witness=False,status=None));continue
   p=matrix(pi,arm,mode);save(f'LP_{name}.npz',p);E,b,U,bd=p['E'],p['b'],p['U'],p['bounds'];sc=p['row_scale'];us=p['ineq_scale']
   z=linprog(np.zeros(E.shape[1]),A_eq=E/sc[:,None],b_eq=b/sc,A_ub=U/us[:,None] if len(U) else None,b_ub=np.zeros(len(U)) if len(U) else None,bounds=[(None if np.isneginf(lo) else lo,None if np.isposinf(hi) else hi) for lo,hi in bd],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});calls+=1
   row=dict(pose=pi,source_index=int(ar(O/'SOURCE_FROZEN.npz')['source_indices'][pi]),arm=arm,contact_model=mode,execution='executed',status=int(z.status),message=z.message,nit=int(z.nit),witness=False)
   if z.success:
    x=z.x;res=E@x-b;fr=U@x;a=FLOOR+p['P']@x[:int(p['muscle_count'])];ok=err(res)<=1e-8 and np.max(fr,initial=0)<=1e-8 and np.all(x>=bd[:,0]-1e-10) and np.all(x<=bd[:,1]+1e-10) and a.min()>=FLOOR-1e-10 and a.max()<=1+1e-10
    save(f'WITNESS_{name}.npz',dict(x=x,activation90=a,residual_raw=res,friction_residual=fr));row.update(witness=bool(ok),raw_eq=err(res),friction=float(np.max(fr,initial=0)))
   if mode=='outerbox':outer_ok=row['witness']
   rows.append(row);dump('SOLVE_PROGRESS.json',dict(calls=calls,rows=rows));print(json.dumps(row),flush=True)
 chosen=None
 for arm in ['mapped66','independent90']:
  found=[r for r in rows if r['arm']==arm and r['contact_model']=='native' and r['witness']]
  if found:chosen=min(found,key=lambda r:r['pose']);break
 dump('RESULT.json',dict(rows=rows,calls=dict(LP=calls),selected=chosen,body_gate='eligible' if chosen else 'skipped_no_native_witness',scope='Conditional static LP; native forward force differs from LP witness. No autonomous or neural result. Status2 retained; no new infeasibility certificate claimed.'))

def audit_matrices():
 # Arrays-only audit materializes even skipped cases; this does NOT execute LP.
 assert not (O/'MATRIX_AUDIT_BUILD.json').exists()
 for pi in range(33):
  for arm,mode in [('independent90','outerbox'),('independent90','native'),('mapped66','native')]:
   target=O/f'LP_{pi}_{arm}_{mode}.npz'
   p=matrix(pi,arm,mode)
   if target.exists():
    old=ar(target);assert set(old)==set(p) and all(np.array_equal(v,old[k]) for k,v in p.items())
   else:save(target.name,p)
 dump('MATRIX_AUDIT_BUILD.json',dict(matrices=99,new_LP_calls=0,skipped_cases='Matrices available for audit only; never solved. Endpoint matrices rebuilt arrays-only from cached state.',source_sha=sha(Path(__file__))))

def verify():
 # Fresh process; no mujoco/scipy or construction helper used.
 f=ar(O/'SOURCE_FROZEN.npz');b=ar(B/'RUN/BODY.npz');m=ar(O/'COMPILED_MODEL.npz');oldm=ar(N/'COMPILED_MODEL.npz');ix=np.rint(np.linspace(2200,2600,33)).astype(int)
 assert np.array_equal(ix,f['source_indices']) and len(set(ix))==33 and np.array_equal(ix[-1:], [2600])
 for k,v in b.items():assert np.array_equal(v[ix],f[k],equal_nan=True),k
 assert err(f['time'],ix*.0005)<1e-9 and err(f['phase_rad'],np.maximum(0,f['time']-.15)*10*np.pi%(2*np.pi))==0
 assert len(m)==469 and all(np.array_equal(v,oldm[k]) for k,v in m.items())
 qa=f['joint_qpos_addresses'];bd=f['joint_bounds'];assert np.array_equal(f['joint_within_limits'],(f['qpos'][:,qa]>=bd[:,0])&(f['qpos'][:,qa]<=bd[:,1]))
 mp=js(O/'AUTHORITATIVE_MAPPING.json');roots=ar(B/'IDENTITIES.npz')[mp['root_key']];groups={};G=np.zeros((90,66));pg=[]
 for r in mp['mapping']:
  key=tuple(r['local_indices']);assert list(roots[list(key)].astype(object))==r['tier_root_ids']
  if key and key not in groups:groups[key]=len(groups)
  g=groups[key] if key else -1;pg.append(g)
  if g>=0:G[r['actuator_index'],g]=1
 assert len(groups)==66 and pg.count(-1)==16 and np.array_equal(G,ar(O/'GROUP_MASKS.npz')['G'])
 errors={k:0. for k in ['A90','base','gain','bias','native_world_J','constraint_projection','E','b','bounds','U']};result=js(O/'RESULT.json');checked=[]
 for pi in range(33):
  st=ar(O/f'NATIVE_{pi}.npz');a=ar(O/f'MUSCLE_{pi}.npz');re=ar(O/f'REACTIONS_{pi}.npz');cs=js(O/f'CONTACTS_{pi}.json');J=st['J'];assert np.array_equal(st['qpos'],f['qpos'][pi]) and not st['qvel'].any() and not st['qfrc_applied'].any() and not st['xfrc_applied'].any()
  AA=np.array([[float(st['moment'][u,j])*float(a['gain'][u]) for u in range(90)] for j in range(48)])
  base=np.array([float(st['qfrc_passive'][j])-float(st['qfrc_bias'][j])+math.fsum(float(st['moment'][u,j])*float(a['bias'][u]) for u in range(90)) for j in range(48)])
  errors['A90']=max(errors['A90'],err(AA,a['A90']));errors['base']=max(errors['base'],err(base,a['base']));errors['constraint_projection']=max(errors['constraint_projection'],err(J.T@st['efc_force'],st['qfrc_constraint']))
  # Independent scalar stationary native force-length/bias formula.
  for u in range(90):
   p=m['actuator_gainprm'][u];lr=m['actuator_lengthrange'][u];r0,r1,F,scale,lo,hi,vm,fp,fv=map(float,p[:9]);assert F>0
   L=r0+(float(st['length'][u])-lr[0])/((lr[1]-lr[0])/(r1-r0));midlo=(lo+1)/2;midhi=(hi+1)/2
   if L<lo or L>hi:fl=0.
   elif L<=midlo:fl=.5*((L-lo)/(midlo-lo))**2
   elif L<=1:fl=1-.5*((1-L)/(1-midlo))**2
   elif L<=midhi:fl=1-.5*((L-1)/(midhi-1))**2
   else:fl=.5*((hi-L)/(hi-midhi))**2
   pb=m['actuator_biasprm'][u];assert np.array_equal(p[:9],pb[:9])
   bias=0. if L<=1 else -F*fp*.5*((L-1)/(midhi-1))**2 if L<=midhi else -F*fp*(.5+(L-midhi)/(midhi-1))
   errors['gain']=max(errors['gain'],abs(-F*fl-a['gain'][u]));errors['bias']=max(errors['bias'],abs(bias-a['bias'][u]))
  covered=set();footcols=[];internal=[];ib=[];excluded=[];feet=[c for c in cs if c['foot_floor']]
  for c in cs:
   rr=c['rows'];covered.update(rr)
   if c['floor'] and not c['foot_floor']:excluded+=rr;continue
   if not rr:continue
   assert c['dim']==3 and len(rr)==4
   if c['foot_floor']:footcols.extend(J[rr])
   else:internal.extend(J[rr]);ib.extend([[0,np.inf]]*len(rr))
  for row,t in enumerate(st['efc_type']):
   if row in covered:continue
   covered.add(row);t=int(t)
   bounds=[-np.inf,np.inf] if t==0 else [-float(st['efc_frictionloss'][row]),float(st['efc_frictionloss'][row])] if t in [1,2] else [0,np.inf] if t in [3,4] else None
   assert bounds is not None;internal.append(J[row]);ib.append(bounds)
   if t==3:
    ji=int(st['efc_id'][row]);da=int(m['jnt_dofadr'][ji]);q=st['qpos'][int(m['jnt_qposadr'][ji])];lo,hi=m['jnt_range'][ji];pred=np.zeros(48);pred[da]=1 if abs(q-lo)<=abs(q-hi) else -1;assert err(J[row],pred)<=1e-12
  assert covered==set(range(len(J))) and not excluded
  nf=np.array(footcols).reshape(-1,48).T;ic=np.array(internal).reshape(-1,48).T;ib=np.array(ib).reshape(-1,2)
  assert np.array_equal(nf,re['native_foot_columns']) and np.array_equal(ic,re['internal_columns']) and np.array_equal(ib,re['internal_bounds'])
  for c,jp in zip(feet,a['J']):
   sign=1 if m['geom_type'][c['geom1']]==0 else -1;local=sign*np.array(c['frame'])@jp;pred=np.array([local[0]+sg*c['friction'][t]*local[t+1] for t in range(2) for sg in [1,-1]])
   errors['native_world_J']=max(errors['native_world_J'],err(J[c['rows']],pred))
  for arm,mode in [('independent90','outerbox'),('independent90','native'),('mapped66','native')]:
   p=ar(O/f'LP_{pi}_{arm}_{mode}.npz');P=np.eye(90) if arm=='independent90' else G;n=P.shape[1];assert np.array_equal(P,p['P'])
   muscle=np.array([[math.fsum(float(AA[j,u])*float(P[u,k]) for u in range(90)) for k in range(n)] for j in range(48)])
   ct=nf if mode=='native' else a['J'].transpose(2,0,1).reshape(48,-1)
   E=np.column_stack([muscle,ct,ic]);rhs=np.array([-base[j]-FLOOR*math.fsum(float(AA[j,u]) for u in range(90)) for j in range(48)])
   cb=np.array([[0,np.inf]]*ct.shape[1]).reshape(-1,2) if mode=='native' else np.array([[-np.inf,np.inf],[-np.inf,np.inf],[0,np.inf]]*len(feet)).reshape(-1,2)
   bounds=np.concatenate([np.array([[0,1-FLOOR]]*n),cb,ib]);assert np.array_equal(bounds,p['bounds'])
   U=np.zeros_like(p['U'])
   if mode=='outerbox':
    for c,co in enumerate(feet):
     for k,(sx,sy) in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):U[4*c+k,n+3*c:n+3*c+3]=[sx,sy,-co['friction'][0]]
   errors['E']=max(errors['E'],err(E,p['E']));errors['b']=max(errors['b'],err(rhs,p['b']));errors['U']=max(errors['U'],err(U,p['U']));assert np.array_equal(p['row_scale'],np.maximum(1,np.maximum(abs(p['E']).max(1),abs(p['b']))))
   rr=next(r for r in result['rows'] if r['pose']==pi and r['arm']==arm and r['contact_model']==mode)
   if pi==32:assert rr['execution']=='cached' and rr['status']==2
   elif mode=='native':assert rr['execution']=='skipped' and rr['status'] is None and not rr['witness']
   else:assert rr['execution']=='executed' and rr['status']==2 and not rr['witness']
   assert not (O/f'WITNESS_{pi}_{arm}_{mode}.npz').exists()
   checked.append(dict(pose=pi,arm=arm,mode=mode,execution=rr['execution'],matrix_verified=True,witness='not_available',solve_repeated=False))
 assert max(errors.values())<=1e-8,errors
 assert result['selected'] is None and result['body_gate']=='skipped_no_native_witness' and result['calls']['LP']==32
 hashes=js(O/'SOURCE_HASHES_BEFORE.json');changed={p:dict(before=h,after=sha(p)) for p,h in hashes.items() if sha(p)!=h};assert set(changed)<= {str(Path(__file__))}
 dump('SOURCE_HASHES_AFTER.json',{p:sha(p) for p in hashes})
 dump('INDEPENDENT_ARRAY_VERIFICATION.json',dict(passed=True,errors=errors,source_indices_exact=True,phase_exact=True,source_fields_exact=True,compiled_exact=469,mapping_roots_exact=True,checked=checked,source_changes=changed,body='skipped_not_passed',witness='none; no residual invented',calls=dict(model_compile=0,forward=0,LP=0,body_step=0,neural=0)))
 print(json.dumps(errors))

if __name__=='__main__':{'prepare':prepare,'capture':capture,'solve':solve,'audit_matrices':audit_matrices,'verify':verify}[sys.argv[1]]()
