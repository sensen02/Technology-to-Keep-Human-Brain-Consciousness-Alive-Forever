"""Offline axis-correspondence feasibility: dataset condyle axes vs model joints."""
from pathlib import Path
import json,math,time,hashlib,zipfile,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/sensen/Desktop/engine修复验收/S2-bbnpyx-availability-20261014-v1')
TOL=15.0
# Declared switch: the protocol text says the dataset TiTar joint has only an incoming
# segment, while the implementation also uses Tar as its outgoing segment. The flag
# reproduces the protocol text literally and reports the counts under both readings.
LITERAL_TITAR='--literal-titar' in sys.argv
PAIRS={'L1':'LF','L2':'LM','L3':'LH','R1':'RF','R2':'RM','R3':'RH'}
GROUP={'LF':'L','LM':'L','LH':'L','RF':'R','RM':'R','RH':'R'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ang(u,v):
 c=float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-30));return math.degrees(math.acos(max(-1.,min(1.,c))))
def main():
 import mujoco
 from engine import canonical_physics as cp
 out=Path('/home/sensen/Desktop/engine修复验收')/('S2-bbnpyx-axis-mapping-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 r=dict(status='RUNNING',scope='OFFLINE_AXIS_CORRESPONDENCE_ONLY_NO_STANCE_BUILT',tolerance_deg=TOL,legs={},sourced={},notes=[])
 paths=[Path(__file__),ROOT/'S2_BBNPYX_AXIS_MAPPING_PROTOCOL.zh-CN.md',Path(cp.__file__),ROOT/'data/kinematic_leg_model/KinematicModel_InitalPosture_Coords3D-2.csv',DATA/'modelAngles_fly01_t001.csv',DATA/'provenance.json']
 hashes={str(p):sha(p) for p in paths}
 try:
  # --- dataset side: condyle axes and keypoint segments in the dataset body frame
  rows=[l.split(',') for l in paths[3].read_text().splitlines()[1:] if l.strip()]
  D={row[0]:np.array([float(x) for x in row[1:4]]) for row in rows}
  J=['ThCx','CxTr','TrFe','FeTi','TiTar']
  dataset={}
  for ds in PAIRS:
   axes={j:D[f'{ds}-{j}V2']-D[f'{ds}-{j}V1'] for j in J}
   pts={j:D[f'{ds}-{j}'] for j in J+['Tar']}
   chain=[pts[j] for j in J]+([] if LITERAL_TITAR else [pts['Tar']])
   seg=[chain[i+1]-chain[i] for i in range(len(chain)-1)]
   sig={}
   for i,j in enumerate(J):
    comps=[]
    if i>0:comps.append(ang(axes[j],seg[i-1]))
    if i<len(seg):comps.append(ang(axes[j],seg[i]))
    sig[j]=comps
   dataset[ds]=dict(axes={k:v.tolist() for k,v in axes.items()},signatures={k:[round(x,6) for x in v] for k,v in sig.items()},_sig=sig,_axes=axes)
  # --- model side: joint axes in world plus neighbouring anchors, one forward only
  m=cp.load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d);assert sha(cp.CANONICAL_XML)==cp.CANONICAL_XML_SHA256
  r['notes'].append('model compiled via canonical_physics.load_model (asset-resolving loader); joints/anchors identical to canonical XML; ctrlrange zero-floor does not affect geometry')
  def jid(name):
   i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);assert i>=0,name;return i
  model={}
  for model_leg,side in PAIRS.items():
   key=side
   names={'Coxa':[f'joint_{side}Coxa_{a}' for a in ('pitch','roll','yaw')],
          'Trochanter':[f'joint_{side}Trochanter_{a}' for a in ('pitch','roll','yaw')],
          'Tibia':[f'joint_{side}Tibia_pitch']}
   present={g:[n for n in grp if any(m.joint(j).name==n for j in range(m.njnt))] for g,grp in names.items()}
   aids={g:np.array([d.xanchor[jid(n)] for n in grp]) for g,grp in present.items() if grp}
   assert len(aids.get('Coxa',[]))==3 and len(aids.get('Trochanter',[]))==3 and len(aids.get('Tibia',[]))==1,(side,aids)
   seg_coxa=aids['Trochanter'][0]-aids['Coxa'][0]
   seg_troch=aids['Tibia'][0]-aids['Trochanter'][0]
   sig={}
   for a in ('pitch','roll','yaw'):
    j=jid(f'joint_{side}Coxa_{a}');axis=d.xmat[m.jnt_bodyid[j]].reshape(3,3)@m.jnt_axis[j];sig[f'Coxa-{a}']=[ang(axis,seg_coxa)]
   for a in ('pitch','roll','yaw'):
    j=jid(f'joint_{side}Trochanter_{a}');axis=d.xmat[m.jnt_bodyid[j]].reshape(3,3)@m.jnt_axis[j];sig[f'Trochanter-{a}']=[ang(axis,-seg_coxa),ang(axis,seg_troch)]
   j=jid(f'joint_{side}Tibia_pitch');axis=d.xmat[m.jnt_bodyid[j]].reshape(3,3)@m.jnt_axis[j];sig[f'Tibia-pitch']=[ang(axis,-seg_troch)]
   model[key]=sig
  # --- dataset mobile DOF in the recorded trial (fixed DOF carry no information per source readme)
  lines=(DATA/'modelAngles_fly01_t001.csv').read_text().splitlines();cols=lines[0].split(',')[1:]
  A=np.loadtxt(lines[1:],delimiter=',',usecols=range(1,len(cols)+1),ndmin=2)
  mobile={ds:{} for ds in PAIRS}
  for c in cols:
   ds,rest=c.split('-',1);pt,dof=rest.rsplit('-',1);mobile[ds][f'{pt}-{dof}']=float(np.ptp(A[:,cols.index(c)]))>0
  # --- matching
  for ds,side in PAIRS.items():
   leg=dict(model_side=side,signatures_model={k:[round(x,6) for x in v] for k,v in model[side].items()},signatures_dataset={k:v for k,v in dataset[ds]['signatures'].items()},matches={},unsourced=[])
   for mname,msig in model[side].items():
    cands=[]
    for jname,dsig in dataset[ds]['_sig'].items():
     if len(dsig)!=len(msig):continue
     for dof in ('yaw','pitch','roll'):
      key=f'{jname}-{dof}'
      if not mobile[ds].get(key,False):continue
      diff=[abs(a-b) for a,b in zip(msig,dsig)]
      if max(diff)<=TOL:cands.append(dict(dataset_dof=key,max_diff_deg=round(max(diff),3),diffs=[round(x,3) for x in diff]))
    cands.sort(key=lambda x:x['max_diff_deg'])
    if len(cands)==1:leg['matches'][mname]=cands[0]
    elif not cands:leg['unsourced'].append(dict(model=mname,reason='NO_MATCH_WITHIN_TOLERANCE'))
    else:leg['unsourced'].append(dict(model=mname,reason='AMBIGUOUS',candidates=cands))
   used=[v['dataset_dof'] for v in leg['matches'].values()]
   if len(used)!=len(set(used)):leg['one_to_one']=False;leg['unsourced'].append(dict(model='LEG',reason='NOT_ONE_TO_ONE',doubled=sorted({u for u in used if used.count(u)>1})))
   else:leg['one_to_one']=True
   leg['sourced_count']=len(leg['matches']);leg['verdict']='PASS_AXIS_CORRESPONDENCE' if leg['one_to_one'] and leg['sourced_count']>=2 else 'FAILED'
   r['legs'][ds]=leg;r['sourced'][ds]=leg['sourced_count']
  ds_mean={}
  for ds in PAIRS:
   vals=[v for k,v in dataset[ds]['signatures'].items() for v in v]
   ds_mean[ds]=round(float(np.mean(vals)),3)
  model_mean={side:round(float(np.mean([v for sig in model[side].values() for v in sig])),3) for side in set(PAIRS.values())}
  r['convention']=dict(literal_titar=LITERAL_TITAR,description='dataset TiTar signature uses only its incoming segment when literal_titar is true (protocol text), otherwise it also uses the Tar outgoing segment (implementation)')
  r['dataset_signature_component_mean_deg']=ds_mean
  r['model_signature_component_mean_deg']=model_mean
  lengths={ds:float(np.linalg.norm(D[f'{ds}-TiTar']-D[f'{ds}-ThCx'])) for ds in PAIRS}
  r['dataset_leg_chain_span']={k:round(v,4) for k,v in lengths.items()}
  r['summary']=dict(legs_pass=sum(1 for v in r['legs'].values() if v['verdict'].startswith('PASS')),sourced_per_leg=r['sourced'],mean_sourced=round(float(np.mean(list(r['sourced'].values()))),3))
  r['status']='PASS_MAPPING_FEASIBILITY' if r['summary']['legs_pass']==6 else 'FAIL_MAPPING_NOT_AVAILABLE_FALLBACK_TO_USER_DATA'
  np.savez_compressed(out/'signatures.npz',**{f'{ds}_{k}':np.array(v) for ds in PAIRS for k,v in dataset[ds]['signatures'].items()},**{f'model_{side}_{k}':np.array(v) for side,sig in model.items() for k,v in sig.items()})
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(9,4))
  labels=[];vals=[]
  for ds,leg in r['legs'].items():
   for mname,c in leg['matches'].items():labels.append(f'{ds}:{mname}->{c["dataset_dof"]}');vals.append(c['max_diff_deg'])
  ax.barh(range(len(vals)),vals,color='#3b7dd8');ax.axvline(TOL,color='r',linestyle='--',label=f'tolerance {TOL} deg')
  ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels,fontsize=6);ax.set_xlabel('max signature mismatch (deg)');ax.legend();ax.set_title('BBNPYX->model joint-axis signature mismatch (offline, no stance built)');fig.tight_layout();fig.savefig(out/'axis-mismatch.png');plt.close(fig)
 except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
 finally:
  r['hashes_before']=hashes;r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['hashes_before']==r['hashes_after']
  with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in paths[:3]:z.write(p,p.name)
  (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(out);print(r['status']);print(r.get('summary',''));print(r.get('error',''));return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
