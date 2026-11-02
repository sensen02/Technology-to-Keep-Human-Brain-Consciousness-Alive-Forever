"""Independent contact-sum/impulse/COM-integral review. No MuJoCo or stepping."""
from pathlib import Path
import json,hashlib,math
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s2-handover-support-dynamics/20261102T071409Z'
O=ROOT/'s2-handover-momentum-parent-review/20261102T071409Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'RESULT.json').exists();files=[Path(__file__),S/'ARRAYS.npz',S/'RESULT.json',S/'MODEL_METADATA.json'];hashes={str(p):sha(p) for p in files}
 dump(O/('FLOAT_TOLERANCE_ADDENDUM.json' if (O/'PREREGISTRATION.json').exists() else 'PREREGISTRATION.json'),dict(scope='Post-run independent contactforce/impulse and scalarquadrature verification only, no model or dynamics',source_sha256=hashes,force='Only realarchivedfloorcontacts signedworldforce; no assumedfootforce or addedcontroller',quadratures='left/right/trapezoid unchanged,no tunedtimeoffset',limits='An actualmomentumresidual must remainvisible; not infermechanicalcause from a curvefit.'))
 with np.load(S/'ARRAYS.npz',allow_pickle=False) as archive:a={k:archive[k] for k in archive.files}
 result=json.loads((S/'RESULT.json').read_text());model=result['model'];mass=float(model['total_mass']);g=abs(float(model['gravity'][2]));W=model['weight'];assert abs(mass*g-W)<1e-12 and abs(mass-model['rootfree_subtree_mass'])<=1e-18
 names=model['geom_names'];floor={i for i,name in enumerate(names) if name=='floor'};assert len(floor)==1
 ground=np.zeros((len(a['native_time']),3));positive=np.zeros(len(ground));first=None;first_row=None
 for i in range(len(ground)):
  rows=a['contact_rows'][a['contact_offsets'][i]:a['contact_offsets'][i+1]];columns=[[],[],[]];nf=[]
  for row in rows:
   g1,g2=int(row[0]),int(row[1])
   if g1 not in floor and g2 not in floor:continue
   sign=1 if g1 in floor else -1
   for j in range(3):columns[j].append(sign*float(row[15+j]))
   if row[4]<0:
    fz=sign*float(row[17]);nf.append(max(fz,0.))
    if fz>0 and first is None and a['native_step'][i]>104000:first=(i,g2 if sign==1 else g1);first_row=row.tolist()
  ground[i]=[math.fsum(v) for v in columns];positive[i]=math.fsum(nf)
 assert abs(ground-a['ground_force']).max()<1e-10 and abs(positive-a['positive_nonfoot_contact_sum']).max()<1e-10
 event=result['events']['positive_contact'];assert first==(event['native_index'],event['geom_id']) and names[first[1]]==event['geom']
 impulse=[];maxinteg=0.
 for title,start in [('prewarm',0),('release',200)]:
  for mode in ['left','right','trapezoid']:
   ni=a[title+'_'+mode+'_native_index'];v=float(a['com_velocity'][start,2]);z=float(a['body_com'][start,2]);vel=[v];height=[z]
   for prev,curr in zip(ni[:-1],ni[1:]):
    dt=float(a['native_time'][curr]-a['native_time'][prev]);aa=ground[prev,2]/mass-g;bb=ground[curr,2]/mass-g;acc=aa if mode=='left' else bb if mode=='right' else (aa+bb)/2
    nxt=v+dt*acc;z+=dt*(v+nxt)/2;v=nxt;vel.append(v);height.append(z)
   error=max(float(abs(np.array(vel)-a[title+'_'+mode+'_velocity']).max()),float(abs(np.array(height)-a[title+'_'+mode+'_height']).max()));maxinteg=max(maxinteg,error);assert error<1e-10
  ni=a[title+'_right_native_index'];dt=np.diff(a['native_time'][ni]);Jground=math.fsum(float(ground[k,2])*float(width) for k,width in zip(ni[1:],dt));Jgravity=-W*math.fsum(map(float,dt));bodymask=(a['body_time']>=a['native_time'][ni[0]]-1e-10)&(a['body_time']<=a['native_time'][ni[-1]]+1e-10);measured=mass*(float(a['com_velocity'][bodymask][-1,2])-float(a['com_velocity'][start,2]));impulse.append(dict(phase=title,ground_impulse_right=Jground,gravity_impulse=Jgravity,net_impulse=Jground+Jgravity,measured_momentum_change=measured,residual_net_minus_measured=Jground+Jgravity-measured))
 # Release no-nonfoot segment fixed 0..18ms, not post-collision averaging.
 release=np.flatnonzero((a['native_step']>104000)&(a['native_step']<=105440)&(a['native_kind']==0));assert len(release)==1440 and not positive[release].any()
 J18=math.fsum(float(ground[i,2])*.0000125 for i in release);gravity18=-W*.018;v0=float(a['com_velocity'][200,2]);index18=int(np.argmin(abs(a['body_time']-1.318)));v18=float(a['com_velocity'][index18,2]);measured18=mass*(v18-v0)
 precollision=dict(window_s=.018,native_endpoints=1440,positive_nonfoot_endpoints=0,mean_foot_weight=float(a['native_foot_force'][release,:,2].sum(1).mean()/W),ground_impulse=J18,gravity_impulse=gravity18,measured_momentum_change=measured18,net_impulse_minus_measured=J18+gravity18-measured18,initial_vz=v0,final_vz=v18,COM_height_delta=float(a['body_com'][index18,2]-a['body_com'][200,2]))
 report=dict(passed=True,source_sha256=hashes,scalar_contact_sum_max_error=float(abs(ground-a['ground_force']).max()),scalar_quadrature_max_error=maxinteg,first_positive_nonfoot=dict(geom=event['geom'],body=event['body'],relative_ms=event['relative_ms'],force_on_fly=event['signed_force']),phase_impulses=impulse,precollision=precollision,calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0,optimizer=0),limits='Recordedpost-stateforwardforces may differ from forces used at implicitintegration. Impulse residual is explicit; no forceoffset fitted or frame times changed. These are trajectorymechanics checks,not evidence of an alternativehandover or walking.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
