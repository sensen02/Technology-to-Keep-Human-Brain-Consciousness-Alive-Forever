"""Single fixed-rule real-bout event reference. No controller or physics imports."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kinematic_turn_events import RULES,extract_turns,body_forward
S=Path('/home/sensen/Desktop/engine修复验收/real-freewalking-3d/20261027T150000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def signature(r):return [(e['source_frame'],e['incoming'],e['outgoing'],e['ambiguity']) for e in r['events']]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 root=Path(__file__).parent;inputs=[S/'bout_0000__kp_data.npy',S/'info__kp_names.json',S/'REFERENCE.npz'];before={str(p):sha(p) for p in inputs}
 dump(o/'PREREGISTRATION.json',dict(rules=RULES,input_hashes=before,commands=[sys.executable+' -m unittest -v test_kinematic_turn_events',sys.executable+' '+str(Path(__file__))+' --output '+str(o)],no_parameter_search=True,scope='whole fixed bout_0000; kinematic turning events only',neural_calls=0,physics_calls=0))
 run=subprocess.run([sys.executable,'-m','unittest','-v','test_kinematic_turn_events'],cwd=root,capture_output=True,text=True);(o/'TEST.log').write_text(run.stdout+run.stderr);assert run.returncode==0
 names=json.loads((S/'info__kp_names.json').read_text());raw=np.load(S/'bout_0000__kp_data.npy').reshape(-1,50,3).astype(float);frames=np.load(S/'REFERENCE.npz')['source_frame'];forward=body_forward(raw,names)
 legs=['T1L','T2L','T3L','T1R','T2R','T3R'];results={};all_events=[];checks=[]
 angle=.73;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
 transformed=raw@R.T*3.7+np.array([12.,-5.,8.]);changed=body_forward(transformed,names)
 np.testing.assert_allclose(changed,forward*3.7,atol=1e-12)
 fig,axs=plt.subplots(7,1,figsize=(16,16),sharex=True)
 for j,leg in enumerate(legs):
  r=extract_turns(forward[:,j],frames);assert signature(r)==signature(extract_turns(changed[:,j],frames));smooth=r.pop('smoothed')
  # Independent raw-array dot reconstruction and explicit sorted median directional checks.
  independently=[]
  for i in range(len(raw)):
   c=raw[i,names.index('Scutellum')];axis=raw[i,names.index('Antenna_Base')]-c
   independently.append(float(np.dot(raw[i,names.index(leg+'_TaTip')]-c,axis)/np.linalg.norm(axis)))
  np.testing.assert_allclose(independently,forward[:,j],atol=1e-12)
  half=RULES['window_frames']//2
  med=np.array([sorted(independently[max(0,i-half):min(len(raw),i+half+1)])[len(independently[max(0,i-half):min(len(raw),i+half+1)])//2] if len(independently[max(0,i-half):min(len(raw),i+half+1)])%2 else np.mean(sorted(independently[max(0,i-half):min(len(raw),i+half+1)])[len(independently[max(0,i-half):min(len(raw),i+half+1)])//2-1:len(independently[max(0,i-half):min(len(raw),i+half+1)])//2+1]) for i in range(len(raw))])
  np.testing.assert_allclose(med,smooth,atol=1e-12)
  tol=np.ptp(independently)*RULES['deadband_fraction_of_segment_range']
  for e in r['events']:
   start=e['turn_interval_frames'][0]-int(frames[0]);stop=e['turn_interval_frames'][1]-int(frames[0]);left=med[start]-med[start-1];right=med[stop+1]-med[stop]
   assert (left>tol and right<-tol) if e['incoming']=='FORWARD' else (left<-tol and right>tol)
   e['independent_direction_verified']=True;all_events.append(dict(leg=leg,**e))
  results[leg]=r;ax=axs[j];t=(frames-frames[0])/800;ax.plot(t,forward[:,j],alpha=.45,label='raw reconstructed source');ax.plot(t,smooth,label='fixed median9')
  for e in r['events']:
   i=e['index'];ax.scatter(t[i],forward[i,j],c='orange' if e['ambiguity'] else 'black',marker='x' if e['ambiguity'] else 'o')
   axs[6].scatter(t[i],j,c='orange' if e['ambiguity'] else 'black',marker='^' if e['outgoing']=='FORWARD' else 'v')
  ax.set_ylabel(leg+' / source units');ax.grid(alpha=.2);ax.legend(fontsize=7)
 all_events.sort(key=lambda e:(e['source_frame'],e['leg']))
 axs[6].set_yticks(range(6),legs);axs[6].set_xlabel('Nominal seconds from frame/800 (not acquisition timestamps)');axs[6].set_title('All candidates: orange ambiguous, black unflagged; same-frame ties not ordered')
 fig.suptitle('Real freely running fly: kinematic reversals only / no contact or gait labels');fig.tight_layout();fig.savefig(o/'KINEMATIC_EVENTS.png',dpi=130);plt.close(fig)
 Path('/home/sensen/Desktop/REAL_FLY_KINEMATIC_EVENTS.png').write_bytes((o/'KINEMATIC_EVENTS.png').read_bytes())
 dump(o/'EVENTS.json',dict(legs=results,all_events=all_events,classification='KINEMATIC_TURNS_NOT_CONTACT_OR_VALIDATED_STEPS',reliable_cycle_status='NOT_ESTABLISHED',same_frame_ties='simultaneous within source frame; leg name ordering only for serialization'))
 assert {str(p):sha(p) for p in inputs}==before
 dump(o/'VERIFICATION.json',dict(engineering_passed=True,test_count=5,test_exit_code=0,extraction_exit_code=0,all_real_event_directions_independently_checked=True,positive_scale_and_rigid_transform_invariant=True,source_hashes_unchanged=True,limitations='invariance of this implementation does not establish upstream processing invariance; nominal800Hz only; no physical distances/contact'))
 for name in ('kinematic_turn_events.py','test_kinematic_turn_events.py',Path(__file__).name):(o/name).write_bytes((root/name).read_bytes())
 counts={leg:dict(events=len(r['events']),ambiguous=sum(bool(e['ambiguity']) for e in r['events']),candidate_cycles=len(r['candidate_cycles'])) for leg,r in results.items()}
 dump(o/'SUMMARY.json',counts);dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file()});print(json.dumps(dict(output=str(o),counts=counts)))
if __name__=='__main__':main()
