"""Dense2730 head from fixed ef768 LH-split; balanced qualified/corrective histories."""
from pathlib import Path
import importlib.util,sys,json
import torch,numpy as np
HERE=Path(__file__).resolve().parent
PARENT_DIR=HERE.parent/'left_hind_stance_refit';PARENT=PARENT_DIR/'TRAIN/checkpoint_004000.pt';PARENT_SHA='ef768c788c94aeecbb49f0bd1ccb5c424333af36ec514941b72e49ab0d4917ff'
s=importlib.util.spec_from_file_location('_dagger_fixed_parent_LH_io',PARENT_DIR/'model_io.py');legacy=importlib.util.module_from_spec(s);sys.modules[s.name]=legacy;s.loader.exec_module(legacy)
sha256=legacy.sha256;exact=legacy.exact;verify_hashes=legacy.verify_hashes;BASE=legacy.BASE;REFERENCE=legacy.REFERENCE;DATA=legacy.DATA
QUALIFIED=HERE.parent.parent/'attempt02/right_yaw_m060';ACTUAL=PARENT_DIR/'EVAL_right';CORRECTIVE=HERE/'ORACLE/CORRECTIVE_EXAMPLES.npz'
class DaggerHead(legacy.old.HeadRefit):
 VERSION='RIGHT_ONLY_yaw006_bound3_equal_qualified_CORRECTIVE_densehead_v1'
 def identity(self):
  i=super().identity();i.update(version=self.VERSION,kind='RIGHT_ONLY_equal_qualified_corrective_densehead',parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,head_initialization='full42matrix exactly assembled from ef768 LH-split',optimizer='freshAdam_head2730_lr.0003',loss_episodes=['qualified_right','actualNN_CORRECTIVE'],episode_weights=[.5,.5]);return i

def frozen(m):return {k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('head.')}
def initialize():
 parent,p=legacy.load_checked(PARENT,PARENT_SHA);m=DaggerHead(parent.base);m.encoder.load_state_dict(parent.encoder.state_dict(),strict=True);m.gru.load_state_dict(parent.gru.state_dict(),strict=True)
 with torch.no_grad():m.head.weight.copy_(parent.head.weight);m.head.bias.copy_(parent.head.bias)
 if not exact(frozen(m),frozen(parent)) or not torch.equal(m.head.weight,parent.head.weight) or not torch.equal(m.head.bias,parent.head.bias) or sum(q.numel() for q in m.parameters() if q.requires_grad)!=2730:raise ValueError('exact densehead assembly/frozen parent required')
 return m

def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def episodes():
 audit=json.loads((HERE/'ORACLE/VERIFY.json').read_text());oracle=json.loads((HERE/'ORACLE/RESULT.json').read_text())
 if not audit['all_passed'] or not all(audit['checks'].values()) or oracle['mj_step_calls']!=0 or oracle['max_warm_action_error']>1e-10 or oracle['max_prefix_geometry_error']>1e-10:raise ValueError('independent oracle-prefix gate required before ANN fit')
 verify_hashes(audit['source_hashes']);verify_hashes(oracle['input_source_hashes']);verify_hashes(oracle['artifacts'])
 teacher=json.loads((QUALIFIED/'RESULT.json').read_text())
 if not teacher['original_all5_gates'] or not teacher['cycles']['all_legs_completed'] or teacher['native']['post_nonfoot_peak_native']!=0 or -teacher['actual_signed_yaw_deg']<30:raise ValueError('original qualifiedRIGHT teacher no longer eligible')
 q=arrays(DATA);c=arrays(CORRECTIVE);actual=arrays(ACTUAL/'neural_decisions.npz');_,parent=legacy.load_checked(PARENT,PARENT_SHA)
 result=[]
 for name,z in [('qualified_right',q),('actualNN_CORRECTIVE',c)]:
  x=torch.from_numpy(z['X238'])[:,None];u=torch.from_numpy(z['command2'])[:,None];y=torch.from_numpy(z['Y42'])[:,None]
  if x.shape!=(2880,1,238) or u.shape!=(2880,1,2) or y.shape!=(2880,1,42):raise ValueError('each complete2880 required')
  if name=='qualified_right':ref=parent['replay']['means'];bh=parent['replay']['base_states'];rh=parent['replay']['residual_states']
  else:
   if not np.array_equal(z['X238'],actual['X238']) or not np.array_equal(z['command2'],actual['command']):raise ValueError('actualNN corrective inputs must remain unchanged')
   ref=torch.from_numpy(actual['mean'])[:,None];bh=torch.from_numpy(actual['hidden_after'][:,:128])[:,None];rh=torch.from_numpy(actual['hidden_after'][:,128:])[:,None]
  result.append(dict(name=name,x=x,u=u,y=y,parent_means=ref,parent_base_states=bh,parent_residual_states=rh))
 return result

def prepare(m):
 eps=episodes();cache=[]
 for e in eps:
  pc=legacy.old.old.cached(m,e['x'],e['u'])
  with torch.no_grad():means,h=m.sequence(e['x'],e['u'])
  cache.append(dict(base=pc['means'],gate=pc['gate'],hidden=h[1].detach(),base_states=h[0],means=means))
 return dict(episodes=eps,hidden=torch.cat([c['hidden'] for c in cache],1),base=torch.cat([c['base'] for c in cache],1),gate=torch.cat([c['gate'] for c in cache],1),y=torch.cat([e['y'] for e in eps],1),means=torch.cat([c['means'] for c in cache],1),base_states=torch.cat([c['base_states'] for c in cache],1))
def cached_forward(m,c):return c['base']+c['gate']*m.RESIDUAL_BOUND*torch.tanh(m.head(c['hidden']))
def objective(pred,c):return (pred[960:].tanh()-c['y'][960:]).square().mean()
neutral=legacy.old.neutral

def load_checked(path,digest,require_final=True):
 if sha256(path)!=digest:raise ValueError('dagger head digest')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!='RIGHT_DAGGER_DENSE_HEAD_V1' or p['config']['physical_eval_only_step']!=4000 or p['config']['episode_weights']!=[.5,.5]:raise ValueError('new strict equal-twoepisode densehead format required')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes']);m=DaggerHead(legacy.old.old.legacy.load_baseline_actor())
 if p['identity']!=m.identity():raise ValueError('dagger architecture/domain identity mismatch')
 m.load_state_dict(p['state_dict'],strict=True);parent,_=legacy.load_checked(PARENT,PARENT_SHA)
 if not exact(frozen(m),frozen(parent)):raise ValueError('frozen base/encoder/GRU changed')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('nonfinite dagger model')
 step=p['curve'][-1]['step']
 if step not in (2000,4000) or (require_final and step!=4000):raise ValueError('only fixedfinal4000 for physics')
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.0003);opt.load_state_dict(p['optimizer'])
 if len(opt.state)!=2 or any(float(v['step'])!=step for v in opt.state.values()) or opt.param_groups[0]['lr']!=.0003 or sum(q.numel() for q in m.adapter_parameters())!=2730:raise ValueError('freshdensehead Adam2/.0003 counters mismatch')
 return m.eval(),p
