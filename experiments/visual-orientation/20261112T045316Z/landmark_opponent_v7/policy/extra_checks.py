"""Actualfixedneuralreceptor/newconv1/exactother12/old720checks; no runtimecolorsteering."""
from data import *
def extra_checks(m,parent):
 rgb,manifest=library();d=arrays('train');meta=read(HERE/'DATASET.json')['splits']['train']['rows'];checks={}
 with np.load(OLD/'train.npz',allow_pickle=False) as z:
  for k in z.files:checks['old720_bitexact_'+k]=np.array_equal(d[k],z[k])
 previous=parent['state_dict'];current=m.state_dict();inherited=[k for k in previous if not k.startswith('conv1.')];checks['exactother12parenttensors']=len(inherited)==12 and all(torch.equal(current[k],previous[k]) for k in inherited);checks['freshconv1onechannelshape']=tuple(m.conv1.weight.shape)==(8,1,3,3) and tuple(m.conv1.bias.shape)==(8,)
 # RecreateONLYseedednewarchitecture for freshconv1 proof; no optimization happens here.
 architecture=module('_opponent_preflight_fresh_arch',HERE/'model.py')
 with torch.random.fork_rng(devices=[]):torch.manual_seed(20261112);fresh=architecture.VisualPolicy()
 checks['conv1freshseedexact']=torch.equal(m.conv1.weight,fresh.conv1.weight) and torch.equal(m.conv1.bias,fresh.conv1.bias)
 checks['fixedreceptorbuffers_exact']=torch.equal(m.opponent_weight,torch.tensor([1.,-.5,-.5],dtype=torch.float64).reshape(1,3,1,1)) and m.opponent_bias.item()==0 and not m.opponent_weight.requires_grad and not m.opponent_bias.requires_grad
 checks['fixedreceptor_not_optimizer_parameter']=len(list(m.parameters()))==14 and 'opponent_weight' not in dict(m.named_parameters()) and 'opponent_bias' not in dict(m.named_parameters())
 checks['newidentity_declaresengineeringopponent']=m.identity()['format']=='VISION_FIXED_NEURAL_OPPONENT_ARCH_V7' and m.identity()['shared_eye_CNN']==[1,8,8] and m.identity()['biological_retina_validity'] is False
 sample=rgb[[0,1,2,6]].astype(np.float64)/255.;expected=np.maximum(sample[...,0]-.5*sample[...,1]-.5*sample[...,2],0.);tx=torch.from_numpy(sample).permute(0,1,4,2,3).reshape(8,3,32,32);actual=torch.relu(torch.nn.functional.conv2d(tx,m.opponent_weight,m.opponent_bias)).reshape(4,2,32,32).detach().numpy();checks['neural1x1_numpy_pixeloperator_parity']=np.max(np.abs(actual-expected))<=1e-15
 expectedscale=np.array([4. if r['motif'] in ('empty','center','occluder_only') else 1. for r in meta]);checks['same240neutral4_480other1']=np.array_equal(expectedscale,d['episode_scale']) and np.sum(expectedscale==4)==240;checks['pointweights_stillsum1']=np.max(abs(d['weights'].sum(0)-1))<1e-12;checks['360pairs720once']=len(pair_table())==360 and sorted(i for p in pair_table() for i in p)==list(range(720));checks['mixedscale1_1_4_4']=d['episode_scale'][expand([0,160])].tolist()==[1.,1.,4.,4.]
 lookup={r['frame_id']:r for r in manifest['rows']}
 for name in ('validation','controls'):
  a=arrays(name);checks[name+'_only_newactualV12V13']={lookup[int(i)]['config_id'] for i in a['frame_ids'].ravel()}=={'V12','V13'} and np.all(a['episode_scale']==1)
 if not all(bool(v) for v in checks.values()):raise ValueError('opponentpreflight '+str({k:bool(v) for k,v in checks.items() if not bool(v)}))
 return {k:bool(v) for k,v in checks.items()}
