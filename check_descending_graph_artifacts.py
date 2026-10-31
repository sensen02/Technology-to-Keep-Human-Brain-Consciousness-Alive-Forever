"""Compare production descending graph artifacts to independent manual-cache fixture."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.sparse import load_npz
from engine.embodied.descending_command import command_input,DescendingCommand,DNP09_ROOTS
O=Path('/home/sensen/Desktop/engine修复验收/s5-descending-graph-verification/20261028T073000Z')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--we',required=True);ap.add_argument('--wi',required=True);ap.add_argument('--identities',required=True);ap.add_argument('--root-key',default='root_ids');ap.add_argument('--label',default='071');ap.add_argument('--amplitude',type=float,default=.3);args=ap.parse_args();paths={'We':Path(args.we),'Wi':Path(args.wi)};counts={}
 for name,path in paths.items():
  a=load_npz(path).tocsr();b=load_npz(O/f'EXPECTED_{name}.npz').tocsr();assert a.shape==b.shape==(12002,12002);assert (a!=b).nnz==0;counts[name]=dict(nnz=int(a.nnz),values_exact=True,artifact=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 roots=np.load(args.identities)[args.root_key];expected=np.load(O/'EXPECTED_TIER_IDENTITIES.npz')['root_ids'];np.testing.assert_array_equal(roots,expected);built=dict(n=len(roots),tier=dict(root_ids=roots));config=DescendingCommand(amplitude_nA=args.amplitude,duration_s=.05);tested=[]
 for t in [0.,.0005,.049,.0495,.05,.0505,.2495]:
  current,trace=command_input(built,config,t);value=args.amplitude if t<.05 else 0.;expected_current=np.zeros(12002);expected_current[-2:]=value;np.testing.assert_array_equal(current,expected_current);assert trace['root_ids']==list(DNP09_ROOTS) and trace['local_indices']==[12000,12001];tested.append(dict(relative_s=t,nonzero_cells=int(np.count_nonzero(current)),sum_nA=float(current.sum())))
 for amplitude in [0.,args.amplitude]:
  config=DescendingCommand(amplitude_nA=amplitude,duration_s=.05)
  for k in range(250):
   t=k*.001;current,_=command_input(built,config,t);np.testing.assert_array_equal(current[:-2],np.zeros(12000));assert np.all(current[-2:]==(amplitude if k<50 else 0))
 result=dict(passed=True,production_graphs=counts,identities_exact=True,pulse_only_two_source_identified_neurons=True,all_250_command_windows_checked=True,exact_pulse_withdrawal_s=.05,boundary_cases=tested,new_tier_builds=0,new_neural_steps=0,new_physics_steps=0,scope='Exact graph/identity and pure command-vector verification; no neural/physical behavior proof.')
 target=O/f'PRODUCTION_{args.label}_GRAPH_AND_COMMAND_CHECK.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
