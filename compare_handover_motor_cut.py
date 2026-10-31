"""Saved causal readout-cut comparison; never simulates."""
from pathlib import Path
import numpy as np,json
B=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-causal-motorcut/20261028T066000Z')
def main():
 a=np.load(B/'BODY.npz');b=np.load(O/'BODY.npz');t=b['time'];switch=1.;pre=t<=switch+1e-9;post=t>=switch-1e-9
 for k in ['time','qpos','qvel','activation']:
  np.testing.assert_array_equal(a[k][pre],b[k][pre])
 ah=np.load(B/'HANDOVER_BEFORE.npz');bh=np.load(O/'HANDOVER_BEFORE.npz');assert set(ah.files)==set(bh.files)
 assert all(np.array_equal(ah[k],bh[k]) for k in ah.files)
 assert not b['command'][post].any() and not b['assist'][post].any() and not b['external'].any() and not b['warning'].any()
 r=json.loads((O/'NEURAL_RECORDS.json').read_text());assert all(not x['motor_output_connected'] and not np.array(x['applied_command']).any() for x in r);assert any(np.array(x['decoded_next_command']).any() for x in r);assert sum(x['motor_spikes'] for x in r)>0
 def onset(x):
  ids=np.flatnonzero(post&(x['nonfoot_count']>0));return None if not len(ids) else float(t[ids[0]]-switch)
 diverge=np.max(abs(a['qpos']-b['qpos']),axis=1);ix=np.flatnonzero(diverge>1e-10);force=np.max(abs(a['muscle_force']-b['muscle_force']),axis=1);fi=np.flatnonzero(force>1e-10)
 last=t>=t[-1]-.2;weight=json.loads((O/'GAINS.json').read_text())['weight'];results={}
 for label,x in [('connected061',a),('cut066',b)]:results[label]=dict(first_nonfoot_relative_s=onset(x),last200ms_nonfoot_fraction=float((x['nonfoot_count'][last]>0).mean()),last200ms_foot_load_ratio=float(x['foot_force'][last,:,2].sum(1).mean()/weight),final_com_height=float(x['com'][-1,2]),activation_max=float(x['activation'][post].max()),muscle_force_abs_max=float(abs(x['muscle_force'][post]).max()))
 out=dict(passed=True,warmup_and_entire_handover_before_exact=True,cut_applied_commands_zero=True,decoded_proposals_nonzero=True,motor_spikes_cut=sum(x['motor_spikes'] for x in r),motor_spikes_connected=sum(x['motor_spikes'] for x in json.loads((B/'NEURAL_RECORDS.json').read_text())),first_qpos_divergence_relative_s=None if not len(ix) else float(t[ix[0]]-switch),first_muscle_force_divergence_relative_s=None if not len(fi) else float(t[fi[0]]-switch),max_qpos_difference=float(diverge.max()),results=results,scope='Causal motor-readout cut, not neural silencing. Similar collapse onset does not imply neurons or output have no physical effect. Native activation floor preserved; zero command does not mean zero activation/force.')
 target=O/'MOTOR_CUT_COMPARISON.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
