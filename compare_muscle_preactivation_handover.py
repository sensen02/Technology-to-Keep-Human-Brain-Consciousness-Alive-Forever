"""Saved-only behavior contrast for explicit muscle-warmup engineering trial."""
from pathlib import Path
import json,numpy as np
P=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-preactivation/20261028T078000Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
def metrics(p):
 b=np.load(p/'BODY.npz');before=np.load(p/'HANDOVER_BEFORE.npz');h=float(before['time']);t=b['time'];neural=t>=h-1e-9;late=t>=t[-1]-.1;weight=json.loads((p/'GAINS.json').read_text())['weight'];f=np.flatnonzero(neural&(b['nonfoot_count']>0));r=json.loads((p/'NEURAL_RECORDS.json').read_text());warm=~neural
 return dict(directory=str(p),handover_s=h,body_states=len(t),prewarm_dx=float(b['qpos'][np.flatnonzero(neural)[0],0]-b['qpos'][0,0]),warmup_nonfoot_fraction=float(np.mean(b['nonfoot_count'][warm]>0)),handover_act_max=float(before['act'].max()),handover_act_norm=float(np.linalg.norm(before['act'])),handover_velocity_norm=float(np.linalg.norm(before['qvel'])),handover_assistance_norm=float(np.linalg.norm(before['qfrc_applied'])),first_nonfoot_relative_s=None if not len(f) else float(t[f[0]]-h),last100ms_nonfoot_fraction=float(np.mean(b['nonfoot_count'][late]>0)),last100ms_foot_load_ratio=float(np.mean(b['foot_force'][late,:,2].sum(1))/weight),final_com_height=float(b['com'][-1,2]),motor_events=sum(x['motor_spikes'] for x in r),neural_assistance_zero=bool(not b['assist'][neural].any()),external_force_zero=bool(not b['external'].any()),warning_zero=bool(not b['warning'].any()))
def main():
 result=dict(baseline=metrics(B),muscle_warmup=metrics(P),scope='Two engineering interventions together: fixed muscular preactivation during allowed warmup, and activation kinetics .005/.030. Not an isolated attribution or biological/autonomous walking claim. Compare after transient as well as initialfall.')
 assert result['muscle_warmup']['neural_assistance_zero'] and result['muscle_warmup']['external_force_zero'] and result['muscle_warmup']['warning_zero']
 o=P/'MUSCLE_WARMUP_COMPARISON.json';assert not o.exists();o.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
