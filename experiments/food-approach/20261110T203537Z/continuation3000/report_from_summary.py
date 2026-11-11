"""Summarize completed saved five-condition evidence without physics/optimization."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent

def main():
 path=HERE/'RECOVERY_FINAL3000_SUMMARY.json';s=json.loads(path.read_text())
 if s['status']!='COMPLETE' or set(s['conditions'])!={'neutral','left','right','speed085','speed090'}:raise ValueError('all five actual conditions must settle first')
 lines=['# Fixed3000 exact continuation: complete actual evaluation','',
 'Checkpoint: `'+s['checkpoint']+'`','',
 'SHA-256: `'+s['checkpoint_sha256']+'`','',
 'Training resumed exact parent1000 model/Adam/RNG/full curve, changing only lr0.001→0.0001 for2000 fixed additional updates. No old captured source/data changes, no bound or network change. Training203.682s; all four integrity checkpoints passed22 checks each.','',
 '| Condition | Engineering eligible | Saved actual integrity | Forward | Slip max | Yaw rad |','|---|---|---|---:|---:|---:|']
 for c in ('neutral','left','right','speed085','speed090'):
  r=s['conditions'][c];a=r.get('autonomous',{})
  lines.append('|'+c+'|'+str(r['eligible'])+'|'+str(r['saved_trajectory_verified'])+'|'+str(a.get('forward_signed_native_per_s'))+'|'+str(a.get('slip_ratio_max'))+'|'+str(r.get('yaw_change_rad'))+'|')
 lines+=['', 'All conditions eligible: **'+str(s['all_conditions_eligible'])+'**. Saved-trajectory integrity for all: **'+str(s['saved_trajectory_verification'])+'**.','',
 'Original all5 gates, all6 sensed cycles, strict nonfoot zero, turn sign/magnitude0.01 and speed reduction versus matched neutral remain unchanged. Integrity proves recorded checkpoint inference/state/force/source/trajectory consistency, not behavioral success. Full native D1–D8 certification was not run. No food behavior optimization, arrival, ingestion or satiety was performed.','',
 'Original pipeline219 interrupted before a neutral trace persisted; original directory/log/summary remain intact. No completed condition was repeated. Recovery220 reran the necessary incomplete neutral and executed four previously unstarted cases in fresh directories, retaining frozen evaluator sources. Recovery association files bind exact commands/checkpoint and new orchestration source.','',
 'See `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/diagnostics/README.md` for loss-stability and capacity/observability review. Improved teacher fit under lower lr does not establish the cause of a physical failure; onset ramp/step mismatch and any subsequent structural or curriculum change require separately declared experiments.','',
 'No automatic training extension or new physical condition is authorized by this report. Parent decides the next bounded task under the active user objective.']
 (HERE/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 hashes={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (path,HERE/'FINAL_REPORT.md',Path(__file__))}
 (HERE/'FINAL_REPORT_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print(json.dumps(dict(report=str(HERE/'FINAL_REPORT.md'),all_conditions_eligible=s['all_conditions_eligible'],saved_trajectory_verification=s['saved_trajectory_verification'])))
if __name__=='__main__':main()
