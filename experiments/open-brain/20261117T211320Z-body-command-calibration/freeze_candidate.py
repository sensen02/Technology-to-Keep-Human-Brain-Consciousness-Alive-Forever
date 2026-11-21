from scope import *
import argparse
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('checkpoint',type=Path);a=p.parse_args()
r=json.loads((a.folder/'RESULT.json').read_text())
write(HERE/'SELECTION.json',dict(checkpoint=str(a.checkpoint.resolve()),sha256=sha(a.checkpoint),calibration_all_relative=r['all_relative'],
 calibration_pass=sum(p['relative_pass'] for p in r['pairs']),calibration_total=len(r['pairs']),
 status='FROZEN_FOR_NEW_PHASE_COMPARISON' if r['all_relative'] else 'CALIBRATION_STILL_HAS_DEVIATIONS',
 formal_seeds=[6211],formal_variants=[2],teacher_relative_protocol=str(HERE/'PROTOCOL.zh-CN.md')))
print('frozen',r['all_relative'],sum(p['relative_pass'] for p in r['pairs']),len(r['pairs']))
