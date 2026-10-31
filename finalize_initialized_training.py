"""Persist verified matched checkpoint decision; never executes physics."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output
    evidence=json.loads((out/'INDEPENDENT_CHECK.json').read_text())
    assert evidence['passed'] and not (out/'CHECKPOINT.json').exists()
    cfg=json.loads((out/'ACCEPTANCE_PROTOCOL.json').read_text())
    assert cfg['require_composite_strict_decrease'] and cfg['require_safety_nonworsening_in']==['early','preparation','formal']
    arms={r['label']:r for r in evidence['checks']}
    b,c=arms['baseline'],arms['updated']; tol=cfg['tolerance']
    safety=all(c[phase][key]>=b[phase][key]-tol if key=='min_height_ratio'
               else c[phase][key]<=b[phase][key]+tol
               for phase in cfg['require_safety_nonworsening_in'] for key in cfg['safety_metrics'])
    keys=('support_ms','longest_all_gates_true_endpoint_span_ms')
    support=all(c['formal'][key]>=b['formal'][key]-tol for key in keys) and any(c['formal'][key]>b['formal'][key]+tol for key in keys)
    accepted=bool(c['optimization_loss']<b['optimization_loss'] and safety and support)
    assert accepted==evidence['comparisons']['updated']['physical_joint_acceptance']
    with np.load(out/'DIRECTION.npz',allow_pickle=False) as z: baseline=z['weights'].copy()
    with np.load(out/'UPDATE.npz',allow_pickle=False) as z: candidate=z['updated'].copy()
    retained=candidate if accepted else baseline
    np.savez(out/'CHECKPOINT.npz',weights=retained)
    record=dict(accepted=bool(accepted),selected_arm='updated' if accepted else 'baseline',
        decision='RETAIN_VERIFIED_UPDATE_FOR_LATER_VALIDATION' if accepted else 'REJECT_UPDATE_RETAIN_SOURCE_WEIGHTS',
        source_weights=json.loads((out/'PREREG.json').read_text())['same_weights'],
        checkpoint=str(out/'CHECKPOINT.npz'),checkpoint_sha256=hashlib.sha256((out/'CHECKPOINT.npz').read_bytes()).hexdigest(),
        independent_check_sha256=hashlib.sha256((out/'INDEPENDENT_CHECK.json').read_bytes()).hexdigest(),
        acceptance_protocol_sha256=hashlib.sha256((out/'ACCEPTANCE_PROTOCOL.json').read_bytes()).hexdigest(),
        comparison=evidence['comparisons']['updated'],deployment=False,behavior_claim='NONE',
        note='Probe arms never selected; retained source weights do not imply warm initialization is physically accepted')
    (out/'CHECKPOINT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
