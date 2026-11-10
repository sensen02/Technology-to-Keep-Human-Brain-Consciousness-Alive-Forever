"""Exact observed RH contact slip partition; no removed windows or gate changes."""
import json
import numpy as np
from pathlib import Path
from identity import HERE,atomic_json,sha256


def main():
    directory=HERE/'EVAL_ROLE_2000'
    report=json.loads((directory/'RESULT.json').read_text())
    trace=directory/'trace_1ms.npz'
    with np.load(trace,allow_pickle=False) as z:
        arrays={k:z[k] for k in ['time','loads','speed','padz','control_source','com']}
    mask=arrays['control_source']==3;t=arrays['time'][mask];loads=arrays['loads'][mask];speeds=arrays['speed'][mask];padz=arrays['padz'][mask]
    forward_norm=float(np.linalg.norm(arrays['com'][-1,:2]-arrays['com'][mask][0,:2])/(t[-1]-t[0]))
    # Diagnostic sampled discrete integral matches branch slip except trapezoid/end conventions.
    leg=4;phase=np.floor(((t-5.8)%4.8)/.8).astype(int).clip(0,5)
    plan_active=phase==4  # declared teacher reference RH slot, not actual state feedback.
    weighted=loads[:,leg]*speeds[:,leg]
    actual_stance=loads[:,leg]>=.1*24.446354755878005
    rows=[]
    for name,select in [('all',np.ones(len(t),bool)),('planned_RH_slot',plan_active),('other_planned_slots',~plan_active),
                         ('actual_load_ge_0.1W',actual_stance),('actual_load_lt_0.1W',~actual_stance),
                         ('first0.5s',t<6.3),('last4.8s',t>=10.6)]:
        rows.append(dict(name=name,samples=int(select.sum()),fraction_total_load_speed_integral=float(weighted[select].sum()/weighted.sum()),
            mean_load_W=float(loads[select,leg].mean()/24.446354755878005),
            load_weighted_contact_speed=float(weighted[select].sum()/max(loads[select,leg].sum(),1e-20)),
            max_padbottom=float(padz[select,leg].max())))
    # Native acceptance is independent; this diagnostic never alters full-window 1ms gate.
    atomic_json(HERE/'RH_SLIP_DIAGNOSIS.json',dict(trace=str(trace),trace_sha256=sha256(trace),
         original_max_slip=report['autonomous']['slip_ratio_max'],original_per_leg=report['autonomous']['per_leg_slip_ratio'],
         whole_window_forward_norm_native_s=forward_norm,RH_parts=rows,
         scope='full ANN window retained; partitions overlap and are diagnostic only, plannedslots reference labels NOT actorinput or actualgait groundtruth'))
    print(rows)

if __name__=='__main__':main()
