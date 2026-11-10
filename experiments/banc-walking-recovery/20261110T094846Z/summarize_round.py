"""Bounded round summary, source ledger and plots from final experiment artifacts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pairs=[('20261110T091009Z','20261110T074936Z','actor previous-action input masking'),
      ('20261110T091600Z','20261110T074936Z','expert quintic swing and continuous support allocation'),
      ('20261110T093347Z','20261110T091600Z','contact transfer/retiming controlled physical validation'),
      ('20261110T094051Z','20261110T074936Z','smooth validated demonstrations, original actor'),
      ('20261110T094554Z','20261110T094051Z','train-only frozen actor input standardization'),
      ('20261110T094846Z','20261110T094051Z','off-trajectory assisted imitation data collection')]
    for new,old,reason in pairs:
        directory=ROOT/new;ledger=directory/'CHANGE_LEDGER.json'
        if ledger.exists():continue
        entries=[]
        for name in ['models.py','student_wave.py','static_teacher.py','wave_reference.py','physics.py','recovery_bc.py','PREREG.json']:
            path=directory/name;previous=ROOT/old/name
            if not path.exists():continue
            entries.append(dict(old_file=str(previous) if previous.exists() else None,
              old_sha256=sha(previous) if previous.exists() else None,new_file=str(path),new_sha256=sha(path),
              changed=not previous.exists() or sha(previous)!=sha(path),reason=reason))
        ledger.write_text(json.dumps(dict(parent=str(ROOT/old),entries=entries),indent=2)+'\n')
    variants=[('Original FF','20261110T074936Z/EVAL_FF238_HANDOVER'),
      ('Original GRU','20261110T074936Z/EVAL_GRU238_HANDOVER'),
      ('Masked FF','20261110T091009Z/EVAL_FF_MASK'),('Masked GRU','20261110T091009Z/EVAL_GRU_MASK'),
      ('Smooth FF','20261110T094051Z/EVAL_FF_SMOOTH'),('Smooth GRU','20261110T094051Z/EVAL_GRU_SMOOTH'),
      ('Normalized FF','20261110T094554Z/EVAL_FF_NORM'),('Normalized GRU','20261110T094554Z/EVAL_GRU_NORM'),
      ('Recovery FF','20261110T094846Z/EVAL_FF_RECOVERY')]
    results=[]
    for name,folder in variants:
        path=ROOT/folder/'RESULT.json'
        if not path.exists():continue
        result=json.loads(path.read_text())
        results.append(dict(name=name,path=str(path),sha256=sha(path),autonomous=result['autonomous'],
            transition_exact=result['transition']['all_exact'],cycles=result['autonomous_cycles']['complete_per_leg']))
    teacher=json.loads((ROOT/'20261110T093347Z/early_unload/RESULT.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(13,9),layout='constrained')
    paths=[r['name'] for r in results]
    for ax,key,title,gate in [(axes[0],'com_height_min','Autonomous minimum COM height (native)',1.2),
                               (axes[1],'nonfoot_force_max_native','Autonomous nonfoot force peak / body weight',.2)]:
        vals=[r['autonomous'][key] for r in results]
        if key.startswith('nonfoot'):vals=np.asarray(vals)/24.446354755878005
        ax.bar(paths,vals,color=['#af493c' if not r['autonomous']['all_gates'] else '#287c69' for r in results])
        teacher_value=teacher[key] if not key.startswith('nonfoot') else teacher[key]/24.446354755878005
        ax.axhline(gate,color='#b00020',ls='--',label='unchanged derived engineering gate')
        ax.axhline(teacher_value,color='#287c69',ls=':',label='smooth algorithm teacher (not autonomous)')
        ax.tick_params(axis='x',rotation=25);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Actual continuous neural takeover: controlled changes, failures retained; teacher pass is separate')
    plot=HERE/'ROUND2_PHYSICAL_RESULTS.png';fig.savefig(plot,dpi=140);plt.close(fig)
    summary=dict(results=results,teacher_scope='algorithm assisted, allfivegates pass; not autonomous or finalacceptance',
       interpretation='architecture training budgets differ FF1000/GRU300; comparisons are per architecture; no best-score success claim')
    (HERE/'ROUND2_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('completed_neural_conditions',len(results),'all_transition_exact',all(r['transition_exact'] for r in results),
          'all_gate_pass_names',[r['name'] for r in results if r['autonomous']['all_gates']],str(plot))

if __name__=='__main__':main()
