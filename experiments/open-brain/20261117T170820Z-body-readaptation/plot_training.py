"""Show actual body optimization and parameter changes."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from context import HERE

initial=json.loads((HERE/'FIT/INITIAL_FIT.json').read_text())
rows=[json.loads(line) for line in (HERE/'FIT/JOURNAL.jsonl').read_text().splitlines()]
delta=json.loads((HERE/'FIT/ACTUAL_PARAMETER_UPDATES.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
axes[0].plot([0]+[r['epoch'] for r in rows],[initial['rmse_rad']*1000]+[r['development']['rmse_rad']*1000 for r in rows],c='#245d82')
axes[0].set(xlabel='Training epoch',ylabel='Development joint target RMSE (mrad)',
            title='New fixed-neck physical teacher trajectories')
names=[name for name in delta if name.startswith('reflex.') or name=='fourier']
axes[1].barh(names,[delta[name]['l2'] for name in names],color=['#bb6b48' if name=='fourier' else '#245d82' for name in names])
axes[1].set(xlabel='L2 norm of parameter update',xscale='log',title='Actual learned body parameters')
fig.savefig(HERE/'TRAINING.png',dpi=155)
print('Saved actual learning curve and parameter updates',flush=True)
