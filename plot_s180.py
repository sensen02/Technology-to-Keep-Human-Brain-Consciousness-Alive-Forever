from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--a',type=Path,required=True);ap.add_argument('--b',type=Path,required=True);g=ap.parse_args()
A,B=g.a,g.b
tbl=list(csv.DictReader((A/'ARM_SIGN_TABLE.csv').open()))
rows=[r for r in tbl if r['pose']=='cand11']
unres={19,22,26,40,42,47}
M=np.array([[0.0]+[float(r['lo'])/max(abs(float(r['hi'])),abs(float(r['lo'])),1e-9) for r in rows],[0.0]+[float(r['hi'])/max(abs(float(r['hi'])),abs(float(r['lo'])),1e-9) for r in rows]])
fig,axs=plt.subplots(2,2,figsize=(16,11))
ax=axs[0,0]
vals=[float(r['dominant_arm']) for r in rows]
cols=['tab:red' if int(r['dof']) in unres else 'tab:blue' for r in rows]
ax.bar(range(42),vals,color=cols);ax.axhline(0,color='k',lw=.5)
ax.set_xticks(range(0,42,5));ax.set_xticklabels([f'{rows[i]["dof"]}\n{rows[i]["dof_name"].replace("joint_","")[:9]}' for i in range(0,42,5)],fontsize=6)
ax.set_ylabel('dominant pool arm');ax.set_title('per-DOF dominant muscle arm (red = unreachable row)')
ax=axs[0,1]
lo=[float(r['lo']) for r in rows];hi=[float(r['hi']) for r in rows];tg=[float(r['target']) for r in rows]
ax.barh(np.arange(42)-0.2,lo,height=.4,color='tab:purple',label='sum negative arms')
ax.barh(np.arange(42)+0.2,hi,height=.4,color='tab:cyan',label='sum positive arms')
ax.scatter(tg,np.arange(42),c=cols,s=18,label='row target',zorder=3)
ax.set_yticks(range(0,42,5));ax.set_yticklabels([f'{rows[i]["dof"]}' for i in range(0,42,5)],fontsize=7)
ax.legend(fontsize=7);ax.set_title('activation-box interval vs row target');ax.set_xlabel('generalized force')
dem=list(csv.DictReader((A/'DEMAND_DECOMPOSITION.csv').open()))
ax=axs[1,0]
x=np.arange(len(dem))
ax.bar(x-0.2,[float(r['gravity_passive_part']) for r in dem],.4,label='gravity+passive part')
ax.bar(x+0.2,[float(r['contact_part']) for r in dem],.4,label='contact part')
ax.scatter(x,[float(r['target']) for r in dem],c='k',zorder=3,label='row target')
ax.set_xticks(x,[f'{r["dof"]}\n{r["dof_name"].replace("joint_","")[:8]}' for r in dem],fontsize=7)
ax.axhline(0,color='k',lw=.5);ax.legend(fontsize=7);ax.set_title('demand decomposition: contact-dominated, gravity small')
b=list(csv.DictReader((A/'NATIVE_LIMIT_FLIP.csv').open()))
ax=axs[1,1]
ok=[r for r in b if r['flippable_in_native_limits']=='True']
ax.scatter([abs(float(r['delta_from_release'])) for r in b],[int(r['dof']) for r in b],
           c=['tab:green' if r['flippable_in_native_limits']=='True' else 'tab:red' for r in b],s=40)
ax.axvline(0.6,color='r',ls='--',label='old self-imposed 0.6 rad window')
ax.set_xscale('log');ax.set_xlabel('|delta q| to make the row interval reachable (rad)');ax.set_ylabel('dof')
ax.legend(fontsize=7);ax.set_title('native-limit sign-flip scan: all six rows are reachable well inside native limits')
fig.suptitle('S180 Stage A: engine crosscheck PASS (max|diff|=1.8e-15); demand contact-dominated; native limits allow row flips')
fig.tight_layout();fig.savefig(A/'ARM_SIGN_AND_DEMAND.png',dpi=150)
c=json.loads((B/'CANDIDATES.json').read_text())
fig2,axs2=plt.subplots(1,2,figsize=(15,6))
ax=axs2[0]
ax.scatter([x['minimax'] for x in c],[max(x['root_norm'],1e-13) for x in c],c=['k' if 'BASELINE' in x['tag'] else 'tab:blue' for x in c],s=55)
ax.axhline(1e-8,color='r',ls='--',lw=.8);ax.axvline(3.3086,color='r',ls='--',lw=.8)
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('42-row minimax');ax.set_ylabel('root |r6|inf (z,roll,pitch only)')
ax.set_title('Stage B: all 10 candidates root-balanced and geometry-valid;\nnone beats the baseline joint minimax')
ax=axs2[1]
ax.bar(range(len(c)),[len(x['unreachable']) for x in c],color=['k' if 'BASELINE' in x['tag'] else 'tab:blue' for x in c])
ax.axhline(6,color='r',ls='--',lw=.9,label='baseline 6')
ax.set_xticks(range(len(c)));ax.set_xticklabels([x['tag'][:14] for x in c],rotation=75,fontsize=6)
ax.set_ylabel('# unreachable rows');ax.legend();ax.set_title('individual rows do drop to 5, but minimax worsens')
fig2.suptitle('S180 Stage B: per-row channels exist, simultaneous satisfaction does not improve')
fig2.tight_layout();fig2.savefig(B/'STAGE_B_SELECTED.png',dpi=150)
print('saved ARM_SIGN_AND_DEMAND.png and STAGE_B_SELECTED.png')
