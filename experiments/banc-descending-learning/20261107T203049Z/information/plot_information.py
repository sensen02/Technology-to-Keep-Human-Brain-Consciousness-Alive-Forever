"""Preregistered diagnostic figure from saved fit records; no simulation steps."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REPS=('encoded4','MNinstant','MNhistory','rawphysical')
LEGS=('LF','LM','LH','RF','RM','RH')
TITLES={'stance_air':' stance / air (test, balanced acc.)',
        'knee_velocity_sign':' knee-velocity sign (test, balanced acc.)',
        'signed_knee_velocity':' signed knee velocity (test, $R^2$)'}


def main():
    r=json.loads((HERE/'FIT_RESULTS.json').read_text());a=json.loads((HERE/'ENCODER_AUDIT.json').read_text())
    fits={(f['leg'],f['representation'],f['task']):f for f in r['fits']}
    fig,axes=plt.subplots(2,2,figsize=(15.5,10.5));fig.suptitle('BANC offline information diagnostic 20261107T203049Z: frozen coarse encoder vs same-leg MN readout\n'
        'PREREGISTERED 72 fits, ridge lambda=1, train=gen0+1 (24 episodes), test=gen2+3 (24 episodes). Recoverability from correlated saved data, NOT perception/movement success.',fontsize=11)
    panels=[('stance_air',axes[0,0]),('knee_velocity_sign',axes[0,1]),('signed_knee_velocity',axes[1,0])]
    for task,ax in panels:
        grid=np.full((len(LEGS),len(REPS)),np.nan)
        for i,leg in enumerate(LEGS):
            for j,rep in enumerate(REPS):
                m=fits[(leg,rep,task)]['metrics']
                value=m.get('balanced_accuracy_test') if task!='signed_knee_velocity' else m.get('r2_test')
                if value is not None:grid[i,j]=value
        ax.imshow(grid,cmap='viridis',vmin=0 if task!='signed_knee_velocity' else -0.2,vmax=1.0,aspect='auto')
        for i in range(len(LEGS)):
            for j in range(len(REPS)):
                v=grid[i,j]
                ax.text(j,i,'undefined' if np.isnan(v) else f'{v:.3f}',ha='center',va='center',
                        color=('white' if (np.isnan(v) or v<0.55) else 'black'),fontsize=10)
        ax.set_xticks(range(len(REPS)));ax.set_xticklabels(REPS);ax.set_yticks(range(len(LEGS)));ax.set_yticklabels(LEGS)
        ax.set_title(TITLES[task]+(' (chance=0.5)' if task!='signed_knee_velocity' else ' (0=no better than train mean)'))
        if task=='stance_air':ax.set_xlabel('encoded4 stance/air is TAUTOLOGICAL by design: campaniform strain contains foot load',fontsize=9.5,color='#7a1f1f')
        if task=='knee_velocity_sign':ax.set_xlabel('frozen hook_direction is 0.05*tanh(|v|/10): the sign is not globally identifiable from it',fontsize=9.5,color='#7a1f1f')
        if task=='signed_knee_velocity':ax.set_xlabel('rawphysical contains the target column: R2 = 1-1e-7 (lambda=1 shrinkage)\nprivileged pipeline consistency check, not a learned result',fontsize=9.5,color='#7a1f1f')
    ax=axes[1,1]
    ranks=a['ranks']
    labels=['q42','qvel42','foot_load6','all 90\ninputs']
    captured=[ranks['q_block_24x42'],ranks['qvel_block_24x42'],ranks['foot_load_block_24x6'],ranks['channel_map_24x90']]
    totals=[ranks['nonzero_q_columns']+ranks['zero_q_columns'],ranks['nonzero_qvel_columns']+ranks['zero_qvel_columns'],ranks['nonzero_foot_load_columns']+ranks['zero_foot_load_columns'],90]
    x=np.arange(len(labels))
    ax.bar(x-0.2,totals,0.4,label='input columns in block',color='#b0b7c3')
    ax.bar(x+0.2,captured,0.4,label='independent directions the 24 channels capture',color='#2b6cb0')
    for i,(t,c) in enumerate(zip(totals,captured)):ax.text(i+0.2,c+1,f'{c}',ha='center',fontsize=10,color='#2b6cb0')
    ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylabel('count');ax.legend(loc='upper right',fontsize=9)
    zero_total=len(a['zero_columns'])
    ax.set_title(f'Frozen encoder input sensitivity: rank {ranks["channel_map_24x90"]} of 90 ({zero_total} exactly zero columns)')
    zero_q=ranks['zero_q_columns'];zero_v=ranks['zero_qvel_columns']
    ax.set_xlabel(f'{zero_q} q columns zero (all 3 trochanter angles x 6 legs) | {zero_v} qvel columns zero (all non-knee speeds)\n'
                  f'{ranks["q_block_24x42"]} independent q directions from {ranks["nonzero_q_columns"]} nonzero q columns | 18/18 engineered alias fixtures bit-identical\n'
                  f'Jacobian vs central finite difference (1e-6) max error {a["jacobian_max_error_vs_central_difference"]:.2e}; actual saved currents reproduced with error {a["recorded_current_max_error_nA"]}',
                  fontsize=8.5,va='top')
    fig.tight_layout(rect=(0,0,1,0.94))
    out=HERE/'FIT_SUMMARY.png';fig.savefig(out,dpi=140);plt.close(fig)
    print(json.dumps({'figure':str(out),'bytes':out.stat().st_size,'fits_plotted':sum(1 for f in r['fits'] if f['fitting_status']=='FITTED'),'physics_steps':0,'neural_steps':0}))


if __name__=='__main__':main()
