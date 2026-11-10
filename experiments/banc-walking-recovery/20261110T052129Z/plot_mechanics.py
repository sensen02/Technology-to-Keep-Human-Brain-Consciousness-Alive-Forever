"""Paired contact-condition evidence; plots saved data only, no simulation."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent


def main():
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for column,mode in enumerate(['zero','policy']):
        for condition,label,color in [('A_adhesion','Declared adhesion on','#b24a3d'),('B_noadhesion','Adhesion off','#327a96')]:
            with np.load(R/'mechanical_diag'/f'{mode}_{condition}_1ms.npz',allow_pickle=False) as d:
                time=d['time_s'];weights=d['loads']
                speed=(weights*d['contact_speed']).sum(1)/np.maximum(weights.sum(1),1e-12)
                # Display 50ms causal moving mean. Raw data and native-unit window
                # metrics remain in saved evidence; smoothing does not redefine gates.
                smooth=np.convolve(speed,np.ones(50)/50,mode='valid')
                axes[0,column].plot(time[49:],smooth,label=label,color=color)
                axes[1,column].plot(time,d['com'][:,2],label=label,color=color,alpha=.85)
        axes[0,column].set_title('Zero commanded rate / finite-PD hold' if mode=='zero' else 'Same saved learned policy')
        axes[0,column].set_ylabel('Load-weighted foot speed (native/s)\n50ms causal display average')
        axes[0,column].set_yscale('symlog',linthresh=.002)
        axes[1,column].set_ylabel('COM height (native)')
        axes[1,column].axhline(1.20,color='black',linestyle='--',linewidth=1,label='Derived engineering minimum')
        for row in range(2):
            axes[row,column].set_xlabel('Physical simulation time (s)')
            axes[row,column].grid(alpha=.2);axes[row,column].legend(fontsize=8)
    fig.suptitle('Paired same-body contact test: adhesion sustains motion at zero command; policy still does not walk')
    fig.savefig(R/'mechanical_contact_comparison.png',dpi=150);plt.close(fig)
    print(str(R/'mechanical_contact_comparison.png'))

if __name__=='__main__':main()
