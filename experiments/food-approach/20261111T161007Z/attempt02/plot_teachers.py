from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(10,5))
for name,label in [('left_yaw_p060','LEFT teacher: not qualified'),('right_yaw_m060','RIGHT teacher: qualified')]:
    folder=HERE/name;r=json.loads((folder/'RESULT.json').read_text())
    with np.load(folder/'YAW_1MS.npz',allow_pickle=False) as z:
        x=z['time'][4800:]-5.8;y=np.rad2deg(np.unwrap(np.r_[float(z['boundary_yaw']),z['yaw_world'][4800:]]));y=y[1:]-y[0]
    ax.plot(x,y,lw=2,label=f'{label} ({r["actual_signed_yaw_deg"]:+.2f} deg)')
ax.axhline(30,color='gray',ls=':');ax.axhline(-30,color='gray',ls=':')
ax.set(xlabel='Seconds after straight warmup',ylabel='Actual heading change (degrees)',title='Large-turn physical TEACHERS, not neural autonomy\nRight: slip 0.4770, all 6 cycles, nonfoot zero | Left: slip/cycle/nonfoot failed')
ax.grid(alpha=.25);ax.legend();fig.tight_layout()
path=HERE/'TEACHER_COMPARISON.png';fig.savefig(path,dpi=150)
print(path)
