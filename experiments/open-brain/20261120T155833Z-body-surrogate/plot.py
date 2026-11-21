import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from paths import HERE
from matplotlib.font_manager import FontProperties
from pathlib import Path

font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():
    plt.rcParams['font.family']=FontProperties(fname=font).get_name()
plt.rcParams['axes.unicode_minus']=False
r=json.loads((HERE/'evaluation.json').read_text())
with np.load(HERE/'loop.npz') as z:
    ref=z['reference'][:,0]; p=z['poses'][:,:,0]; e=z['brain_in']; d=z['arrived']
fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
labels=['物理身体','替代身体 · 给定足压','替代身体 · 自算足压','自算足压 · 关闭前足上行']
colors=['#23272b','#137e75','#c56423','#8276aa']
for a,l,c in zip([ref,p[:,0],p[:,1],p[:,2]],labels,colors):
    xy=a[:,:2]-a[:1,:2]
    ax[0,0].plot(*xy.T,color=c,label=l)
    ax[0,0].plot(*xy[-1],'.',color=c)
ax[0,0].set(aspect='equal',xlabel='x 位移 (mm)',ylabel='y 位移 (mm)',title='同一起步与脑端驱动 · 2 秒轨迹')
ax[0,0].legend(fontsize=8)
t=np.arange(400)*.005
for i in range(3):
    ax[0,1].plot(t,np.linalg.norm(p[:,i]-ref,axis=-1),color=colors[i+1],label=labels[i+1])
ax[0,1].set(xlabel='时间 (s)',ylabel='胸部标记误差 (mm)',title='连续自回归替换')
keys=['network_marker_mm','velocity_marker_mm','hold_marker_mm']
v=[1000*np.mean([a[k] for a in r]) for k in keys]
bars=ax[1,0].bar(['身体网络','匀速外推','保持姿态'],v,color=[colors[1],'#7b8791','#a8b0b5'])
ax[1,0].bar_label(bars,fmt='%.1f')
ax[1,0].set(ylabel='标记平均误差 (µm)',ylim=(0,67),title='留出六回合 · 5 ms 动作预测')
for col,side in enumerate(['左前足','右前足']):
    rate=e[:,1,col].reshape(-1,500).sum(-1)*20
    ax[1,1].step((np.arange(len(rate))+1)*.05,rate,where='post',label=side,color=colors[col+1])
ax[1,1].set(xlabel='时间 (s)',ylabel='入脑事件率 (Hz，50 ms 窗)',title='自算足压 · 实际过颈后的脑侧读数')
ax[1,1].legend(fontsize=8)
for a in ax.flat:a.spines[['right','top']].set_visible(False)
fig.savefig(HERE/'BODY_SURROGATE.png',dpi=135)
