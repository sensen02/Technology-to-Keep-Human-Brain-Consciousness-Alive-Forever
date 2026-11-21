"""Draw same-orchard comparison from the persisted neural and body traces."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

HERE=Path(__file__).resolve().parent


def draw(root,out):
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']='Noto Sans CJK JP'
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(2,2,figsize=(11,7.3),layout='constrained')
    fig.suptitle('P1 果园 · 从饱足到重新觅食',fontsize=18,x=.024,ha='left')
    for name,label,color in [('HUNGRY','低饱足','#a63f26'),('FULL','高饱足','#285d93'),
                             ('FADE','饱足自行下降','#147e70')]:
        b=np.load(root/name/'TRACE.npy',allow_pickle=False)
        state=np.load(root/name/'MEAL.npy',allow_pickle=False)
        gate=np.load(root/name/'NUTRIENT_GATE.npy',allow_pickle=False)
        times=np.arange(len(gate))*.02
        ax[0,0].plot(state[:,0],state[:,2],color=color,label=label,lw=2.2)
        ax[1,0].plot(times,gate[:,1],color=color,lw=2.)
        ax[0,1].plot(10*b[::14,1],10*b[::14,2],color=color,lw=2.)
        ax[0,1].scatter(10*b[-1,1],10*b[-1,2],color=color,s=28,zorder=4)
        ax[1,1].plot(state[:,0],np.cumsum(state[:,5]),color=color,lw=2.)
    axis=ax[0,1]
    axis.add_patch(Circle((42.5,-8.2),6.,facecolor='#d0d7d4',edgecolor='#707e78',alpha=.54,zorder=1))
    axis.add_patch(Circle((72.5,3.8),5.6,facecolor='#e7ada4',edgecolor='#a64a49',alpha=.5,zorder=1))
    axis.text(41,-19,'已空地果',ha='center')
    axis.text(73,12,'第二颗地果',ha='center')
    axis.set(xlabel='x (mm)',ylabel='y (mm)',xlim=(28,83),ylim=(-24,18),aspect='equal')
    ax[0,0].legend(loc='best',frameon=False)
    ax[0,0].set(xlabel='模拟时间 (s)',ylabel='工程循环营养信号')
    ax[1,0].axhline(.60,ls='--',lw=1.,color='#63696c',alpha=.7)
    ax[1,0].set(xlabel='模拟时间 (s)',ylabel='归一化分数',ylim=(-.03,1.03))
    ax[1,1].set(xlabel='模拟时间 (s)',ylabel='体积（工程单位）')
    for axis,title in zip(ax.flat,('身体循环营养','实际身体路线','学习层的搜索分数','实际累计摄入'),strict=True):
        axis.set_title(title,loc='left',fontsize=12)
        axis.spines[['top','right']].set_visible(False)
        axis.grid(alpha=.2)
    fig.savefig(out,dpi=158,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    draw(args.root,args.output)
