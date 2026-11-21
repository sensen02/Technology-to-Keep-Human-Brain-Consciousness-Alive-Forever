"""Render energy, behavior and feeding outcomes from the persisted evidence."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from assess import CASES, read_case

LABELS = dict(FRESH='地面果有食·自主', EMPTY_GROUND='地面果耗尽·自主', GUIDED='给定外枝果')
COLORS = dict(FRESH='#087f72', EMPTY_GROUND='#b25425', GUIDED='#6247a5')
FRUITS_MM = ((42.5,-8.2),(72.5,3.8),(84.5,38.5),(116.8,75.5),
             (123.5,-35.8),(135.7,-52.3),(153.6,-40.4))


def draw(root, output, diagnostics=None, ground_extension=None):
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font = font_manager.FontProperties(family='Noto Sans CJK JP')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplot_mosaic([['energy','path'],['food','activity'],['hunger','altitude']],
                                  figsize=(13,10.5),layout='constrained')
    fig.suptitle('P1 果园觅食与能量 · 三组同起点实测', x=.017, ha='left', fontsize=19,
                 fontproperties=font)
    for i,(x,y) in enumerate(FRUITS_MM):
        ax=axs['path']
        ax.scatter(x,y,c='#cc3342',s=81,alpha=.35 if i>2 else .85,zorder=3)
        if i in (0,1,2):
            ax.annotate(('近地果','远地果','外枝果')[i],(x,y),xytext=(6,6),
                        textcoords='offset points',fontsize=9,fontproperties=font)
    groups = [(root,case,LABELS[case],COLORS[case],'-') for case in CASES]
    if diagnostics:
        for case,title in (('FRESH','地面制动复测'),('GUIDED','平滑航向引导复测')):
            source = ground_extension if case=='FRESH' and ground_extension else diagnostics
            if (source/case/'RESULT.json').exists():
                groups.append((source,case,title,COLORS[case],'--'))
    for group,case,label,color,style in groups:
        result,trace,meals,_,score = read_case(group/case)
        m = {name:meals[:,i] for i,name in enumerate(result['meal_columns'])}
        axs['energy'].plot(m['time_s'],m['reserve'],color=color,lw=2.,ls=style,label=label)
        axs['food'].plot(m['time_s'],np.cumsum(m['ingested']),color=color,lw=2.,ls=style)
        axs['hunger'].plot(m['time_s'],m['hunger'],color=color,lw=2.,ls=style)
        axs['altitude'].plot(trace[::22,0],10*trace[::22,3],color=color,lw=1.65,ls=style)
        axs['path'].plot(10*trace[::14,1],10*trace[::14,2],color=color,lw=1.8,ls=style)
        axs['path'].scatter(10*trace[-1,1],10*trace[-1,2],s=22,color=color,zorder=5)
        if result['first_air_s'] is not None:
            idx = np.searchsorted(trace[:,0],result['first_air_s'])
            axs['path'].scatter(10*trace[idx,1],10*trace[idx,2],s=70,marker='^',
                                edgecolor=color,facecolor='white',zorder=7)
        if score['first_exhaustion_s'] is not None:
            axs['energy'].scatter(score['first_exhaustion_s'],0.,marker='x',color=color,
                                  s=46,zorder=5)
        if group==root:
            costs=score['running_budget_by_mode']
            base=0.
            for mode,shade in zip(costs,('#9cabb1','#5f819b','#d18538','#9d4a60'),strict=True):
                axs['activity'].barh(label,costs[mode],left=base,color=shade)
                base+=costs[mode]
    axs['path'].scatter(0.,0.,marker='+',color='#272c2e',s=80,zorder=6)
    axs['path'].set(xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
    axs['energy'].legend(frameon=False,prop=font,fontsize=9)
    axs['energy'].set(ylabel='工程能量单位',ylim=(-.008,None),xlabel='模拟时间 (s)')
    axs['food'].set(ylabel='工程食物体积',xlabel='模拟时间 (s)')
    axs['activity'].set(xlabel='累计耗能（工程单位）')
    axs['activity'].invert_yaxis()
    stage_names=('静息','步行驱动','蹬伸','拍翅飞行')
    stage_colors=('#9cabb1','#5f819b','#d18538','#9d4a60')
    axs['activity'].set_ylim(2.6,-.94)
    axs['activity'].legend(handles=[Patch(color=color,label=name) for name,color in zip(stage_names,stage_colors,strict=True)],
                           loc='upper right',frameon=False,prop=font,ncol=4,handlelength=1.2,fontsize=8)
    axs['activity'].axvline(.24, color='#30383b',alpha=.38,ls=':')
    axs['hunger'].set(xlabel='模拟时间 (s)',ylabel='归一化 0–1',ylim=(0,1.02))
    axs['altitude'].set(xlabel='模拟时间 (s)',ylabel='质心高度 (mm)')
    axs['altitude'].axhline(6.8,ls=':',color='#6d7070',alpha=.45)
    for name,title in dict(energy='可用储备 × 首次耗竭',path='实际轨迹 △ 离地',
                           food='累计实际摄入',activity='各动作阶段分账',
                           hunger='饥饿代理 · 由循环营养计算',altitude='实际飞行高度').items():
        ax = axs[name]
        ax.set_title(title,loc='left',fontsize=13,fontproperties=font)
        ax.grid(alpha=.19)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(output,dpi=160,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--diagnostics',type=Path)
    parser.add_argument('--ground-extension',type=Path)
    args=parser.parse_args()
    draw(args.root,args.output,args.diagnostics,args.ground_extension)
