"""Render measured closure traces, no control or simulation changes."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import mujoco
from run_neural_muscle_closure import prepare_body, OUT
import engine.memory_guard as mg


def main():
    doc=json.loads((OUT/'closure_results.json').read_text())
    z=np.load(OUT/'closure_traces.npz')
    episodes=doc['measured']['episodes']
    arms=['normal','sensory_cut','motor_cut','spike_output_silence']
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for arm in arms:
        ep=episodes[arm]; t=np.array([r['time_end_s'] for r in ep['records']])*1000
        axes[0,0].plot(t,[r['height_model']/ep['initial_height_model'] for r in ep['records']],label=arm)
        axes[0,1].plot(t-5,z[arm+'_controls'].sum(axis=1),label=arm)
        axes[1,0].plot(t,[r['motor_spikes'] for r in ep['records']],label=arm)
    axes[0,0].axhline(.9,c='k',ls='--',lw=1)
    axes[0,0].set(title='Free-body height (native control floor retained)',ylabel='height / initial height',xlabel='time (ms)')
    axes[0,1].set(title='Actual neural commands applied to muscles',ylabel='sum of normalized controls',xlabel='interval start (ms)')
    axes[1,0].set(title='Mapped motor-pool spikes (5 ms intervals)',ylabel='spikes (unique mapped cells)',xlabel='interval end (ms)')
    for ax in axes.flat[:3]: ax.legend(fontsize=8); ax.grid(alpha=.25)
    axes[1,1].axis('off')
    text=('ANNOTATION / INTERFACE\n74 usable muscles; 65 complete pools, 9 partial\n16 unresolved: neural command zero\n\nMEASURED SHORT-EPISODE OUTCOME\nNormal: 24.0 ms to 90% initial height\nInput/output silence: 14.9 ms\nMotor cut at 15 ms: 19.6 ms\nRepeat + replay: exactly identical\nExtra LF touch: no command difference\n\nLIMITS\nStill falls; NOT stable standing or walking\nNative minimum control = 0.0001\nStrict neural-only actuation: NOT established\nDeclared encoder/translation, not biology\nSI mass/force calibration unresolved')
    axes[1,1].text(0,1,text,va='top',fontsize=11)
    fig.suptitle('Genuine motor-pool path: causal response, not autonomous locomotion',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(OUT/'closure_summary.png',dpi=160);plt.close(fig)
    model,seated,*_=prepare_body()
    renderer=mujoco.Renderer(model,height=480,width=640)
    cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=[0,0,1.3]; cam.distance=12;cam.azimuth=120;cam.elevation=-25
    frames=[]
    for k in range(len(z['normal_trajectory'])+1):
        panels=[]
        for arm in ('normal','spike_output_silence'):
            d=mujoco.MjData(model)
            if k==0:d.qpos[:]=seated
            else:
                row=z[arm+'_trajectory'][k-1]
                d.qpos[:]=row[:model.nq];d.qvel[:]=row[model.nq:model.nq+model.nv];d.act[:]=row[-model.na:]
            mujoco.mj_forward(model,d);renderer.update_scene(d,camera=cam)
            panels.append(renderer.render().copy())
        frame=np.concatenate(panels,axis=1)
        # Labels are rendered by matplotlib to keep frame annotations explicit.
        f,ax=plt.subplots(figsize=(12.8,5.2),dpi=100)
        ax.imshow(frame);ax.axis('off')
        f.suptitle(f'Normal closure (left) | Spike-output silence (right) | t={k*5} ms\n50 ms measured trajectory; slowed playback; neither maintains initial height',fontsize=11,y=.98)
        f.subplots_adjust(left=0,right=1,bottom=0,top=.89);f.canvas.draw()
        frames.append(np.asarray(f.canvas.buffer_rgba())[:,:,:3].copy());plt.close(f)
    renderer.close()
    imageio.mimsave(OUT/'closure_comparison.mp4',frames,fps=5,macro_block_size=1)
    imageio.imwrite(OUT/'closure_video_first.png',frames[0])
    imageio.imwrite(OUT/'closure_video_last.png',frames[-1])
    print('Rendered summary, 11-frame slowed video, first/last stills',OUT)


if __name__=='__main__':
    with mg.guarded_run('render_neural_muscle_closure', **mg.BUDGETS['render']):
        main()
