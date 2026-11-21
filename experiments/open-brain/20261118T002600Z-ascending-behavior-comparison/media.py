"""Render only saved states from the formal live brain/neck/body runs."""
from neural_runtime import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES=('native_full','reference_boundary','translated','no_up')
LABELS=('Native brain + engineered sense','Reference AN boundary','Trained post-neck translation','No ascending input to brain')
COLORS=('black','tab:orange','tab:blue','tab:green')


def load_arrays(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def main():
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    pairs=[];pins={};fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,seed in enumerate((10901,11311)):
        base=HERE/'FORMAL'/f'seed{seed}';items=[]
        for mode,label,color in zip(MODES,LABELS,COLORS,strict=True):
            arrays=load_arrays(base/mode/'TRACE.npz');result=read(base/mode/'RESULT.json')
            model=mujoco.MjModel.from_binary_path(str(base/(mode+'.mjb')));data=mujoco.MjData(model)
            model.mat_reflectance[:]=0
            renderer=mujoco.Renderer(model,height=240,width=400)
            items.append((model,data,arrays,result,renderer));t=(np.arange(len(arrays['position']))+1)*DT
            axes[row,0].plot(t,arrays['position'][:,0]-arrays['initial_position'][0],label=label,color=color)
            axes[row,1].plot(t,np.rad2deg(np.unwrap(arrays['yaw'])-arrays['yaw'][0]),label=label,color=color)
            for file in (base/mode/'TRACE.npz',base/mode/'RESULT.json',base/(mode+'.mjb')):pins[str(file)]=sha(file)
        for ax in axes[row]:
            ax.axvspan(.22,1.5,color='gray',alpha=.08);ax.grid(alpha=.25);ax.set_xlabel('Time (s)');ax.legend(fontsize=7)
        axes[row,0].set_ylabel(f'Seed {seed}: forward X (mm)');axes[row,1].set_ylabel('Yaw (degrees)')
        pairs.append(items)
    fig.suptitle('Live independent bodies / own sensory feedback; common upstream PFL3 stimulation')
    fig.tight_layout();fig.savefig(HERE/'BEHAVIOR_METRICS.png',dpi=140);plt.close(fig)
    def frame(item,tick,label,seed):
        model,data,a,r,renderer=item;data.qpos[:]=a['qpos'][tick];mujoco.mj_kinematics(model,data)
        camera=mujoco.MjvCamera();camera.lookat[:]=[*a['position'][tick,:2],.85]
        camera.distance=8.;camera.azimuth=105.;camera.elevation=-23.
        renderer.update_scene(data,camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,400,48),fill=(20,20,20));draw.text((5,2),label,font=font,fill='white')
        draw.text((5,24),f'seed{seed} t={(tick+1)*DT:.3f}s  playback0.125x',font=font,fill='white')
        return np.asarray(im)
    fig,axes=plt.subplots(4,4,figsize=(16,10))
    for row,(seed,tick) in enumerate(((10901,4999),(10901,14999),(11311,4999),(11311,14999))):
        items=pairs[0 if seed==10901 else 1]
        for col,(item,label) in enumerate(zip(items,LABELS,strict=True)):
            axes[row,col].imshow(frame(item,tick,label,seed));axes[row,col].axis('off')
    fig.suptitle('Formal full bodies from saved native states; no rendered trajectory changes')
    fig.tight_layout();fig.savefig(HERE/'BODY_STATES.png',dpi=130);plt.close(fig)
    writer=imageio.get_writer(str(HERE/'PAIRED_BEHAVIOR.mp4'),fps=25,codec='libx264',quality=7,macro_block_size=16)
    count=0
    try:
        for tick in range(0,20000,50):
            rows=[]
            for seed,items in zip((10901,11311),pairs,strict=True):
                rows.append(np.concatenate([frame(item,tick,label,seed) for item,label in zip(items,LABELS,strict=True)],axis=1))
            writer.append_data(np.concatenate(rows,axis=0));count+=1
    finally:
        writer.close()
        for items in pairs:
            for _,_,_,_,renderer in items:renderer.close()
    verify(pins)
    write(HERE/'MEDIA.json',dict(video=str(HERE/'PAIRED_BEHAVIOR.mp4'),video_sha256=sha(HERE/'PAIRED_BEHAVIOR.mp4'),
        frames=count,fps=25,video_seconds=count/25,simulated_seconds=2.,playback=.125,source_sha256=pins,
        render_only_change='floor reflectance zero',new_physical_steps=0,neural_steps=0))
    print('rendered',count,'saved-state paired frames',flush=True)


if __name__=='__main__':main()
