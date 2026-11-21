"""Scientific command/teacher diagnosis and same-run physical replay footage."""
from scope_diag import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont


def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def main():
    result=read(HERE/'FINAL_LOCALIZATION.json');pins={};t=(np.arange(20000)+1)*DT
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for row,seed in enumerate(SEEDS):
        for name,color in (('reference','black'),('translated','tab:blue')):
            f=HERE/'BODY'/f'seed{seed}'/(name+'_ann')/'TRACE.npz';a=load(f);pins[str(f)]=sha(f)
            axes[row,0].plot(t,a['command'].mean(1),label=name,color=color)
            axes[row,1].plot(t,(a['command'][:,1]-a['command'][:,0])/2,label=name,color=color)
            axes[row,2].plot(t,np.rad2deg(np.unwrap(a['yaw'])-a['yaw'][0]),label=name+' ANN',color=color)
            f=HERE/'BODY'/f'seed{seed}'/(name+'_teacher')/'TRACE.npz';a=load(f);pins[str(f)]=sha(f)
            axes[row,2].plot(t,np.rad2deg(np.unwrap(a['yaw'])-a['yaw'][0]),'--',label=name+' teacher',color=color)
        for col,label in enumerate(('Mean L/R drive','Half R-L drive','Physical yaw (degrees)')):
            axes[row,col].set_ylabel(f'Seed {seed}: '+label);axes[row,col].set_xlabel('Time (s)')
            axes[row,col].legend(fontsize=8);axes[row,col].grid(alpha=.25)
    fig.suptitle('Frozen commands replayed into the same full physical model; brain feedback disconnected')
    fig.tight_layout();fig.savefig(HERE/'COMMAND_AND_BODY.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,7))
    labels=['Candidate','Reference','Ref advance + candidate turn','Candidate advance + ref turn',
            'Replace all steering DNs','Replace DNa02','Replace DNa01','Replace DNg13','Replace propulsion DNs']
    names=['translated','reference','refadvance_studentturn','studentadvance_refturn',
           'swap_all_steer','swap_DNa02','swap_DNa01','swap_DNg13','swap_propulsion']
    for ax,row in zip(axes,result['rows'],strict=True):
        values=[row['ann_yaw'][n] for n in names];ax.barh(np.arange(len(names)),values,color=['tab:blue','black']+['.55']*7)
        ax.set_yticks(np.arange(len(names)),labels,fontsize=8);ax.invert_yaxis()
        ax.axvline(row['ann_yaw']['reference'],color='tab:orange',ls='--',label='Reference yaw')
        for i,v in enumerate(values):ax.text(v+.3,i,f'{v:.2f}',va='center',fontsize=8)
        ax.set_xlabel('Physical yaw (degrees)');ax.set_title(f"Seed {row['seed']}; conditional replacements")
        ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Interventions are nonlinear and nonadditive; no weights or gains changed')
    fig.tight_layout();fig.savefig(HERE/'CHANNEL_INTERVENTIONS.png',dpi=140);plt.close(fig)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    items=[]
    for name in ('reference','translated'):
        row=[]
        for mode in ('ann','teacher'):
            f=HERE/'BODY/seed10901'/(name+'_'+mode);m=mujoco.MjModel.from_binary_path(str(f/'BODY.mjb'));d=mujoco.MjData(m)
            a=load(f/'TRACE.npz');m.mat_reflectance[:]=0;renderer=mujoco.Renderer(m,height=240,width=400)
            row.append((m,d,a,renderer,name,mode));pins[str(f/'TRACE.npz')]=sha(f/'TRACE.npz');pins[str(f/'BODY.mjb')]=sha(f/'BODY.mjb')
        items.append(row)
    def frame(item,tick):
        m,d,a,renderer,name,mode=item;d.qpos[:]=a['qpos'][tick];mujoco.mj_kinematics(m,d)
        camera=mujoco.MjvCamera();camera.lookat[:]=[*a['position'][tick,:2],.85];camera.distance=8.;camera.azimuth=105.;camera.elevation=-23.
        renderer.update_scene(d,camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,400,48),fill=(20,20,20));draw.text((5,2),f'{name} command | {mode} body',font=font,fill='white')
        draw.text((5,24),f't={(tick+1)*DT:.3f}s | command replay | 0.125x',font=font,fill='white')
        return np.asarray(im)
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    for r,row in enumerate(items):
        for c,item in enumerate(row):axes[r,c].imshow(frame(item,14999));axes[r,c].axis('off')
    fig.suptitle('Same command, ANN vs teacher; actual new physical replay at 1.5s')
    fig.tight_layout();fig.savefig(HERE/'BODY_REPLAY_STATES.png',dpi=130);plt.close(fig)
    writer=imageio.get_writer(str(HERE/'COMMAND_BODY_REPLAY.mp4'),fps=25,codec='libx264',quality=7,macro_block_size=16)
    count=0
    try:
        for tick in range(0,20000,50):
            writer.append_data(np.concatenate([np.concatenate([frame(item,tick) for item in row],axis=1) for row in items],axis=0));count+=1
    finally:
        writer.close()
        for row in items:
            for _,_,_,renderer,_,_ in row:renderer.close()
    verify(pins)
    write(HERE/'MEDIA.json',dict(frames=count,fps=25,video_seconds=count/25,simulated_seconds=2.,playback=.125,
      source_sha256=pins,video=str(HERE/'COMMAND_BODY_REPLAY.mp4'),video_sha256=sha(HERE/'COMMAND_BODY_REPLAY.mp4'),
      render_only_change='floor reflectance zero; no pose, geometry or trajectory edits',new_physical_steps=0,new_neural_steps=0))
    print('rendered',count,'new replay frames',flush=True)


if __name__=='__main__':main()
