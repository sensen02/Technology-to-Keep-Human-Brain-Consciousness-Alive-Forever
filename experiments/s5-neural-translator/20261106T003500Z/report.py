"""Chinese report and same-trajectory physical replay; never starts episodes."""
import os,sys,json,hashlib
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
R=Path('/run/media/sensen/Data2/cell_wound_prototype');C=Path(__file__).resolve().parent
sys.path[:0]=[str(R),str(R/'tools')]
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator')/C.name
MODEL=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,d):(O/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def representative(c):return next(r for r in c['results'] if r['release_time_s']==1.3)
def render(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates=s['candidates'];fig,axs=plt.subplots(2,2,figsize=(13,8));names=[c['name'] for c in candidates]
    for ax,key in zip(axs.ravel(),['J','foot_support_mean','com_z_min','first_nonfoot_s']):
        vals=[representative(c)['J'] if key=='J' else (representative(c)['parts'][key] if representative(c)['parts'][key] is not None else representative(c)['duration_s']) for c in candidates];ax.bar(np.arange(len(vals)),vals,color=['#387b51' if c['guard_pass'] else '#bc6347' for c in candidates]);ax.set_xticks(np.arange(len(vals)),names,rotation=40,ha='right');ax.set_title('1.300s '+key);ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(O/'REGRESSION_GUARD.png',dpi=150);plt.close(fig)
    selection=[('OLD / zero',representative(s['zero'])),('UNCONSTRAINED',representative(s['unconstrained'])),('BEST NONZERO '+s['best_nonzero']['name'],s['representative']),('GUARDED '+s['guarded']['name'],representative(s['guarded']))]
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for name,r in selection:
        a=load(Path(r['folder'])/'NATIVE.npz')['rows'];x=a[:,0]-1.3;axs[0].plot(x,a[:,10],label=name);axs[1].plot(x,a[:,1:7].sum(1)/24.446354755878005);axs[2].plot(x,a[:,7]/24.446354755878005)
    axs[0].legend(fontsize=8);axs[0].set_ylabel('COM z');axs[1].set_ylabel('Foot load / W');axs[2].set_ylabel('Nonfoot load / W');axs[2].set_xlabel('Time after release / s')
    for ax in axs:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(O/'TRAJECTORY_COMPARISON.png',dpi=150);plt.close(fig)
    import mujoco,imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    m=_load_mjcf(MODEL).compile()
    physics=[k for k in dir(m) if not k.startswith('_') and isinstance(getattr(m,k,None),np.ndarray) and not k.startswith(('vis','light','mat_','tex_')) and k!='geom_rgba']
    ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    before={k:ah(getattr(m,k)) for k in physics};gids=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_MESH);assert len(gids)==71
    inventory=[dict(id=int(g),name=m.geom(int(g)).name,old_rgba=m.geom_rgba[g].tolist()) for g in gids];m.geom_rgba[gids,3]=1
    assert before=={k:ah(getattr(m,k)) for k in physics};m.vis.global_.offwidth=max(m.vis.global_.offwidth,640);m.vis.global_.offheight=max(m.vis.global_.offheight,480)
    dump('VISUAL_INVENTORY.json',dict(mesh_count=71,visible=71,physics_arrays_unchanged=True,geoms=inventory))
    warm=load(O/'DATA/fresh_final_warmup/BODY.npz');mask=warm['time']<1.3-1e-9;series=[]
    for label,r in selection:
        b=load(Path(r['folder'])/'BODY.npz');series.append((label,r,{k:np.concatenate([warm[k][mask],b[k]]) for k in ['time','qpos','qvel','ctrl','com']}))
    renderer=mujoco.Renderer(m,height=480,width=640);d=mujoco.MjData(m);cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.distance=8;cam.azimuth=125;cam.elevation=-28;opt=mujoco.MjvOption();mujoco.mjv_defaultOption(opt);opt.sitegroup[:]=0
    for name in ['mjVIS_TENDON','mjVIS_ACTUATOR','mjVIS_JOINT']:opt.flags[getattr(mujoco.mjtVisFlag,name)]=0
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);samples=[];mapping=[];err=0.
    with imageio.get_writer(O/'REGRESSION_COMPARISON_8x_slow.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16) as writer:
        for fi in range(301):
            when=fi*.005;panels=[];rows=[]
            for label,r,b in series:
                ix=int(np.argmin(abs(b['time']-when)));rows.append(ix);d.qpos[:]=b['qpos'][ix];d.qvel[:]=b['qvel'][ix];d.ctrl[:]=b['ctrl'][ix];mujoco.mj_forward(m,d);err=max(err,float(np.max(abs(d.subtree_com[m.body('Thorax').id]-b['com'][ix]))));cam.lookat[:]=[d.qpos[0],d.qpos[1],.9];renderer.update_scene(d,camera=cam,scene_option=opt)
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle([0,0,640,100],fill=(20,20,20));draw.text((10,8),label,font=font,fill='white');draw.text((10,34),f't={when:.3f}s | 8x slow | '+('ALGORITHM WARMUP' if when<1.3 else 'NEURAL ONLY'),font=font,fill=(255,220,90));draw.text((10,62),'Walking / intention NOT ESTABLISHED',font=font,fill='white');panels.append(np.asarray(im))
            frame=np.concatenate(panels,axis=1);writer.append_data(frame);mapping.append(dict(frame=fi,time_s=when,rows=rows))
            if fi in [0,250,260,280,300]:samples.append(frame)
    renderer.close();assert err==0.;imageio.imwrite(O/'VIDEO_SAMPLES.png',np.concatenate(samples));reader=imageio.get_reader(O/'REGRESSION_COMPARISON_8x_slow.mp4');assert reader.count_frames()==301
    for idx in [0,260,300]:assert reader.get_data(idx).shape==(480,2560,3)
    reader.close();dump('VIDEO_FRAME_MAP.json',mapping);dump('VIDEO_VERIFICATION.json',dict(frames=301,fps=25,slow_factor=8,replay_com_error=err,visible_meshes=71,physics_arrays_unchanged=True,source_warmup=str(O/'DATA/fresh_final_warmup/BODY.npz'),source_warmup_sha256=sha(O/'DATA/fresh_final_warmup/BODY.npz'),sources=[dict(label=label,folder=r['folder'],sha256=sha(Path(r['folder'])/'BODY.npz')) for label,r,_ in series],visual_inspection='NOT_COMPLETED_MODEL_HAS_NO_IMAGE_INPUT'))

def main():
    s=js(O/'SUMMARY.json');v=js(O/'VERIFICATION.json');assert v['passed'];render(s)
    rows=['|候选|1.3s J|足载均值/W|最低COM|首次非足/s|开发最差J|保护门|','|---|---:|---:|---:|---:|---:|---|']
    for c in s['candidates']:
        r=representative(c);p=r['parts'];rows.append(f"|{c['name']}|{r['J']:.9f}|{p['foot_support_mean']:.9f}|{p['com_z_min']:.9f}|{(p['first_nonfoot_s'] if p['first_nonfoot_s'] is not None else r['duration_s']):.9f}|{max(a['J'] for a in c['results']):.9f}|{c['guard_pass']}|")
    final=['|新释放时刻/s|零修正J|受保护J|最佳非零J|','|---|---:|---:|---:|']
    for a in s['final_pairs']:final.append(f"|{a['phase']}|{a['zero']['J']:.9f}|{a['guarded']['J']:.9f}|{a['best_nonzero']['J']:.9f}|")
    cleanup=js(O/'CLEANUP_LEDGER.json');prov=js(O/'DATA/PROVENANCE.json')
    text=f'''# S5 神经翻译器：已知较好释放轨迹回归保护实验

完整行为门通过 {s['behavioral_passes']}/{s['total']}；自主支撑/行走未建立。受保护选择：{s['guarded']['name']}；单列最佳非零：{s['best_nonzero']['name']}。这是有限研究实验，不是成品。绝对力/SI、走路意图保持均为 NOT_ESTABLISHED。

## 数据与有限拟合

保护集源为旧版真实1.300s释放轨迹 `{prov['protection']['source']}`，仅其满足无非足接触和绝对高度下限的连续初始前缀 {prov['protection']['prefix_rows']} 个1ms动作；首次正非足接触或绝对COM低于0.9658022579833524后不再取样。旧1.3s只是相对较好，仍未通过完整D门；此前缀也不等同于完整D门安全认证。保护目标是零残差及旧实际力矩，不将全部预热步态标为零。保护样本明确属于训练/早停FIT数据，不叫留出集。

训练只用231405档案closed_train与aggregate_1/2的既有离线修正标签；开发轨迹保持整条来源隔离，既往最终状态不纳入。所有输入从0.2s起实际预热尖峰以及释放后每个动作前已完成NEURAL.spikes204重建10/40/100ms滤波；原5/20/50ms特征另行逐位复现。原66池5ms、SNN/编码器/RH0.375/旧矩阵/执行器限幅不变。网络在线不读身体/时间/相位，无在线PD或CPG，未增加身体状态补偿。教师标签是离线参考建议，不是已证明必要动作。

罚系数0/.1/.5/2/10/50，同一冻结refit_1起点，seed20261106，Adam学习率0.0003，最多120轮，patience12。归一化实际施加力矩拟合损失加保护残差和施加力矩变化各半；开发误差加保护FIT损失早停。初始模型与零修正均保留。1.3s保护要求J不升且足载/最低COM/首次非足时刻不降，容差1e-9。选模只看开发五状态，未用新最终结果。

{chr(10).join(rows)}

## 冻结后的四个新状态

{chr(10).join(final)}

零/受保护/非零相同模型去重，未把复用计成新试验。非零1.3s重复逐位一致；真实全运动切断J={s['motor_cut']['J']:.9f}，感觉切断J={s['sensory_cut']['J']:.9f}，新分支切断J={s['translator_cut']['J']:.9f}，后者精确回到旧零基线。零重复也精确一致。

## 实际执行与验证

本新实验实际{s['total']}条释放（含2条工程复现，其结果在开发中复用），{s['failed_starts']}次失败释放起点；父代理另行审计的废弃002653Z尝试实际消耗10条基线释放、0个训练候选，不混入本实验结果比较；含废弃尝试的资源累计为{s['total']+10}条释放。1次完整1.4s预热，累计{s['simulated_seconds']:.1f}模拟秒。TRIALS.json逐条记录起点和flags。本目录另有2次登记前工程启动失败（bash-31、bash-32均退出1，均0次物理/拟合）：第一处为桌面符号链接规范路径比较误判，第二处为修正入口静态计数断言发现清理函数也是第4处同类比较。AMENDMENT_01及其SUPPLEMENT保留原源码/失败入口和原因。最终直接执行具体experiment_v2.py，四处路径静态检查和AST检查先通过；没有覆盖任何BEFORE。精确旧zero/unchanged BODY/NATIVE/J复现在拟合前通过。单重进程CPU串行。

独立验证全部{v['run_count']}条的尖峰历史、旧池率、动作/实际力矩、交接连续性、原生评分；重算{v['offline_reference_labels_recomputed']}个实际采用的离线标签，检查所有训练前缀与来源。{v['input_hashes_verified']}项冻结输入哈希均未变。所有释放零MuJoCo警告、辅助与外力为零。验证源码独立实现重构与预测，没有导入实验驱动。

视频 `{O/'REGRESSION_COMPARISON_8x_slow.mp4'}` 由同次保存的qpos/qvel直接重放，完整1.3s算法预热+0.2s撤辅，301帧25fps/8倍慢放；71个mesh不透明，物理数组不变，COM重放误差0，首中末解码通过。当前模型不支持图像输入，未声称完成肉眼视觉检验。

实际清理{len(cleanup['deleted'])}项、{sum(x['bytes'] for x in cleanup['deleted'])}字节：登记失败创建的4个空目录（重登记时重建）与本新目录构造器生成、没有读者的COVERAGE.json。所有失败、候选、来源与验证材料保留。保留文件均在FILE_INVENTORY.json以完整路径注明“核心必需/重要辅助”。旧档案不改字节、不添加文件，全球tracker由父代理处理。

## 解释边界

本实验只检验这组有限罚权能否保住已知较好轨迹并改善其他状态；未达行为门即为失败。不能由拟合/保护取舍推出神经信息唯一缺失、必要恢复动作或意图保持。理想关节力矩plant的生物有效性与真实肌肉仍未建立。

源码完整目录：`{C}`。证据完整目录：`{O}`。预登记PREREGISTRATION.json、冻结输入INPUT_HASHES_BEFORE.json、模型CONFIGS、数据DATA、轨迹RUNS、SUMMARY.json、VERIFICATION.json为核心必需；图像/视频/本报告为重要辅助。
'''
    (O/'REPORT.zh-CN.md').write_text(text)
    before=js(O/'INPUT_HASHES_BEFORE.json');assert all(sha(p)==h for p,h in before.items());dump('FINAL_INPUT_HASH_VERIFICATION.json',dict(checked=len(before),unchanged=True))
    files=sorted([p for p in O.rglob('*') if p.is_file()]+[p for p in C.rglob('*') if p.is_file()])
    inventory=[]
    for p in files:
        auxiliary=p.suffix in ['.png','.mp4'] or p.name in ['report.py','REPORT.zh-CN.md','CLEANUP_LEDGER.json','VIDEO_FRAME_MAP.json','VISUAL_INVENTORY.json']
        inventory.append(dict(path=str(p),importance='重要辅助' if auxiliary else '核心必需',purpose='同轨迹展示/说明/清理记账' if auxiliary else '运行复现、参数数据来源或独立核验证据'))
    inventory.extend([dict(path=str(O/n),importance='核心必需',purpose='全量文件重要程度或哈希封存；哈希清单本身不自哈希') for n in ['FILE_INVENTORY.json','MANIFEST_SHA256.json']])
    dump('FILE_INVENTORY.json',inventory)
    files=sorted([p for p in O.rglob('*') if p.is_file() and p.name!='MANIFEST_SHA256.json']+[p for p in C.rglob('*') if p.is_file()]);dump('MANIFEST_SHA256.json',{str(p):sha(p) for p in files})
    print('REPORT_COMPLETE',O,'manifest',len(files),flush=True)
if __name__=='__main__':main()
