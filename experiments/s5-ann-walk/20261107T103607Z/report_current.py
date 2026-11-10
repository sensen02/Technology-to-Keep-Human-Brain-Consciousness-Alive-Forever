"""Current neural duration evidence and unchanged failed-slip boundary."""
import json
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk')
OUT=BASE/'20261107T103607Z'
SPECS=[('先抬脚教师 gain32','20261107T101114Z/neural/teacher_lift_4s'),
       ('先抬脚教师 gain128','20261107T101753Z/teacher_gain128_4s'),
       ('替代ANN短窗','20261107T102842Z/ann_screen'),
       ('替代ANN长窗','20261107T103607Z/ann_12s'),
       ('切ANN输出/保持','20261107T103607Z/neural_cut_hold_4s'),
       ('切关节力矩','20261107T103607Z/motor_cut_4s'),
       ('条件积分教师','20261107T103309Z/teacher_conditional_4s')]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text())


def main():
    rows=[];hashes={}
    for name,rel in SPECS:
        f=BASE/rel;r=read(f/'RESULT.json');v=read(f/'VERIFICATION.json')
        g=v['gait']['locomotion'];p=v['physics_replay'];support=v['support']['release']
        material=p.get('material_contact_velocity',{})
        rows.append(dict(name=name,folder=str(f),controller=r['controller'],release_seconds=r['release_seconds'],
            native_D=v['support']['passed'],D_failed=[k for k,x in v['support']['checks'].items() if not x],
            support=support,real_steps=v['gait']['all_legs_stepping'],gait=g,
            material_slip=v['material_slip_passed'],slip_ratio=material.get('slip_to_forward_speed_ratio'),
            slip_speed=material.get('force_weighted_mean_tangential_speed'),
            replay_performed=p['performed'],replay_pass=p.get('passed'),
            policy_reconstruction=v['policy']['passed'],policy_max_error=v['policy']['policy_rate_max_error'],
            final_goal=False))
        hashes[str(f/'RESULT.json')]=sha(f/'RESULT.json');hashes[str(f/'VERIFICATION.json')]=sha(f/'VERIFICATION.json')
    ann=rows[3];teacher=rows[1]
    cuts=rows[4:6]
    status=dict(importance='核心必需',complete=False,stage='Goalround3 broad-lift path and replacement-ANN sustainedsupport',
        outcome='ReplacementANN maintains fullnativeD/sixlegsteps/nearstraightbodytravel for11s after1swarmup; actualmaterialslip stillfails allsix legs. No finalstablewalkingpass.',
        controller='LearnedreplacementmotorANN, no originalconnectome in thisarm; after1s no teacher/phaseclock/lookup, onlycausalown-targethistory andfixedweights ->originalboundedactuator',
        original_body='full71mesh49/48/42neq0/free root/originalgravity/contact/friction/nativecaps; gain128 finitePD',
        weights=str(BASE/'20261107T102842Z/MODEL.npz'),weights_sha256=sha(BASE/'20261107T102842Z/MODEL.npz'),
        results=rows,
        neural_output_necessity='ANN release has repeatedsixlegsteps; replacement-outputcut holds target andretainsstandingbutstopsrepeatedsteps. Motorcutcollapses. Notoriginalconnectomeablation.',
        geometry=str(BASE/'20261107T101114Z/INDEPENDENT_VERIFICATION.json'),
        saturation_diagnosis=str(BASE/'20261107T101753Z/TRACKING_LIMITS.json'),
        conditional_diagnostic='Explicitlocalconditionalintegration arm didnotresolveperlegslip, andisnotusedin main11sANN. Finalfrozenverifierrecheck separate104555methoddirectory; V1reports/provenancepreserved.',
        next='Do notrelaxslipbound orraiseRHtorquecaps. Diagnose RHstanceposture/leverage usingmeasuredcontactloads; neuronal imitationmayneedvelocity-sensitiveerrorloss. Anynewcandidate independentlytestedwithsamegates.',
        source_hashes=hashes,
        units='native; SIforce/originalbrainnecessity/biologicalvalidity NOT_ESTABLISHED')
    fig,axes=plt.subplots(3,1,figsize=(12,11),constrained_layout=True)
    native=np.load(OUT/'ann_12s/NATIVE.npz')['rows'];show=native[::max(1,len(native)//10000)]
    axes[0].plot(show[:,0],show[:,1:7].sum(1)/24.446354755878005,lw=.65,label='ANN foot load / W')
    axes[0].plot(show[:,0],show[:,7]/24.446354755878005,lw=.7,label='non-foot / W')
    axes[0].axvline(1.,color='k',ls='--');axes[0].set(ylabel='Support',xlabel='Seconds',title='Replacement ANN only after1s; native support passes, material slip FAILS')
    axes[0].legend()
    for tag,label in [('ann_12s','ANN11s'),('neural_cut_hold_4s','neural-output cut'),('motor_cut_4s','motor cut')]:
        z=np.load(OUT/tag/'BODY.npz');mask=z['time']>=1.
        axes[1].plot(z['com'][mask,0],z['com'][mask,1],lw=.9,label=label)
    axes[1].set(xlabel='World x (native)',ylabel='World y (native)',aspect='equal');axes[1].legend()
    x=np.arange(6)
    axes[2].bar(x-.18,teacher['slip_ratio'],width=.36,label='Teachergain128')
    axes[2].bar(x+.18,ann['slip_ratio'],width=.36,label='ReplacementANN11s')
    axes[2].axhline(.5,color='r',ls='--',label='unchanged slip bound0.5')
    axes[2].set(xticks=x,xticklabels=['LF','LM','LH','RF','RM','RH'],ylabel='Material speed / bodyforward speed')
    axes[2].legend();fig.savefig(OUT/'CURRENT_COMPARISON.png',dpi=150);plt.close(fig)
    p=OUT/'CURRENT_STATUS.json'
    if p.exists():p.read_text()
    p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    a=ann['support'];g=ann['gait']
    text=f'''# S5 当前结果：替代ANN撤教师后11秒持续承重迈步，滑移验收未通过

**稳定低滑移自主行走目标仍未完成。** 新替代ANN在1s物理教师预热之后连续接管11s，原生全部支撑门、六足持续卸载/着地、前进和直行筛查通过；实际足地材料点滑移仍失败，不能把支撑和位移通过当成全目标完成。

这一次比旧3Hz短时/转圈结果有明确进展：完整身体保持高度并近似直线推进，不再接管约1.8s后倒地；但仍有明显足地拖滑。

## 同次神经接管实测

- 0–1s为算法教师物理预热，1–12s为替代ANN高层输出加原有限增量PD。原连接体脑没有在该臂运行，不能把此能力归功于原脑。
- 仅固定网络权重和40帧真实自身目标历史参与部署；预热全程填充历史，接管不重置物理。部署后没有步态表、相位时钟、教师调用或根部位移命令。
- 全身71mesh、49qpos/48qvel/42actuator、自由root、neq0；原重力[0,0,-9801]、接触/摩擦、原力矩上限、30rad/s目标速率及12.5µs物理步长保持。局部有限PD的gain128为明确工程参数，执行器自身保持必须单列。
- 11s撤教师窗足载均值 **{a['foot_support_mean']:.6f} W**，10分位{a['foot_support_p10']:.6f} W，非足反力占比{a['nonfoot_share_mean']:.6f}、自由落体0，COM最低{a['com_z_min']:.6f}，原D1–D8/D4b全部通过，未删失败帧或更换评分窗。
- 周期平均身体前向累计 **{g['cycle_mean_body_forward_travel']:.6f} native**，速度{g['cycle_mean_forward_speed']:.6f} native/s；真实原始root/COM世界x净位移约{g['raw_net_xy'][0]:.6f}，不可把身体周期摆动路径当额外行走距离。航向总变化{g['heading_change_rad']:.6f}rad，直线性{g['straightness']:.6f}，直行工程门通过。
- 独立重放从原始初态重新积分，仅赋原界关节控制。身体位置、速度、COM、足位置、力矩、抽样接触和神经输出/目标积分/历史交接最大误差均0.0。
- 每0.5ms抽样实际接触材料点切向速度，并用接触点Jacobian×qvel交叉核对。六足速度/前进速度分别为 **{', '.join(f'{r:.3f}' for r in ann['slip_ratio'])}**；全部高于0.5门，**滑移失败**。此抽样不是连续12.5µs接触速度记录。

## 必要对照与结论边界

| 条件 | 撤教师后时长 | 原生全部D门 | 六足持续迈步 | 足载/W | 非足反力占比 |
|---|---:|---|---|---:|---:|
'''
    for r in rows[3:6]:
        s=r['support'];text+=f"| {r['name']} | {r['release_seconds']:.1f}s | {r['native_D']} | {r['real_steps']} | {s['foot_support_mean']:.5f} | {s['nonfoot_share_mean']:.5f} |\n"
    text+='''
切ANN输出为零增量，原目标记忆和有限PD保留；身体继续站着但不再持续迈步。真正断关节力矩后身体塌陷。故不能把站立保持归功于持续神经输出，但持续迈步依赖当前替代网络输出。这个“切脑”只指替代网络输出切断，不是原连接体脑必要性对照。

ANN长窗已完整独立物理重放；两个切断臂做了原生数组、命令、输出及运动重算，没有额外完整物理重放，验证范围明确单列。尚未验证不同地形/外力扰动/广泛初态泛化。

## 本轮参考和学习的实际改变

旧参考峰值抬脚虽高，但大部分摆动期足部低、回摆时仍加载。新参考改为先抬脚、中段回摆、最后下降；摆动期脚方向自由，原支撑几何不变，并保持全部真实跗节碰撞几何平台期至少0.1native间隙。逐帧IK会在落地前产生冗余关节返回跳变，因此改用整段约束轨迹IK。所有角界/速率界直接进求解，最终原生30rad/s不变。

101114候选独立4097点FK/全部真实足底重算通过：位置误差.002055、平台最低间隙.248768、最大精确PCHIP速度20.06446rad/s，原支撑knots逐位一致。仅两次内部QP到达迭代上限的候选步，在实际原边界/信赖域和非线性下降检查后接受；物理判据没有放宽。先前失败脚本/证据保留，没有拿未通过速率的曲线做物理演示。

新gain32教师仍有滑移，但三个足已通过；相位分解显示摆动期法向冲量占比由约0.10降到0.0214，提前抬脚确实减少拖地，余下滑移主要在支撑期。gain128使部分足继续改善，但RH在中段支撑长时间触及原力矩上限，不能只增加刚度或提高力矩上限。

在这一物理改善基础上，明确预登记并行ANN模仿筛查（没有谎称教师全部通过）：同一因果历史MLP训练3000步，多步自预测损失，离线20s自主目标回路保持振幅约.999、约19.94周期。物理接管保持稳定，但神经近似比教师引入更多滑移，说明目标角拟合误差小仍不足以完成接触质量验收。

## 剩余问题与下一最小动作

- RH中段支撑的Coxa_pitch和Trochanter_yaw持续触及原生力矩界；后者原PD未限幅需求可达上限约25.5倍，额外力矩实际没有施加。需检查源支撑姿态/力臂/载荷分配，不能归咎落脚瞬间或只增加gain。
- 条件积分防止已饱和方向继续累积目标，明确是新局部执行器接口测试，原力矩/速率和零输出保持不变。教师实测与独立重放未解决滑移，未被用于主11sANN证据。一次子代理READY后又加验证检查造成源记录竞争，V1报告/出处完整保留，104555冻结方法目录已对相同物理输入重新重放，不以旧hash冒充新源码。
- 接下来只做来源足姿态保持下的力臂/关节冗余诊断（0物理），判断是否可在原力矩界内分配载荷；有证据才准备新参考。原神经轨迹依然保留作稳定基线，所有候选必须再过同一实际材料滑移门。

## 视频与主要证据

'''
    for label,path in [('当前ANN视频',OUT/'ann_12s/WALK_4x_slow.mp4'),
                       ('不同步态相位拼版',OUT/'ann_12s/PHASE_PREVIEW.png'),
                       ('全时段预览',OUT/'ann_12s/PREVIEW.png'),
                       ('独立物理/神经输出复核',OUT/'ann_12s/VERIFICATION.json'),
                       ('当前对照图',OUT/'CURRENT_COMPARISON.png'),('结构化状态',OUT/'CURRENT_STATUS.json')]:
        text+=f'- {label}：`{path}`\n'
    text+='''
视频4倍慢放，总48s对应12s物理，前1s教师与后11s替代ANN有明确标记，标题明确滑移未通过。1440帧解码、71mesh不透明、渲染COM差0，已实际打开全时段及八个不同步态相位图；未逐帧目视全部1440帧。

源代码在 `/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/` 下对应时间戳。当前持续goal保持active，未标记完成。绝对SI力、生物有效性及原脑必要性：NOT_ESTABLISHED。
'''
    p=OUT/'CURRENT_REPORT.zh-CN.md'
    if p.exists():p.read_text()
    p.write_text(text)
    print(json.dumps(dict(report=str(p),complete=False,ANN_seconds=ann['release_seconds'],native_support_pass=ann['native_D'],slip_pass=ann['material_slip']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
