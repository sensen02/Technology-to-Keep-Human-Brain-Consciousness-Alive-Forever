"""Goal-round2 physical-feedback teacher evidence, including all failed variants."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk')
OUT=BASE/'20261107T091243Z'
SPECS=[('原参考，关闭反馈','20261107T091243Z/anchor_feedback_off'),
       ('世界足锚定','20261107T091243Z/anchor_screen'),
       ('有界速度残差','20261107T092322Z/residual_screen'),
       ('接触动力学QP','20261107T092817Z/wbc_screen'),
       ('QP+时钟预测','20261107T094008Z/wbc_lookahead'),
       ('QP+硬接触加速度','20261107T094419Z/hard_contact_screen')]
LABELS=['Reference control','World anchors','Velocity residual','Contact dynamics QP','QP + endpoint prediction','QP + hard contact']


def read(p):return json.loads(p.read_text())


def main():
    rows=[];fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    for i,(label,relative) in enumerate(SPECS):
        folder=BASE/relative;r=read(folder/'RESULT.json');s=r['validation']['release']
        verify=read(folder/'VERIFICATION.json') if (folder/'VERIFICATION.json').exists() else None
        if i==0:
            verify=read(BASE/'20261107T083316Z/neural/teacher_aligned_4s/VERIFICATION.json')
            slip=read(BASE/'20261107T084945Z/BASELINE_0P5MS.json')['physics_replay']['material_contact_velocity']
        else:slip=verify['physics_replay']['material_contact_velocity']
        motion=verify['gait']['locomotion'];ratio=slip['slip_to_forward_speed_ratio']
        row=dict(name=label,folder=str(folder),controller=r['controller'],
            native_D=r['validation']['passed'],D_failed=[k for k,v in r['validation']['checks'].items() if not v],
            foot_load=s['foot_support_mean'],nonfoot_share=s['nonfoot_share_mean'],COM_min=s['com_z_min'],
            steps=verify['gait']['all_legs_stepping'],body_forward_speed=motion['cycle_mean_forward_speed'],
            heading_change=motion['heading_change_rad'],slip_ratios=ratio,
            material_slip_pass=bool(np.asarray(ratio).max()<=.5 and motion['cycle_mean_forward_speed']>.01),
            replay=verify['physics_replay']['passed'],verified_neural=False)
        if 'qp_fallback_count' in r:row['qp_fallbacks']=r['qp_fallback_count'];row['qp_ticks']=r['qp_ticks']
        rows.append(row)
        with np.load(folder/'NATIVE.npz') as z:n=z['rows']
        with np.load(folder/'BODY.npz') as z:b={k:z[k].copy() for k in z.files}
        show=n[::max(1,len(n)//6000)]
        color=plt.cm.tab10(i)
        axes[0].plot(show[:,0],show[:,1:7].sum(1)/24.446354755878005,color=color,lw=.6,alpha=.8,label=LABELS[i])
        axes[1].plot(show[:,0],show[:,10],color=color,lw=.9)
        axes[2].plot(b['com'][:,0],b['com'][:,1],color=color,lw=.8,label=LABELS[i])
    for ax in axes[:2]:ax.axvline(1.,color='k',ls='--',lw=1)
    axes[0].set(ylabel='Foot load / W',title='Continuous algorithm-teacher diagnostics; no neural success')
    axes[0].legend(ncol=2,fontsize=8)
    axes[1].axhline(.8*1.2072528224791905,color='r',ls=':',label='D8 floor')
    axes[1].set(xlabel='Simulation time (s)',ylabel='COM height (native)')
    axes[2].set(xlabel='World x (native)',ylabel='World y (native)',aspect='equal')
    axes[2].legend(ncol=2,fontsize=8)
    fig.savefig(OUT/'ROUND2_COMPARISON.png',dpi=150);plt.close(fig)
    status=dict(importance='核心必需',goal='S5 stable walking after teacher withdrawal',complete=False,
        stage='Goalround2 contact-feedback and contact-dynamics teacher diagnostics',results=rows,
        original_body_preserved='71mesh,nq49nv48nu42neq0,gravity/contact/nativecaps unchanged',
        neural='No new ANN trained thisround: none of these teacherreferences passes allphysical locomotiongates',
        next='Use measured swingclearance diagnosis: earlier/widerlift beforepositivehorizontalreturn; boundedIK/densegeometry/rates first, then actualphysicalslip, only then ANN imitation.',
        upcoming_asset=str(BASE/'20261107T095519Z'),
        body_diagnostics=str(OUT/'ANCHOR_DIAGNOSIS.json'),
        QP_diagnostics=str(BASE/'20261107T092817Z/AUDIT_QP.json'),
        lookahead_diagnostics=str(BASE/'20261107T094008Z/AUDIT_LOOKAHEAD.json'),
        swingclearance_diagnostics=str(BASE/'20261107T094008Z/SWING_CLEARANCE_DIAGNOSIS.json'),
        units='native; absolute SI force/biological validity/original-brain necessity NOT_ESTABLISHED')
    p=OUT/'ROUND2_STATUS.json'
    if p.exists():p.read_text()
    p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    text='''# S5 稳定行走：第二轮接触反馈教师实测

当前仍未实现完整稳定自主行走。本轮没有重训ANN：新的物理教师均未通过所有支撑、步序、实际物质滑移和直行要求。所有连续算法控制结果单列，不能当作神经能力。

原完整身体、重力、接触/摩擦、原关节力矩上限、30rad/s目标速率界与12.5µs物理步长保持。各试验总长4s，其中前1s相同参考预热；之后3s仍由明确算法教师控制。无实际根部轨迹赋值、托举或外力注入。独立物理重放只从原初态积分实际受限关节命令。

## 同一物理判据下的结果

| 条件 | 全D门 | 足载均值/W | 非足反力占比 | 最低COM | 六足持续迈步 | 前向native/s | 实际滑移门 |
|---|---|---:|---:|---:|---|---:|---|
'''
    for r in rows:
        text+=f"| {r['name']} | {r['native_D']} | {r['foot_load']:.4f} | {r['nonfoot_share']:.4f} | {r['COM_min']:.4f} | {r['steps']} | {r['body_forward_speed']:.4f} | {r['material_slip_pass']} |\n"
    text+='''
失败后身体滑行、旋转或恢复部分足载不能算步行；向前累计也不能替代真实接触速度。滑移门保留每足物质点切向速度≤0.5倍身体周期平均前进速度，接触速度统一每0.5ms采样，原生D门仍使用全部12.5µs记录。

## 已核实的机制与限定

1. 首版世界足锚定在接管时改变承载关节目标，1.010s先出现身体高度下降，1.015s之后IK误差才增大。后续16/18次名义支撑进入时足未加载，说明不能只用名义相位重设地面锚点。实际物理状态未被控制器写入，暖态/交接/183感觉输入与时钟均重算一致；不能把失败笼统归咎日志或数值警告。
2. 有界速度残差保留原稳定目标并反馈实际接触物质速度，支撑与直行保持，但滑移变差。其字段TEACHER.ik_residual已声明为[物质速度,关节残差范数]，不是IK残差；三个额外残差数组未直接保存，日志限制如实保留。
3. QP保留完整无驱动根部动力学、原力矩界、保守实际摩擦内接方锥；独立模型重建正确，但数值求解不代表物理可实现。原QP599/600次可接受求解，早期失稳不是fallback。实际5ms末关节力矩与冻结当前状态的预测明显不同，另实际地面反力与规划反力不符。
4. 单独加入5ms关节终态预测的力矩映射，在1.1s部分减小力矩误差，但并未稳定身体；原QP软接触目标本就接近满足，不能误称它完全忽略足接触。硬零材料接触加速度是另一个明确的规划假设，实际MuJoCo柔性接触没有改写；其物理结果仍须单列失败。

这些失败不证明身体或神经路线不可能，也不授权放宽验收、增加力矩、隐藏根部驱动或归功于原脑。

## 下一参考改动来自实际测量

同一已对齐参考的峰值抬脚高度约0.25native，但抬高超过0.03/0.05/0.10的时间只占名义摆动期0.4269/0.3684/0.2807，约0.47的正向足回摆发生在抬高不足0.1时。在实际gain32基线，LH/RH仍承重时发生的正向目标回摆分别占0.4725/0.4017。

因此下一项只试训练参考的时序：先抬脚、在中段前摆、最后落脚，保留同一实际身体/峰值摆幅/名义步序/支撑足几何，先做受原关节界和速度界约束的离线完整模型IK；若足工作空间不允许，直接记录失败，不用动画代替。即便该参考几何通过，仍需真实物理滑移/支撑/直行检查，再训练读取已声明本体感觉的替代ANN，并验证撤去教师至少10s。该改动是数据来源明确的工程重定时，不能冒称原始动物运动或当前已完成。

## 证据位置

'''
    for key,value in [('汇总状态',OUT/'ROUND2_STATUS.json'),('比较图',OUT/'ROUND2_COMPARISON.png'),
        ('首版失败同次视频',OUT/'anchor_screen/WALK_4x_slow.mp4'),
        ('首版独立诊断',OUT/'ANCHOR_DIAGNOSIS.json'),
        ('QP独立模型/实现诊断',BASE/'20261107T092817Z/AUDIT_QP.json'),
        ('终态预测独立诊断',BASE/'20261107T094008Z/AUDIT_LOOKAHEAD.json'),
        ('摆动间隙实际诊断',BASE/'20261107T094008Z/SWING_CLEARANCE_DIAGNOSIS.json')]:
        text+=f'- {key}：`{value}`\n'
    text+='''
父级bash-55/56/58/63/65/67均物理运行exit0但成功标准按原始数组判定；bash-57/59/64/66/68独立重放exit2表示验收失败，不是崩溃。OSQP1.1.3依赖只在092817Z/deps隔离安装，版本和本机扩展hash已记录，原MuJoCo/模型/旧冻结源不改，生成字节码缓存已清理。bash-62失败视频480帧解码/71mesh可见/COM渲染差0，预览和独立诊断图已实际打开。

183维感觉契约已实现并验证不会修改物理状态，只含当前关节/目标/足负载/重力方向/身体速度，不含教师相位、时钟或锚点。它只是后续替代ANN输入准备，本轮没有训练或宣称原连接体脑自主能力。

持续goal保持active，未标记complete/blocked。绝对SI力、生物有效性和原脑必要性NOT_ESTABLISHED。
'''
    p=OUT/'ROUND2_REPORT.zh-CN.md'
    if p.exists():p.read_text()
    p.write_text(text)
    print(json.dumps(dict(report=str(p),complete=False,variants=len(rows)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
