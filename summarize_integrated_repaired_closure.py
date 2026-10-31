"""Saved-only outcome and artifact sealing for the one integrated release."""
from pathlib import Path
import json,hashlib
import numpy as np
from run_one_second_motor_pair import dump,sha
O=Path('/home/sensen/Desktop/engine修复验收/s4-integrated-repaired-closure/20261028T032000Z')
D=Path('/home/sensen/Desktop/S4_INTEGRATED_REPAIRS_FULLBODY_SLOW.md')
def main():
 z=np.load(O/'BODY.npz');t=z['time'];last=t>t[-1]-.2+1e-9;non=z['nonfoot_floor_count'];idx=np.flatnonzero(non>0);pre=json.loads((O/'STATIC_PREFLIGHT.json').read_text());weight=pre['weight_native'];fz=z['ground_force_foot_world'][:,:,2].sum(1);nf=z['ground_force_nonfoot_world'][:,2];h=z['com_position'][:,2];gate=bool(not non[last].any() and h[last].min()>=.8*h[0] and fz[last].mean()/weight>=.8)
 result=dict(physical_duration_s=float(t[-1]),samples=len(t),first_nonfoot_s=None if not len(idx) else float(t[idx[0]]),initial_com_height=float(h[0]),final_com_height=float(h[-1]),last200_mean_com_height=float(h[last].mean()),last200_nonfoot_fraction=float((non[last]>0).mean()),last200_mean_foot_fz=float(fz[last].mean()),last200_mean_nonfoot_fz=float(nf[last].mean()),last200_mean_foot_load_ratio=float(fz[last].mean()/weight),last200_foot_force_fraction=float(fz[last].mean()/(fz[last].mean()+nf[last].mean())),last200_foot_contact_legs_minmax=[int(z['foot_contact'][last].sum(1).min()),int(z['foot_contact'][last].sum(1).max())],support_screen_pass=gate,walking_certified=False,warning_count=int(z['warnings'].sum()),finite=bool(all(np.isfinite(z[k]).all() for k in ['qpos','qvel','activation','command','muscle_force'])),external_force_max=float(abs(z['external_force']).max()))
 dump(O/'OUTCOME.json',result)
 ex=json.loads((O/'EXECUTION.json').read_text());vid=json.loads((O/'VIDEO_VERIFICATION.json').read_text())
 text=f'''# 综合修复配置：一次真实闭环释放与全身慢放

视频：`/home/sensen/Desktop/S4_INTEGRATED_REPAIRS_FULLBODY_SLOW.mp4`

## 实际采用与没有采用
- 当前canonical65d为物理基底，保留全身71mesh、原质量/惯量、全部231附着点、90肌肉原增益/偏置/激活时间常数和接触参数。
- 接入修后S3V2站姿、对应9项关节范围、42项与本次站姿对齐的springref、左右折翅body变换。
- 90条肌肉的lengthrange按新姿态/原关键帧腱长比整体缩放，保留原关键帧的每肌肉带内分数与相对带宽。这是明确工程标定，不是生理认证；并非把旧10倍肌力或185点NOT_ACCEPTED路径假设混入。
- 初态载荷静态±0.1深度括区未覆盖体重目标，故不加深、不扩大搜索，使用原保存站姿。根高度位移={pre['initial_rootz_shift']:.8g}，初足反力/重量={pre['initial_foot_load_ratio']:.6g}。六足均接触、无非足承载不等于初态动态平衡。
- 运动池走可信准入保护；seed0、历史静息神经连接、真实身体反馈、原5ms感觉/命令与生产解码。没有CPG/PD/根固定/姿态伺服/外力/命令重放/肌肉预加载。

## 一次实际运行
身体步长12.5us，神经步500us，实际时长{t[-1]:.6f}s。实际计数 `{ex['counts']}`，退出{ex['exit_code']}。

结果：**{'通过预声明末200ms足撑筛查，但仍非行走认证' if gate else '未通过末200ms足撑筛查，轨迹下沉并侧翻倒地，不能称站立或行走'}**。
- 首次非足触地：{result['first_nonfoot_s']}秒。
- COM初高/末高：{h[0]:.6f} / {h[-1]:.6f}（模型单位）。
- 末200ms非足接触比例：{result['last200_nonfoot_fraction']:.6f}；平均足/非足竖直反力：{result['last200_mean_foot_fz']:.6f} / {result['last200_mean_nonfoot_fz']:.6f}。
- 末200ms足反力/重量：{result['last200_mean_foot_load_ratio']:.6f}，接触腿数范围{result['last200_foot_contact_legs_minmax']}。
- 警告{result['warning_count']}，全部主要状态有限={result['finite']}，外力最大值{result['external_force_max']}。
这些是同次综合配置的一次工程试验，不是逐因素因果归因或重复性证明。绝对生理力标定仍NOT_ESTABLISHED。本次有真实神经—肌肉—感觉闭环；视频声明没有外置支撑或脚本控制，不表示没有神经控制。

## 视频
全部帧读取本次保存qpos/速度/激活/命令，固定相机，全身头胸腹六腿双翅可见，71mesh不透明、隐藏肌腱调试线不遮外形，模型本体未删部件。20倍慢放，{vid['frames']}帧/30fps；原始{t[-1]:.3f}s显示{float(vid['probe']['format']['duration']):.3f}s；无插值/脚本动作/补尾定格。已ffprobe、完整解码及代表帧实际查看。

## 失败记录与边界
030000Z首次构建错误地把FlyGym缓存网格当本地路径，派生编译失败：1成功参考模型/另1失败编译、1forward、0身体/神经步。031000Z静态括区未达体重，被过强断言提前终止：2模型/4forward、0身体/神经步。均保留原目录，不改写；032000Z显式采用原站姿回退规则，唯一实际rollout。没有第二条行为轨迹或参数搜索。首个保存态渲染因NPZ重复解压过慢被主动停止（bash-284，SIGTERM，部分渲染forward次数未完整落盘，不能称为600）；部分视频已记录哈希和大小后实际删除，缓存相同保存数组后重渲染一次完成（bash-285）。这不是第二次物理或神经运行。父独立XML核对首次遗漏freejoint索引造成断言失败，修正后通过，失败已在PARENT_MODEL_BINDING_CHECK记录。

## 保留文件重要程度
- 核心必需：`/home/sensen/Desktop/S4_INTEGRATED_REPAIRS_FULLBODY_SLOW.mp4`，本次真实轨迹的全身慢放。
- 核心必需：`{D}`，说明与失败披露；`{O/'INTEGRATED_MODEL.xml'}`、`{O/'INITIAL.npz'}`、`{O/'BODY.npz'}`、`{O/'INTERVALS.npz'}`、`{O/'RECORDS.json'}`、`{O/'CONTACTS.json'}`，模型身份与同步真实轨迹。
- 核心必需：`{O/'MODEL_CHANGES.json'}`、`{O/'STATIC_PREFLIGHT.json'}`、`{O/'OUTCOME.json'}`，逐项变更、初态及行为结果。
- 重要辅助：`/run/media/sensen/Data2/cell_wound_prototype/run_integrated_repaired_closure.py`、`/run/media/sensen/Data2/cell_wound_prototype/render_integrated_repaired_closure.py`、`/run/media/sensen/Data2/cell_wound_prototype/summarize_integrated_repaired_closure.py`，运行/渲染/摘要入口；新时间戳内代码副本与预登记/计数/哈希/逐帧receipt及独立校验记录。
- 重要辅助：030/031失败构建产物，专用于追溯，不是可用默认模型。

清理：已实际删除被替代的首次不完整视频（哈希与大小见RENDER_RETRY）；030/031失败证据保留供复核，未删除历史封存、用户资产或原模型。没有为了“更新”覆盖canonical或旧站姿。

重要辅助的已有入口修改：`/run/media/sensen/Data2/cell_wound_prototype/run_dynamic_neural_gain_calibration.py` 已由父代理改为调用可信池准入保护；旧新哈希与可信/候选输入测试见 `{O/'GUARD_PATCH_LEDGER.json'}`。未重跑旧12候选。重要辅助的独立复核入口：`/run/media/sensen/Data2/cell_wound_prototype/verify_integrated_repaired_saved.py`。
'''
 D.write_text(text);(O/'REPORT.zh-CN.md').write_text(text);(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
