# 运行清单

- P1/P3，S1–S5 身体基础及神经接线。身份固定、门槛和预算见 `PROTOCOL.zh-CN.md`。正式两个入口无需现有训练栈，旧环境仅提供 Python 3.12.14、MuJoCo 3.9.0、SciPy 1.18.1、NumPy 2.5.3。固定 OMP 和 BLAS 为一线程。
- 正式 `run_body.py`、`audit_wiring.py`、`run_neural.py`、`audit_static.py`、二次身体／神经／静力、`verify_and_plot.py` 均正常退出。快照 `build_neural_tier(TierConfig())` 独立重建查询退出 0，指纹一致。首次元数据只读探查按整数与字符 ID 比较得到空，立即按精确字符串重查 208 个旧根，未用于生成产物。
- 身体正式 8,000 原生步、神经 1,920 拍、静力 6 次 LP。原生腿肌单路探针合计 36,000 次激活动力学微步，姿态固定。复验重复此预算。92 数组与身体、神经、静力三份 JSON 逐值或字节相同。实际打开同次 `BODY_SUPPORT.png`。输入与输出 SHA 校验见 `ASSETS.sha256`、`AUDIT_INPUTS.sha256` 及交付校验。
- 实际 4/4 起点和控制下限搭配均未过站稳门。未执行身体神经闭环、正式行走门、找姿搜索或训练。PPO/GAE/优化器不适用。没有引入神经译码，也没有修复或重标定旧肌肉参数。
