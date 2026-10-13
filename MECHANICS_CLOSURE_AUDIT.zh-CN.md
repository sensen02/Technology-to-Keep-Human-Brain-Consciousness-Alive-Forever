# 独立力学闭合证据审计

这是手工实施和测试的研究原型，不是成品、神经控制器或稳定站立验收。

## 纠正内容

- 编译模型90个执行器全部按名称及muscle gain/dynamics核对；没有adhesion执行器。历史末两项实际是右后腿胫节屈/伸肌，不能减2。
- 正确有界增量：`-a <= delta <= 1-a`，更新`a+alpha*delta`并用实际forward残差接受步长；上界采用真实分母的后向差分。
- 目标为42个具名铰链的广义惯性力矩L2，额外报告L∞，非全局最优。根部力、力矩、线/角加速度及逐关节加速度分别记录。
- 力矩除体重有长度量纲，不是无量纲。只报告model力/力矩；canonical质量文档与换算常数存在矛盾，不宣称SI质量或真实果蝇尺度。
- 初态激活可为0，但模型控制下限0.0001；短时自由积分保留原生激活动力学，报告排除肌激活泄漏，不伪称持续严格置零。

## 小规模实测

| 条件 | 自由肌数 | 腿力矩L2(model) | 腿力矩L∞(model) | 关节加速度L∞(rad/s²) | 高度跌破90%时间(s) |
|---|---:|---:|---:|---:|---:|
| ALL90_BY_IDENTITY | 90 | 0.43646442 | 0.15734754 | 338.00331 | 0.024899999999999905 |
| COVERED74_ONLY | 74 | 0.71475887 | 0.33775004 | 726.26134 | 0.025099999999999904 |
| UNCOVERED16_ONLY | 16 | 3.8710298 | 1.9664712 | 3941.6972 | 0.014699999999999967 |

覆盖/全部实测L2比=1.6376108。仅为当前求解器、初姿和接触设置结果，不能推导未覆盖肌不承重。方向导数相对误差=3.18281e-12；负增量回归、上界差分、接受步单调性、输入hash不变均通过。

## 运行命令

```bash
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_mechanics_audit.py --rounds 4 --seeds 0 0.5 1 --seconds 0.05
```

## 独立交付与校验

- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_mechanics_audit.py`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/mechanics_closure_audit.json`
- `/run/media/sensen/Data2/cell_wound_prototype/MECHANICS_CLOSURE_AUDIT.zh-CN.md`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/mechanics_closure_audit.sha256`

契约JSON SHA256: `ddabf5c00aee46420d24801a1f96d72cd4630dd1063896d6a35f4d51ed5f87a8`

受保护输入运行前后hash完全一致：

- `/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` — `65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`
- `/run/media/sensen/Data2/cell_wound_prototype/engine/canonical_physics.py` — `9ffb0e7acb52442d2a73ba970884908a452306dffeeea2ad6ea22eb98b1172bf`
- `/run/media/sensen/Data2/cell_wound_prototype/run_fix_round39.py` — `b743cab54574c48ea4258bc67e6610dfa65d7cf692e9a63ceb885460097d2cba`
- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_rebuild_15_zeroing_cost.py` — `8891a59fd64e08a80eef763f669a7ce8cba7467428b8f88af54acdcd121a4c01`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/neural_rebuild_11_mn_pools.json` — `3244c8c1470ad3009dd3e12fc64b11f97d1672b8bda811f9ceacf0221e1dd94c`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/neural_rebuild_15_zeroing_cost.json` — `dc9e4e4c99880abf1b1f1c2cbed1ce5e167ccbb4b6c4de7370b63b5e48d94e9a`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/stance_margin_round37/stance_for_reflex.json` — `068a0176781e9116c77b0a09b5b8c40951bb320a835ffdaa0c52f11a2faca829`

契约中的PASS只说明审计执行成功，不表示力学闭合；历史代码/日志/XML未修改。未使用pandas，不混用虚拟环境。脚本拒绝覆盖已有同名产物；复跑须指定新的绝对输出目录与日志路径。
