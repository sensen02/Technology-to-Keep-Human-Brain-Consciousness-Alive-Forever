# B门离线解释决策：末端因果可重现，活化数值资格未过

手动研究原型，非成品。原有冻结命令重放的LF屈/伸执行器对关节坐标产生相反力矩；首次命令20ms、首次保存act/force/torque差20.1ms、qpos/qvel20.2ms，约束力在状态尚同的20.1ms已响应；不能把肌肉差直接当净运动力。离线分解误差<2e-15，图已实际查看，无新仿真。

新发现必须阻止完整B门放行：baseline内部act最大2.0，965个采样标量>1；cut flex/ext963/964。compiled dynprm .0001/.0004s、actlimited=false，身体步.0001s。三case首个act>1发生于10.1ms，早于20ms的目标切断；不是切断才引发。保存参数身份见 `/home/sensen/Desktop/engine修复验收/B-local-force-explanation-20261014-v1/activation-parameter-provenance.json`。命令0–1、重复精确不等于内部活化归一化/数值资格通过。未单凭这些量认定唯一根因、未改tau/限制/步长，原历史结果保留为原离散系统证据。

完整报告 `/home/sensen/Desktop/engine修复验收/B-local-force-explanation-20261014-v1/力矩解释与资格限制.zh-CN.md`；report.json/terms.npz/force-terms.png同目录。实现 `/run/media/sensen/Data2/cell_wound_prototype/tools_b_local_force_explanation.py`。独立复核另附。

当前自动目标只完成已有数据解释并停止，不扩成无限研究。下一最小动作：先去重已有S1/S2的激活动力学样例，然后预声明固定命令、独立肌肉激活/步长资格诊断，分别记录原规范/zero-floor/任何新派生，禁止不经协议静默clip。站立需另有来源初态/身体修订选择，不搜索动作或宣称B门/站立通过。
