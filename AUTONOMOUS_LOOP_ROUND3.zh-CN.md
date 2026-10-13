# 自主闭环第三轮：精确输出落地与FeCO感觉证据

仍未完成目标，未恢复外部行走许可、CPG或姿态控制。手工实施、测试中的仿真原型。

## 精确输出落实到运行

独立入口 `/run/media/sensen/Data2/cell_wound_prototype/diagnostics/diag_autonomous_sensorimotor.py` 默认改为 exact-candidate，只接唯一精确注释候选，68个未匹配执行器保持零输出；旧模糊映射只保留显式 legacy-fuzzy 对照选项。名称候选仍未完成解剖验证。

0.3秒三分支实测，22个肌肉执行器有候选映射、68个在所有180个命令区间保持零。180个区间均通过脉冲来源检查。完整探索分支173768网络脉冲，最终高度0.75512mm；输入切断0脉冲，运动读出切断177960脉冲，两者最终高度0.73508mm。身体仍塌陷，最大肌肉命令仍饱和，不是自主站立或正确行为。

结果图已核看：`/run/media/sensen/Data2/cell_wound_prototype/outputs/autonomous_sensorimotor_round3/exploratory.png`。

## 新发现：CSV中存在NPZ未保留的感觉亚型

此前只看NPZ大类会漏掉CSV的Sub Class和Alternative Cell Type(s)。完整Sub Class扫描找到前腿FeCO明确亚型注释211个：claw52、hook53、club106；其中200个存在于NPZ且全部保留在12000子网络（52、52、96），另11个缺于NPZ。别名字段独立支持其中110个，是初查子集而不是全部。部分SNpp主类型跨不同亚型，必须逐root查CSV，不能只凭主类型推断。不能再笼统说本地没有亚型信息。也不能仅搜claw就连线，另有claw_tpGRN味觉标签是假阳性。

文献支持：
- claw：股节—胫节位置，存在屈伸方向和个体角度调谐/迟滞差异；不能把所有claw作为绝对角偏差一个池。
- hook：方向性运动、相位性响应；不能未经证据假定电流与角速度线性。
- club：双向运动与振动响应；不是所有细胞同一频率调谐。

来源：
https://pmc.ncbi.nlm.nih.gov/articles/PMC12048489/
https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/

文献的FANC/FAFB不能直接作为BANC个体root ID对应。当前尚缺当地细胞的方向/调谐及模型坐标转换、输入电流标定。5ms接口的采样率200Hz不能直接表达论文中的高频振动；不能把混叠信号当准确感觉。

新增 `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/feco_translation.py` 明确逐亚型需要的证据，缺失时禁止自动授权注流；它是前置检查，不冒充已完成感受器模型。4项测试通过，另有命令审计6项和映射解析5项通过。

下一步可以依据CSV亚型做更窄的网络传递/采样覆盖检查，不再广播整个腿感觉组；但尚不能凭猜测补屈伸偏好后称正确翻译。目标继续保持进行中。
