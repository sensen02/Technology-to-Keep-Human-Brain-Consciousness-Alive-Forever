# 第七轮：44个有来源支持的输出候选接入身体

目标未完成，飞行暂缓。手工仿真原型。没有新增导航、步态、姿态或动作时间表。

## 本轮唯一主要改变

独立入口默认motor-map改为anatomy-candidate：读取已核验肌肉资产/神经走行/腱路线候选清单，逐root重新解析当前网络索引，并核对来源哈希。明确46个拒绝/未解析执行器保持零，不拿模糊最高分补齐；一个未采样MN仍列为missing，不用-1替代。

这是显式探索模式下的来源支持候选，不是忽略runtime_authorization=false去声称生物学验证。严格模式依然拒绝未标定映射。原exact-candidate和legacy-fuzzy只保留可指定的对照模式。

感觉与神经参数保持上轮设置：96个明确前腿club细胞接收实际股胫角速度幅值的有界近似电流；初始静止输入0；没有目标或姿态参考；该近似仍未标定。

## 三组0.3秒结果

| 条件 | 全网络脉冲 | 最后胸节高度 |
|---|---:|---:|
| 44输出候选+club反馈 | 125018 | 0.73531mm |
| 感觉输入切断 | 0 | 0.73508mm |
| 运动读出切断 | 125338 | 0.73508mm |

全部180个区间通过运动命令来源核查，46个未匹配输出始终为零。完整条件中，44个映射输出×60区间有42.765%的控制达到上限1。身体仍塌陷；扩展输出数量并没有解决协调，不能把接了更多输出说成效果更好。

注意最后高度相似不证明相同吸引子或平衡；完整与切断条件中途物理轨迹和感觉输入明显不同。图已打开核看。

## 接下来

需要分离网络响应饱和、MN群体到肌肉的招募转换以及未接的方向性本体反馈问题。不能仅按是否站起调整感觉增益，也不能用姿态控制器补偿。下一步做不依赖身体目标的固定神经输入响应/输出灵敏度检查，保留标定不足结论。

代码：`/run/media/sensen/Data2/cell_wound_prototype/diagnostics/diag_autonomous_sensorimotor.py`
结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/autonomous_sensorimotor_round7/`
桌面：`/home/sensen/Desktop/cell_wound_prototype/autonomous_sensorimotor_round7/`
