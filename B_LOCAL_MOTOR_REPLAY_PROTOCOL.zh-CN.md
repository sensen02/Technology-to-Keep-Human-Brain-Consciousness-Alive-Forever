# B门：LF胫节末端通道因果诊断协议（执行前）

手动研究原型，非成品。研究的是保存神经命令的末端传递，不运行新感觉/网络，不称新闭环、单根神经元因果或站立。无支架/锁定/新姿态/肌力参数；保持既有P1 zero-floor+投影就座诊断分支。A门原姿态静态资格失败不阻止短时末端响应诊断，但下沉背景必须说明。

## 已有证据去重

冻结P1 `/run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261013_233237_7be93a1d` 已有正常/重复/重放/全运动切断/输出静默。LF胫节屈池normal窗口counts=[0,0,0,3,2,0,0,0,0,0]、伸池=[0,0,0,1,2,1,1,1,0,1]；正常与重复/重放局部角度精确相同，正常对全motor_cut/静默最大角差1.32073/1.34114rad。仅历史描述，不为生物幅度验收；全输出切断不能定位LF单池因果。

已提取末端通道角/速/命令/活化，不缺这些聚合量；真正缺的是同端点actuator_force/moment及只干预两个预定执行器之一、保持其余命令固定的对照。池按LF tibia flex/ext名称事先选定，非按效果大小挑选。共享/跨肌耦合由manifest实际查明，执行器命令切断不冒充源神经元silence。

## 下一次新运行的唯一范围

1. 复用P1正常10x90 applied controls及原qpos，不重新感觉编码/译码/姿态投影/就座。编译cp canonical确认hash，唯一改动ctrl lower=0以匹配P1声明分支，明确记录原生/派生range。fresh MjData初始qvel/act0，无外载。不得用静态LP候选命令。
2. 四个50ms新body-only案例：完整命令baseline、精确重复、LF tibia flex actuator命令全程置0、LF tibia extensor actuator命令全程置0。切断只发生在预定执行器命令层，不清act，保持其他89命令逐位相同。前两个control本来0的阶段自然一致，不制造更早刺激。
3. 时间步采用原P1 body .1ms，命令保持5ms，禁止当前轮换RK4/改dt/tau；这仅历史分支诊断，不证明时钟收敛。先核原P1源码snapshot与当前episode操作是否能同identity重现；若当前工具/实现无法精确重现baseline，停止干预解释，不修旧哈希、不改容差称重放成功。
4. 每个500步mj_step后mj_forward对齐保存所有qpos/qvel/act/ctrl、90force、完整moment、qfrc_actuator/passive/bias/constraint；记录ctrl施加时间及5ms端点与旧P1比较。forward后的warmstart/缓存若影响原P1轨迹，则该扩展采样不是精确同实现重放，必须保留差异并停止，不能冒充复现。
5. 必须先baseline/重复完成及原P1关键状态对齐才运行两cut。工程阈值：重复全字段精确，原P1命令与5ms状态先以array_equal检验并报告；不相同即NOT_REPRODUCED（另项目级说明，不事后加容差放行）。force恒等式moment.T@force与qfrc_actuator rtol1e-10/atol1e-10，有限/无警告/输入hash不变。
6. 因果输出：各case预定关节LF tibia角/速差、目标与全体actuator力矩、moment改变；同端点总广义力分解。不同轨迹上的力差混合几何/状态效应，不能当只由目标肌直接力贡献；首次命令/act/force差按记录网格声明，非生物延迟。可为0，失败/无效应不改输入增益或加电流。

## 资源/停止/交付

最多2000step+2004forward，顺序案例磁盘写/每case约25MB，2分钟外部限时，无网络/安装/服务/全量回归。执行前读当前所需helpers并冻结源码/71meshes/pose/commands；新目录保留raw NPZ/source snapshot/report和图，图须实际打开。异常/基准不能重现先写FAIL停止，不再自动跑干预/整网。

本轮仅读P1归档提取并画局部图，**此协议四案例尚未执行**。不能把这份协议称B门通过或稳定动作完成。下一“继续”仅先实现baseline资格门，按结果决定是否继续两cut，避免重做全量P1。
