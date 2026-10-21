# S2/P2 接触力差峰离线审计预声明

本协议在计算新峰值指标前写入。仅使用 `/home/sensen/Desktop/engine修复验收/S2-contact-clock-protocol-complete-20261014-v1` 的 report.json、8个NPZ和8个contacts.json；规范模型仅 cp.load_model() 读取geom/body名称，不创建数据或步进。手动研究原型，非成品。

## 指标与选择规则

- 各共同端点、排序几何对 p，使用保存的世界力（作用于排序第二geom）；缺失pair在差运算中补零，同时独立记录presence，不能把缺失当真实零接触。令 delta=Fcoarse-Ffine，欧氏差 Dp=||delta||2；分别保留向量、两侧范数。
- 范数相对差 Rp=Dp/max(||Fcoarse||2,||Ffine||2)。分母为零明确 undefined，JSON null，绝不加epsilon或默认为0。
- 每端点全局pairmax=max_p Dp；另算 sum-vector差 ||sum_p Fcoarse-sum_p Ffine||2，和 sum_p Dp。不同几何对方向可相消，且各pair第二geom未必同一物体，因此sum-vector不是系统净外力或支撑力。不能用它代替pairmax。
- 对4个branch/case逐一在固定窗口[1.5,2.3]ms、[7.7,8.7]ms取pairmax最大端点（并列取最早），另给全10ms全局最大端点。窗口仅定位用户已指出约1.9/8.2ms事件，不是阈值。列top5几何对，以差值降序、pair键打破并列。
- 每个所选端点及前后各2个共同端点：保存时间、ncon、pair集合/两侧presence、top5 pair力向量与范数及差；保存全qvel、qfrc_constraint、对应逐DOF coarse-fine差（不同DOF有平移/旋转单位，不混合欧氏范数），另列绝对差最大的5 DOF及原报告标签单位。
- 比较ledger长度、NPZ共同时间/expected_time_s、ncon数组长度，复算原report force_difference_series；保存输入文件SHA256前后以及原source_hashes路径当前哈希核查。独立验证使用另一段不导入审计指标实现的算术，对所选极值、全局极值、账本长度及输入前后哈希复核。

## 证据边界

仅作事件定位和算术审计；ncon/presence稳定不代表接触点、约束活跃集或解稳定。保存账本没有原始contact frame/local force、接触位置/距离/约束内部解，不能从同时变化推断物理原因。若不能证据归因，物理原因明确未知。不增加通过/收敛/生物阈值，力仍为model force units，广义约束力按原DOF单位；不声称身体收敛、站立或生物有效。保持源档案和原dirty tree，不改生产、不服务/硬件/远程/依赖。
