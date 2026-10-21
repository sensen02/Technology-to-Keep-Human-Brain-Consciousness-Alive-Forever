# S2 RK4身体接触时钟比较（执行前声明）

同前协议目标0、pulse/decay、native/zero floor、投影就座pose、10ms。仅RK4派生25/12.5us，两时钟每组合独立fresh共8case，无生产改变。每25us共同端点采样，不比较半步孤立接触索引。

逐DOF用mj_differentiatePos(dt=1)得到姿态差，free前三/slide单位model_length（当前mm比例），free旋转/ball/hinge单位rad，逐标签报告，不把qpos quaternion原值差当转角。qvel差亦按DOF平移model_length/s和角rad/s。

接触按排序geom pair汇总世界坐标力：mj_contactForce局部前三经contact.frame.T变换，对排序pair第2geom正向，反序取负。多个接触点相加，缺失pair按0参与最大差，但计pair presence变化；不当同一个接触点匹配。只比较合力，不声称力矩相同或机械功一致。模型力单位未标定，不标N。附qfrc_constraint逐DOF最大差及ncon，共端点固定控/活化参考/finite/warnings核查。无力/支撑数值PASS阈值，无收敛/生物/站立结论。

父独立检查采样计数/时钟及汇总差，实际看图。来源SHA before/after，非全mesh归档。最后12.5us仍非真值，下一步依差异解释事件局部敏感性而非无限细化或按表现搜索。
