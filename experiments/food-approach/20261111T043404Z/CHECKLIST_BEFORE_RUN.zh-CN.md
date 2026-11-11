# S4/S5：三维ANN偏置完整物理目标优化，启动前检查表

依据 `/run/media/sensen/Data2/cell_wound_prototype/WALKING_RESEARCH_PROTOCOL.zh-CN.md` 和 `/run/media/sensen/Data2/cell_wound_prototype/WALKING_RUN_CHECKLIST.zh-CN.md`。本表为启动前不可变声明；未运行项目不预先通过。后续结果写独立CHECKLIST_FINAL，不改已captured输入。

## A 身份和问题

- 新实验：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/`。对应S4/S5条件化真实运动前置，不是新器官/创伤组件。
- 父checkpoint绝对路径/sha见ROOT/PREREG.json（eeced head2000），旧neutral/fullright工程通过、fullleft滑移失败.51484。旧最优单RH偏置+.05仍.50945未通过；不作身体不可行结论。
- 主要变量：RH三输出神经偏置联合delta3，索引29/32/34，其他NN张量和身体/物理指标冻结。CEM只在完整回合之间提议新的神经参数，部署步内仍由实际感觉/神经记忆/命令→ANN42增量→有限PD，不用外置gait/phase/pose反馈生成动作。
- 预算：最多4代×4候选=16个LEFT完整回合，max2进程CPU1；算法/PCG64seed20261111、antithetic顺序、界限±.25、elite2更新/目标/早停/选择规则已ROOT预登记，看到当前代全4结果后才更新下一代。只有实际全门通过候选可选择，未跑mean不算模型通过。
- 每代四NN文件和proposal身份在任何该代物理前冻结，优化器state/RNG/score/更新和对应NNsha逐项保存。实际输入源、compiled参数、资产和环境由原物理recorder身份记录；启动前source_paths逐项file/hash检查。
- 五工程门与strictNF0/all6/yaw.01/full1+4.8+9.6按原定义，完整9.6自主窗包含起始渐变，slip<=.5不能改分母/删窗；原D1—D8正式确认另列，不以endpoint工程score替代。

## B 适用正确性门（待实际证据）

- PPO损失/GAE/KL：不适用，本实验无PPO/critic/logstd/GAE/梯度更新。不得伪称检查了PPO。CEM提议/精确tensor变更/RNG恢复/完整物理目标排序需要实际短预检。
- BC优化器接续：不适用，无Adam更新。旧checkpoint原优化状态仅谱系保留，不声称新bias由旧Adam训练。
- 必须验证：实际RH关节名称顺序及允许三scalar变化、全部其他权重逐位一致；负/零分支2880均值/动作/128+64隐藏状态不变；source路径/JSON值/脚本命名空间/candidatecase绑定。
- 每个实际回合后必须独立载入对应神经候选重算所有2880输出/192state链、15400端点时钟/previous-action/控制来源/zeroextra/71compiledgeometry/HANDOVER/原门周期yaw。身份失败不是优化score，先保留部分证据并修入口，不能拿缺失量按0通过。
- 运行前所有检查尚未执行，不预先勾选。job_id/PID/退出码将以实际托管工具结果记录。

## C 选后确认（条件项）

- 若有left全PASS候选，同checkpoint实际neutral/fullright和独立审计必须通过；fullright实际全数组与已合格c6b轨迹精确保持。若没有候选通过，不运行这些条件确认。
- 单个固定回合不能宣称命令连续范围/切换/新初态/原生D1—D8。下一步一份selected LEFT native1,232,000原生步记录的方法可并行CODEONLY准备，但在选后资格和记录/审计方案冻结前不执行。
- 最终视频本搜索阶段不适用；若selected经native通过，再从同次状态数组只读渲染完整71mesh/所有阶段，解码与实际打开，渲染不得积分改状态或给身体补动作。

## D 记账与资源

- 旧NN/源码/失败/manifest和三个已更正旧状态指针冻结不再改。本实验CURRENT_STATE.json/LIVE_PROGRESS.json显式可变且排除所有执行源和科学产物封存；immutableproposal和已完成结果不能混入同名可变文件。
- 正常项目内已授权无需新增逐次审批。现可用磁盘此前约160GB；endpoint优化记录远小于native全量，native仅selected一次条件执行。所有作业id跟踪并收集，结束前不可丢弃仍有意义的后台任务。
- 确认无用临时文件即时清理；保留实际失败/参数谱系/检查点和必要复现代码。绝对力/SI/生物单位未独立标定仍NOT_ESTABLISHED。
