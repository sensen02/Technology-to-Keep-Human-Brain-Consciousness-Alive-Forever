# 运动增强首轮执行检查（S2/S5）

规范来源 `/run/media/sensen/Data2/cell_wound_prototype/WALKING_RESEARCH_PROTOCOL.zh-CN.md`；本记录采用其模板，不增设用户审批。

- root `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/`；父感觉闭环064313；用户最新明确要求转向加速、原地、粗地形与足钩。
- 完整身体/原source/已过baseline保留；首轮ABI只NN固定观察，terrain_contact只单球工程fixture，不冒新完整运动。
- 父140命令现有env/bin/python -B motor_abi_preflight.py --execute-parent-nn-only --out MOTOR_ABI_RESULT.json，OPENBLAS/OMP/MKL=1；已managed collect exit0/allchecksPASS；0phys/optimizer/sensorybrain，3valid各32step＋独立base比较，2invalid拒绝，权重/源/别名/RNG不变。样本为settle前32观察，不构成自主运动。
- 父141命令同envpython -B terrain_contact_probe/terrain_contact_preflight.py --run-parent-authorized --out terrain_contact_probe/ACTUAL，线程1；已managed collect exit0/4fixtureallgatesPASS/MuJoCo3.9.0/6400native12.5us，总4×20ms。每caseXML/MJB/compiled信息与native全部接触记录保留。
- parent独立重核两个RESULT源SHA一致；plane旧新1600接触一致；box/sloped旧漏1600新识别1600；no-contact0；normalerr2.22e-16／无外力／adhesion0／根初态后不写。
- PPO/GAE/训练优化器检查不适用：尚未训练。单球没有关节或执行器，不能算六足站立；没有整蝇粗地形／原地验收或新同次运动视频，明确未执行。
- 2子SOURCE_ONLY/ONLYCODE继续实现terrainCore实际XML身份/观测测量/0Core预检，以及stop/pivot工程教师候选(不部署)。任何fullbody运行仍需实际preflight和新源码身份清单，预算固定再启动。
- 单球与ABI结果无需同次全蝇视频，桌面给阶段报告与实际数值，不当已完成运动演示。旧祖先权重／视频／失败证据不删除。
- 当前新root glob *.pyc无文件，尚无确认无用临时文件，必要compiled MJB/fixtureXML属于复现证据保留。没有新安装、服务、外部消息或动物操作。
- 下一：父读两新CORE设计，短实际验证flat兼容/box接触/terrainrelative净空/实际shadowroot坐标，然后有限完整身体基线与pivot教师测试。
