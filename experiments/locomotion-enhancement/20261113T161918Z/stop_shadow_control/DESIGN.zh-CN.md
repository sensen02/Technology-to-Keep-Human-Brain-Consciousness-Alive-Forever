# 单变量停步对照：固定私有影子根，待父审阅执行

问题仅是：旧actual-shadow stop在候选0.787s失稳，若保持其余条件，改为固定handover私有FK根能否改善支撑？不是新pivot，也不训练ANN。已有actual-shadow stop作为归档对照，不重跑任何已完成身体轨迹。

归档控制 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/pivot_teacher_design/trials/stop/RESULT.json` SHA172c095a422040f05c24814a791d74fac3d1d4e4509eef72b9f36d5bcae5587d，VERIFY SHA94d7e696e4f44f3af512310088d8d706276c690db77f37bcd4e894bf105a0b1a，均写入本目录PREREG。对照原本就是FAIL，不因为新候选改旧结论。

`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/stop_shadow_control/teacher.py` 构造仍直接调用v1 make_teacher(c,'stop')，获得同一实际初始q_des/rest/rest_pad_bottom/锚点/参考和全部原StaticTeacher数值参数；只把action绑定为直接原StaticTeacher.action，跳过每次private q_des非执行器/根同步与initial_rot/origin更新。42关节私有目标继续原IK更新，私有根与初始框架保持handover实际值；每次检查实际qpos/qvel/xfrc/qfrc均未写，私有根不漂。

common.py克隆v1标准库helpers并显式重绑ROOT/FILES/frozen；frozen包含新8核心输入、全部旧v1核心输入、旧标准BASE源、归档RESULT/VERIFY/HANDOVER/IDENTITY/BODY_LENGTH和实际trace/geometry/samples/诊断字节SHA。旧科学文件保持不变。实际identity.capture继续补XML/资产/导入源与编译数组。父新增可视plot不属于冻结科学输入，不把无关文件塞入科学身份。

adapter.py重用v1 preflight/runner/audit AST，主数值常量序列严格相等；只改新结果kind与CLI限制stop。预检额外插入同一个Core上的actual-vs-fixed工厂首步action/q_des/desiredtarget精确一致检查；此探针不积分、不改实际状态，实际runner用freshCore，不复用探针teacher。预检唯一Core，两个额外私有FK教师对象不代表额外物理回合。

新的runner完整保留1s settle、4.8s原warm、9.6s stop候选、seed20261113、42有限PD、71mesh、接触、无附着/外力、原安全停止、全部记录、原生峰、旧五门和stop_speed门null。唯一新物理case最多1,232,000原生步，没有新左右pivot、没有参数搜索、没有重跑控制。

saved-only audit先执行原v1独立指标审核，再逐字段比较新旧trace前5800行、geometry前4800行、teacher samples前960行bitexact；比较第960候选首动作同初态、HANDOVER真实状态/target/buffer哈希、初始RNG、compiled model arrays、有限PD参数和体长几何完全一致。不同只能从5.8s后影子根同步策略开始产生；prefix或参数不等则对照不准入。每次只保存新root结果，不重写归档结果。

支撑改善可依据support/NF/COM/drift/fullwindow与旧stop并列；站停速度PASS仍无定义，`task_pass_candidate`继续null。新fixed若FAIL不说明任何身体不可能；若完整支撑改善，只能支持此限定同起点下影子根策略的重要性，不自动证明全机制正确、一般稳停或原地转向可行，更不称神经/neck控制通过。

父先审阅源码，实际命令（使用父已资格Python环境，以下为示例）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/stop_shadow_control/preflight.py --execute-parent-preflight
```

仅预检actualPASS后，再托管启动一次：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/stop_shadow_control/run.py --execute-parent-singlecase
```

run仅两个阶段：一newphysics+一saved/paired audit，JOBS与各stage log完整记exit；不自动重试、不覆盖。产物为本目录PREFLIGHT.json、trials/stop内新轨迹/原审核、MATCHED_COMPARISON.json、SUMMARY.json。代码ONLYCODE检查不等于实际预检/物理通过；子代理没有导入数值库、读取数组、构造Core或执行物理。
