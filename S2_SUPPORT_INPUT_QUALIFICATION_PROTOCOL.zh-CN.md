# A门输入资格：单端点原生力域/映射对照（执行前）

手动研究非成品。仅核实前轮证书的输入代表性，不复求LP、不积分、不改变姿态/摩擦/增益。载入canonical模型和保存raw.npz的qpos，act/ctrl/qvel及外载全0，fresh MjData，恰好1次mj_forward，预算30s。XML/姿态/旧raw/metadata/脚本/protocol及解析模型mesh依赖前后hash记录；网格按flygym实际_load_mjcf的spec.meshes文件路径读取，不按猜测asset目录。此身份是本轮，不能回溯证明上轮未归档mesh。

归档compiled cone/jacobian/solver/flags/gravity与全部constraint类型、contact geom/body/frame/friction/efc_address/dist/include/exclude/margin/gap、关节/腱限位与force clipping/gravcomp/frictionloss/plugins/mocap/equality/外载。只报告实际字段存在者，不将默认值当观测。

MuJoCo condim3 pyramidal每接触4行，计算接触点body2-body1的J，构造n±mu1*t1,n±mu2*t2映射C；逐条比较C列与原生efc_J对应4行转置（顺序允许按最近配对并报告误差，不改列符号或缩放迎合）。要求rtol1e-10/atol1e-10工程身份；接触集/位置/frame/摩擦和旧raw.contact_diamond逐项一致（同容差），旧moment/base/passive/bias也比对。比较native efc_J.T@efc_force与qfrc_constraint检验没有漏约束（浮点误差阈值同上）。若不匹配，记录具体失败但不改历史文件/证书。

旧R的全90列本轮不重新采样，已有仿射与outer候选复测仅有限证据。静态可行域从原生efc_J非负行系数得到；若4行与diamond一致且无未纳入反力/actuator/joint耦合，证书可支持此姿态在理想原生pyramidal力域下的必要条件失败（仍以保存float系数为证，非区间物理证明）。若不一致需修采集映射，不直接调参。soft contact law/qacc稳定性不由此证明。

保存endpoint.npz和report/依赖manifest，先存原始再判定。无新图/长时结果，不为这次一次forward造图。退出0代表资格核验通过，退出1保留不符/异常。下一动作以结果决定，停止重复同端点验收。
