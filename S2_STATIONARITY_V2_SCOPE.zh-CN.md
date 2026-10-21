# S2/P2 stationarity v2执行界

本轮用户继续，实施独立v2，不改失败v1及历史hash。手动研究非成品。仅既定canonical历史projectedpose、artificial双球各fresh一次mj_forward，无step/积分/参数搜索/重试。任一失败停止，case phase/error记录保留。

模型分配后MjData/dense矩阵之前nv1..256；forward后nefc<=4096/nisland<=mjNISLAND，dense J。立即copy保存raw qM/M/parent/CSR metadata、qacc/qacc_smooth/smooth/constraint/J/force/aref/D/R/type/state/岛映射/solver stats/初末state/DOF标签。首先原始NPZ落盘+SHA+phase，再进行任何质量展开或stationarity分析；raw保存仅finite实numeric及最终payload<=64MiB，非RSS保障。若copy/文件写入前失败，phase明确无现场，不能声称永远保证保存。

调用来源明确：mj_fullM(m,dst,d.qM)，纯decode_tree(d.qM,parent)，纯decode_csr(d.M,rowadr,rownnz,colind)，mj_mulM(m,d,res,e_j)全部nv个基向量得到逐列矩阵；另mulM(qacc)。所有操作后与原始动力snapshot和state比字节，不能改初态。保存所有矩阵/逐DOF误差。三解码+mulM矩阵比较采用工程API一致性rtol1e-12/atol1e-14（既有矩阵结构工程尺度，非solver residual阈值），失败保留raw不调阈值。SPD仍用原analyze检查，不jitter。

stationarity用纯tools_s2_stationarity，真实结果REPORT_ONLY/单位分组，非完整KKT/静态平衡/收敛/生物/站立行走。case完成先保存result NPZ/report再下一case；phase记录每case forward调用数、raw/analysis状态及错误。无自动重试。所有历史94路径和新来源hash前后核对。图按case/unit分面并实际查看。独立子审查不执行模型，仅读取raw/result/元数据复算与检查hash。此轮最多总2forward，不以旧NPZ代替新现场。
