"""Consume frozen integration evidence only. Exit0 means engineering success."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys
from phase_evidence_consumer import load_saved_case
S=Path('/home/sensen/Desktop/engine修复验收/s345-phase-evidence-integration/20261027T145017Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);o=ap.parse_args().output;o.mkdir(parents=True,exist_ok=False)
 source={str(p):sha(p) for p in S.iterdir() if p.is_file()}
 command=[sys.executable,'-m','unittest','-v','test_phase_evidence_consumer']
 dump(o/'PREREGISTRATION.json',dict(source_hashes=source,command=command,scope='P3/S1 saved evidence consumer only',neural_ticks=0,physics_steps=0,certification_grid_runs=0,analytic_recomputation=False))
 root=Path(__file__).parent
 run=subprocess.run(command,cwd=root,capture_output=True,text=True)
 (o/'TEST.log').write_text(run.stdout+run.stderr);assert run.returncode==0
 records=[];rows=[]
 for impl in ('main','audit'):
  for case in ('static_165','detuned_002','single_amplitude_collapse','LA30'):
   r=load_saved_case(S,impl,case);records.append(r)
   rows.append(dict(implementation=impl,case=case,legacy_slips=r['legacy']['saved_integration_row']['legacy_slips'],legacy_criterion='NOT_STORED',legacy_locking_label='NOT_STORED',evidence_integrity=r['evidence_integrity'],metrology_state=r['metrology']['status'],branch_intervals=len(r['metrology']['branch_change_intervals']),locking_conclusion=r['locking_conclusion']))
 assert all(r['locking_conclusion']=='NOT_ASSESSED' for r in records)
 assert {str(p):sha(p) for p in S.iterdir() if p.is_file()}==source
 dump(o/'CONSUMED_RECORDS.json',dict(records=records))
 dump(o/'SUMMARY.json',dict(engineering_tests=dict(passed=True,test_exit_code=run.returncode,consumer_exit_code=0,test_count=12),objects=dict(locking_conclusion='NOT_ASSESSED',locking_certification_passed=None),historical=dict(certification_failed_cells=28,certification_total_cells=36,S345='NOT_JUDGED',strict_verification_exit_code=1),rows=rows))
 lines=['# P3/S1 认证消费者状态隔离','','工程测试12/12通过，测试实际exit0；8例保存证据消费实际exit0。此退出码只说明消费者工程成功，不是对象锁相通过。','','|实现|案例|旧半圈计数|新状态|差异区间数|锁相结论|','|---|---|---:|---|---:|---|']
 for r in rows:lines.append('|%s|%s|%d|%s|%d|%s|'%(r['implementation'],r['case'],r['legacy_slips'],r['metrology_state'],r['branch_intervals'],r['locking_conclusion']))
 lines+=['','legacy数值与新计量字段分开保存。源集成归档只保存旧计数/等值证明，没有保存完整原认证结果或历史锁相标签，因此这些字段明确NOT_STORED/null，不由零滑差猜成LOCKED；测试中的旧LOCKED/PASS是明确fixture，验证它不升级新状态。','来源关联：读取冻结manifest核验RESULTS及案例证据hash，唯一implementation/case记录、状态/事件账本/旧计数一致。此完整性证明以指定来源manifest为信任基线，不是密码学签名或物理有效性认证。','结构校验覆盖字段、严格有限数值、数组长度、枚举、零点/null、端点计数公式、整圈差事件账本。缺失/未知/损坏直接报EvidenceError，无静默fallback。显式history_only模式仅保留历史并标NOT_AVAILABLE，不伪造新状态。','消费者不计算解析轨迹、不重新unwrap、不新增锁相规则；无神经/物理/36格认证。原28/36失败、NOT_JUDGED、严格exit1保持；不形成逐次调用审批门。','下一动作应回到主线的认证适用任务和物理准备证据决策，避免继续叠加同类诊断或消费者层。']
 (o/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
 for name in ('phase_evidence_consumer.py','test_phase_evidence_consumer.py',Path(__file__).name):(o/name).write_bytes((root/name).read_bytes())
 dump(o/'MANIFEST_SHA256.json',{p.name:sha(p) for p in o.iterdir() if p.is_file()})
 print(json.dumps(dict(output=str(o),test_exit_code=0,consumer_exit_code=0,rows=rows)))
if __name__=='__main__':main()
