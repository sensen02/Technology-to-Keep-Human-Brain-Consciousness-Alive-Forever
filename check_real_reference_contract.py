"""Offline evidence contract. Unknown is a successful audit result, not acceptance."""
from pathlib import Path
import json,hashlib,sys
S=Path('/home/sensen/Desktop/engine修复验收/real-freewalking-3d/20261027T150000Z')
R=Path('/home/sensen/Desktop/engine修复验收/s4-real-sixleg-reader/20261027T212200Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-real-reference-source-contract/20261027T213000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def cite(p,needle):return [dict(path=str(p),line=i,text=l.strip(),sha256=sha(p)) for i,l in enumerate(p.read_text().splitlines(),1) if needle in l]
def main():
 O.mkdir(parents=True,exist_ok=False);files=[S/'REFERENCE_METADATA.json',S/'REPORT.zh-CN.md',S/'AUTHOR_README.md',S/'POSTPROCESS_SOURCE.py',S/'POSTPROCESS_CONFIG.yaml',S/'bout_0000__kp_data.npy',S/'info__kp_names.json',R/'SOURCE_CONTRACT.json',R/'INTEGRITY.json',Path(__file__)];h={str(p):sha(p) for p in files};sm=json.loads((S/'FINAL_MANIFEST_SHA256.json').read_text());rm=json.loads((R/'MANIFEST_SHA256.json').read_text())
 for p in files:
  if p.parent==S:assert sha(p)==sm[p.name]
  elif p.parent==R:assert sha(p)==rm[p.name]
 refs=[]
 for p,needles in [(S/'AUTHOR_README.md',['Multi-view markerless','data3D.csv','render_bout_clips','Procrustes alignment']), (S/'POSTPROCESS_CONFIG.yaml',['source_hz','target_hz','enabled: true','percentile','target_z']), (S/'POSTPROCESS_SOURCE.py',['kp_data','start_frames']), (S/'REPORT.zh-CN.md',['许可','session1','长度单位','数据文件'])]:
  for word in needles:refs.extend(cite(p,word))
 result=dict(record='session1/2025_10_07_17_15_30/bout_0000',frames=[6555,7239],direct_evidence_supported={'archived_extracted_array_hash_and_names':True,'historical_extraction_range_ledger_present':True,'real_multiview_reconstruction_pipeline':True,'finite_6leg_50keypoint685frame_readout':True,'nominal_sampling_rate_from_archived_processing_config':800},project_adapter_conventions={'origin':'Scutellum','forward':'toward Antenna_Base','left':'WingR_base->WingL_base orthogonalized','up':'forward cross left, NOT calibrated gravity axis','time':'source_frame / nominal800, not acquisition timestamp'},still_missing={'physical_coordinate_units':'NOT_ESTABLISHED','source_lab_axis_orientation':'NOT_ESTABLISHED','record_specific_calibration_and_transform':'NOT_ESTABLISHED','specific_bout_walking_annotation':'NOT_ESTABLISHED','matching_raw_video_visual_check':'NOT_ESTABLISHED','independent_dataset_license':'NOT_ESTABLISHED','code_license':'NOT_ESTABLISHED: pinned LICENSE HTTP404 does not prove no other license exists','full_remote_hdf5_hash':'NOT_AVAILABLE: no full download','true_ground_plane_and_contacts':'NOT_ESTABLISHED'},license_separation={'paper':'archived paper review CC BY-NC4.0; does not grant data/code rights','code':'fixed repo README no explicit grant; LICENSE path404','data':'public Drive availability is not explicit license'},generation_chain={'documented_generic':'multiview videos -> JARVIS data3D.csv -> optional filtering/Procrustes scaling -> STAC/combined processed kp_data','record_specific':'actual per-bout transform/calibration/config not supplied in verified local excerpt','not_inferred':'defaults do not prove this file was interpolated800->1000;685 frames matching endpoints does not prove no upstream imputation'},acceptance='PARTIAL_NO_STATUS_PROMOTION',neither_title_nor_filename_proves_bout_walking=True,need=['record-specific running_bouts_summary.csv row linking6555–7239 and annotation method','matching camera video segment plus frame correspondence','calibration file and exported-unit/axis transformation statement for this session','actual preprocessing log/config/scaling transform for this bout','explicit data license linked by authors'])
 dump(O/'CONTRACT_RESULT.json',result);dump(O/'LOCAL_REFERENCES.json',refs);dump(O/'INPUT_CODE_HASHES.json',h);assert all(sha(Path(p))==v for p,v in h.items());dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,models=0,forward=0,body_steps=0,neural_steps=0,new_video_downloads=0,failures=[],source_gaps_are_not_runtime_failure=True));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print('offline audit exit0; units axes walking-label video licenses NOT_ESTABLISHED')
def fulltext_audit():
 F=Path('/home/sensen/Desktop/engine修复验收/s4-fulltext-evidence/20261027T214000Z');text=(F/'PAPER.txt').read_text();pages=text.split('\f');rows=[]
 topics={'calibration':'We developed a calibration routine','scale':'A Procrustes scale alignment','behavior':'Running data. A series','sampling':'Video is acquired at','availability':'Data and Code Availability'}
 for topic,needle in topics.items():
  matches=[]
  for page,body in enumerate(pages,1):
   lines=body.splitlines()
   for i,line in enumerate(lines):
    if needle in line:matches.append({'pdf_page':page,'excerpt':'\n'.join(lines[max(0,i-1):i+7])})
  assert matches,topic
  rows.append({'topic':topic,'category':'ONLY_GENERAL_METHOD_SUPPORT','matches':matches,'record_specific':False})
 for key in ['session1/2025_10_07_17_15_30','bout_0000','6555','7239']:
  rows.append({'topic':key,'category':'PUBLIC_RECORD_EVIDENCE_NOT_OBTAINED','literal_present_in_complete_pdf_text':key in text})
 assert (F/'PAPER.pdf').read_bytes().startswith(b'%PDF');dump(F/'FULLTEXT_EVIDENCE_TABLE.json',rows);dump(F/'FULLTEXT_CONTRACT.json',{'acceptance':'PARTIAL_NO_STATUS_PROMOTION','physical_units':'NOT_ESTABLISHED','record_axes':'NOT_ESTABLISHED','record_walking_label':'NOT_ESTABLISHED','array_rate':'nominal800Hz; generic acquisition800Hz confirmed, not record-specific resampling audit','paper_license':'CC BY-NC4.0 in archived full PMC HTML','code_license':'NOT_ESTABLISHED','dataset_license':'NOT_ESTABLISHED','supplement_pdf':'download returned preparing-to-download HTML, pdftotext exit1; no bypass attempted','required_record_files':['session calibration landmark coordinates with physical units and camera matrices','actual per-bout Procrustes scale/transform and preprocessing export config','running_bouts_summary.csv row for record/bout/start6555/end7239 and annotation method','camera video/frame mapping for this bout','explicit independent data license']});dump(F/'FULLTEXT_INPUT_CODE_HASHES.json',{str(p):sha(p) for p in [F/'PAPER.pdf',F/'PAPER.txt',F/'PMC.html',F/'SUPPLEMENT.pdf',Path(__file__)]});print('fulltext audit exit0; generic method evidence located; no record-specific promotion')
if __name__=='__main__':
 if '--fulltext' in sys.argv:fulltext_audit()
 else:main()
