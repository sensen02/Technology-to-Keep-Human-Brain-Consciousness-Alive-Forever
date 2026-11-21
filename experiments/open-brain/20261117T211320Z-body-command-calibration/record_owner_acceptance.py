from scope import *
paths=(BASE/'TERRAIN_BODY_CURRENT.json',BASE/'BODY_STOP_REPAIR_CURRENT.json',BASE/'BODY_CURRENT.json')
write(HERE/'PREVIOUS_ENTRIES.json',{str(p):json.loads(p.read_text()) for p in paths})
acceptance=dict(project='P1_open_source_brain',status='OWNER_ACCEPTED_LIMITED_TEACHER_RELATIVE_STAGE',
  statement='用户认为当前身体阶段合格；在同身体、同初态、同颈后指令下以教师能力为参照，开展其余四类指令校对。',
  checkpoint=str(WEIGHT),sha256=sha(WEIGHT),accepted_for_stage_use=True,
  scientific_results_preserved=str(STOP/'FINAL_ACCEPTANCE.json'),
  limits='Historical absolute failures preserved. No claim all teacher failures are paired or all tasks physically completed. No P2/Eon new-brain validation.')
write(HERE/'OWNER_ACCEPTANCE.json',acceptance)
write(BASE/'BODY_STAGE_CURRENT.json',dict(**acceptance,implementation_root=str(HERE),body_entry=str(BASE/'BODY_MODEL_CURRENT.json')))
terrain=json.loads((BASE/'TERRAIN_BODY_CURRENT.json').read_text());terrain.update(status=acceptance['status'],stage_acceptance=str(HERE/'OWNER_ACCEPTANCE.json'),
 stage_entry=str(BASE/'BODY_STAGE_CURRENT.json'),next_actions=['paired_walking_left_right','paired_stride_differential','paired_backward','paired_speed_transitions'])
write(BASE/'TERRAIN_BODY_CURRENT.json',terrain)
print('Recorded owner acceptance with historical physical results preserved')
