"""Structural mapping validation; engineering success never means anatomical completeness."""
def validate_mapping(actuators, pools, rows):
 errors=[];seen=set();aliases={};out=[];shared={}
 for row in rows:
  r=dict(row);a=r['actuator'];p=r.get('pool');alias=r.get('alias');expected=r.get('expected');level=r.get('source_level')
  if a not in actuators:errors.append(['unknown_actuator',a])
  if a in seen:errors.append(['duplicate_actuator',a])
  seen.add(a)
  if alias:
   if alias in aliases and aliases[alias]!=expected:errors.append(['conflicting_alias',alias])
   aliases[alias]=expected
  if level not in ('declared_engineering_alias','direct_annotation','missing_basis'):errors.append(['unknown_source_level',a])
  if p is not None:
   if p not in pools:errors.append(['unknown_pool',a,p])
   elif not pools[p]:errors.append(['empty_pool',a,p])
   shared.setdefault(p,[]).append(a)
  r['mapping_state']='unresolved' if p is None else 'mapped'
  r['activity_state']='unmapped' if p is None else 'mapped_zero_events' if r.get('saved_event_count')==0 else 'mapped_observed_events' if r.get('saved_event_count') is not None else 'not_observed'
  out.append(r)
 for a in sorted(set(actuators)-seen):errors.append(['uncovered_actuator',a])
 return dict(engineering_valid=not errors,biological_mapping_complete=False,errors=errors,rows=out,unresolved=[r for r in out if r['mapping_state']=='unresolved'],shared_pools=[dict(pool=p,actuators=aa,declaration='declared source identity shared, not confirmed muscle-specific anatomy') for p,aa in shared.items() if len(aa)>1])
