import json,sys,math
from collections import defaultdict,Counter
from pathlib import Path
snap=Path('/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_correctness_only_saam60_table_rl_20260913_graph_r1/train/implementation_source_snapshot/src')
sys.path.insert(0,str(snap))
from rl.scenarios.diagnostics.audit_saam_lineage_replay import _process_update,standardized_group_advantages,LineageReplay
p=Path('/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_correctness_only_saam60_table_rl_20260913_graph_r1/train/rollouts.jsonl')
rows=[]; groups=defaultdict(list)
with p.open() as f:
 for line in f:
  if not line.strip(): continue
  r=json.loads(line); rows.append(r); groups[(int(r['policy_global_step']),int(r['example_index']))].append(r)
# group-level raw binary reward and base advantage, only eligible rows get A
items=[]; all_rewards=Counter(); eligible_rewards=Counter(); group_dist=Counter(); trunc=0
for gkey,rs in sorted(groups.items()):
 if len(rs)!=8: raise RuntimeError((gkey,len(rs)))
 elig=[_process_update(r) for r in rs]
 for r in rs:
  all_rewards['correct' if r.get('correct') else 'wrong']+=1
  if _process_update(r): eligible_rewards['correct' if r.get('correct') else 'wrong']+=1
  else: trunc+=1
 c=sum(bool(r.get('correct')) for r in rs); ce=sum(bool(r.get('correct')) and _process_update(r) for r in rs); ne=sum(_process_update(r) for r in rs)
 group_dist[(c,ne,ce)]+=1
 advs=standardized_group_advantages([1.0 if r.get('correct') else 0.0 for r in rs],elig)
 for r,a in zip(rs,advs):
  if _process_update(r): items.append({'step':int(r['policy_global_step']),'ex':int(r['example_index']),'correct':bool(r.get('correct')),'a':float(a),'turns':len(r.get('turns') or []),'error_turns':len(r.get('error_events') or [])})
print('BASE',json.dumps({'rows':len(rows),'groups':len(groups),'eligible':len(items),'excluded':trunc,'all_rewards':all_rewards,'eligible_rewards':eligible_rewards,'group_dist':{str(k):v for k,v in sorted(group_dist.items())}},default=lambda x:dict(x),ensure_ascii=False))
# per cap: trajectory coefficient mass then apply SAAM exact mask for shared keys, but no local error override: use same LineageReplay identities to estimate post-mask coefficient mass.
for cap in [None,2.0,1.5,1.0,0.75]:
 name='none' if cap is None else str(cap)
 total_pos=total_neg=total_abs=0.; clipped=0; clipped_pos=clipped_neg=0.; max_before=0.; max_after=0.
 for x in items:
  a=x['a']; b=a if cap is None else math.copysign(min(abs(a),cap),a)
  if b!=a: clipped+=1; clipped_pos+=int(a>0); clipped_neg+=int(a<0)
  total_abs+=abs(b); total_pos+=max(b,0); total_neg+=max(-b,0); max_before=max(max_before,abs(a)); max_after=max(max_after,abs(b))
 print('CAP',json.dumps({'cap':cap,'trajectories':len(items),'clipped_trajectories':clipped,'clipped_positive':clipped_pos,'clipped_negative':clipped_neg,'abs_mass':total_abs,'positive_mass':total_pos,'negative_mass':total_neg,'max_before':max_before,'max_after':max_after,'negative_positive_ratio':total_neg/total_pos if total_pos else None}))
# Apply exact lineage SAAM mask to per-event coefficient, using capped advantages; key is ambiguous within each step/ex group.
# Build group event sets first.
for cap in [None,2.0,1.5,1.0,0.75]:
 total=pos=neg=removed=0.; amb_events=0; amb_mass=0.; errtraj_pos=0.; local_updates=0
 for gkey,rs in sorted(groups.items()):
  elig=[_process_update(r) for r in rs]
  advs=standardized_group_advantages([1.0 if r.get('correct') else 0.0 for r in rs],elig)
  evall=[]
  for r,a,ok in zip(rs,advs,elig):
   if not ok: continue
   b=a if cap is None else math.copysign(min(abs(float(a)),cap),float(a))
   evs,_=LineageReplay(r).replay()
   for e in evs:
    if e.get('matched') and e.get('action_digest'):
     evall.append((r,e,b,bool(r.get('correct'))))
  bykey=defaultdict(list)
  for r,e,b,corr in evall: bykey[(e['state_digest'],e['action_digest'])].append((r,e,b,corr))
  amb=set(k for k,v in bykey.items() if {x[3] for x in v}=={True,False})
  for r,e,b,corr in evall:
   if (e['state_digest'],e['action_digest']) in amb and not corr: b2=0.; removed+=abs(b); amb_mass+=abs(b); amb_events+=1
   else: b2=b
   total+=abs(b2); pos+=max(b2,0); neg+=max(-b2,0)
   if (not corr) and b2>0: errtraj_pos+=b2
 print('SAAM_CAP',json.dumps({'cap':cap,'abs_mass_after_mask':total,'positive_mass_after_mask':pos,'negative_mass_after_mask':neg,'wrong_positive_mass':errtraj_pos,'newly_masked_wrong_events':amb_events,'removed_wrong_mass':removed,'ambiguous_wrong_mass':amb_mass},ensure_ascii=False))
