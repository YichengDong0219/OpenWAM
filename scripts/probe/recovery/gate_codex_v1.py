"""Pure resource/stage gate, not a GPU scheduler. No processes are launched.
The executor must durably reserve approved time BEFORE acquiring/loading GPU work,
charge actual elapsed time across all allocated GPUs including failures/cleanup,
and stop at the returned cap. Stage evidence comes from validators pinned in the
jointly accepted bundle, never from a new arbitrary 'passed' flag.
"""
import argparse,datetime,hashlib,json,math
from pathlib import Path

def read_pinned(ref):
 p=Path(ref['path']);raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('receipt hash mismatch')
 return json.loads(raw)
def verdict(policy,state,phase,now,bundle=None,policy_sha=None):
 reasons=[];caps=policy['stage_caps_card_seconds']
 if phase not in caps:return {'allowed':False,'reasons':['unknown phase'],'max_wall_seconds_including_cleanup':0}
 deadline=datetime.datetime.fromisoformat(policy['deadline_utc'])
 if now.tzinfo is None:return {'allowed':False,'reasons':['timezone required'],'max_wall_seconds_including_cleanup':0}
 remaining_deadline=(deadline-now).total_seconds()
 used=state.get('used_card_seconds',{});reserved=state.get('reserved_card_seconds',{})
 if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in list(used.values())+list(reserved.values())):return {'allowed':False,'reasons':['invalid accounting'],'max_wall_seconds_including_cleanup':0}
 if (set(used)|set(reserved))-set(caps):reasons.append('unknown ledger phase')
 if state.get('halted',False):reasons.append('pipeline halted')
 if state.get('attempted_invocations',{}).get(phase,0)>=policy['invocation_caps'][phase]:reasons.append('invocation cap reached')
 left=caps[phase]-used.get(phase,0)-reserved.get(phase,0)
 total_left=policy['global_cap_card_seconds']-sum(used.values())-sum(reserved.values())
 if remaining_deadline<=0:reasons.append('absolute deadline expired')
 if left<=0 or total_left<=0:reasons.append('resource cap exhausted')
 if state.get('active_jobs',0)!=0:reasons.append('another allocation active')
 if state.get('admission_query_success') is not True or state.get('foreign_compute_processes')!=0:reasons.append('admission not clear')
 backend=state.get('backend');free=state.get('free_memory_MiB',-1)
 floor=policy['admission_free_MiB'].get(backend)
 if floor is None or type(free) not in (int,float) or not math.isfinite(free) or free<floor:reasons.append('backend or free memory rejected')
 if phase=='V':
  if state.get('validation_candidate') not in policy['validation_candidates']:reasons.append('unapproved validation candidate')
  if state.get('candidate_already_attempted',False):reasons.append('validation candidate already attempted')
 else:
  if not bundle or bundle.get('policy_sha256')!=policy_sha or set(bundle.get('reviewers',[]))!={'codex','claude'}:
   reasons.append('missing exact joint pipeline freeze')
  else:
   for path,h in bundle.get('bound_files',{}).items():
    if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:reasons.append('bound file changed')
   if not bundle.get('bound_files') or not bundle.get('validators'):reasons.append('empty code or validator binding')
   if state.get('backend_manifest_sha256')!=bundle.get('backend_manifest_sha256'):reasons.append('backend drift')
  requirements={'Q_eng':['V'],'M2':['V','Q_eng'],'Q_eval':['V','M2'],'M3':['V','M2','Q_eval']}[phase]
  for need in requirements:
   ref=state.get('receipts',{}).get(need)
   try:
    if not ref:raise ValueError('missing receipt')
    receipt=read_pinned(ref)
    if receipt.get('stage')!=need or receipt.get('engineering_pass') is not True:raise ValueError('stage not passed')
    if not bundle or receipt.get('validator_sha256')!=bundle.get('validators',{}).get(need):raise ValueError('wrong validator')
    if receipt.get('backend_manifest_sha256')!=state.get('backend_manifest_sha256'):raise ValueError('backend mismatch')
    if need in ['Q_eng','Q_eval']:
     count=3 if need=='Q_eng' else 20;start=200000 if need=='Q_eng' else 300000
     seeds=receipt.get('accepted_scene_seeds',[])
     if len(seeds)!=count or len(set(seeds))!=count or seeds!=sorted(seeds) or any(type(s)!=int or s<start for s in seeds):raise ValueError('qualification seed list invalid')
     if receipt.get('candidate_count',10**9)>policy['qualification_candidate_caps'][need]:raise ValueError('candidate cap exceeded')
     if receipt.get('selection_algorithm_sha256')!=bundle.get('selection_algorithm_sha256') or receipt.get('manifest_sha256')!=ref.get('manifest_sha256'):raise ValueError('qualification manifest or selector mismatch')
    if need=='M2':
     if receipt.get('complete_episodes')!=9 or receipt.get('common_prefix_pass') is not True or receipt.get('invariants_pass') is not True:raise ValueError('incomplete engineering pilot')
     if receipt.get('scientific_bundle_sha256')!=bundle.get('scientific_bundle_sha256'):raise ValueError('scientific rules changed')
     # Deliberately NEVER inspect success/rescue counts to authorize M3.
   except (ValueError,KeyError,OSError,TypeError):reasons.append('missing or invalid '+need+' evidence')
 # One GPU allocation at a time. Never round a small remaining budget upward.
 requested=max(0,int(state.get('requested_wall_seconds_including_cleanup',0)))
 hard=max(0,int(min(left,total_left,remaining_deadline,policy['per_invocation_cap_seconds'][phase],requested)))
 if hard<=policy['kill_grace_seconds']:reasons.append('insufficient usable wall budget')
 return {'allowed':not reasons,'reasons':sorted(set(reasons)),'max_wall_seconds_including_cleanup':hard,'max_work_seconds':max(0,hard-policy['kill_grace_seconds']),'must_reserve_card_seconds_before_launch':hard,'allocated_gpu_count':1}
def main():
 a=argparse.ArgumentParser();a.add_argument('--policy',required=True);a.add_argument('--state',required=True);a.add_argument('--phase',required=True);a.add_argument('--bundle');a.add_argument('--bundle-sha');a.add_argument('--now',required=True);x=a.parse_args()
 raw=Path(x.policy).read_bytes();policy=json.loads(raw);state=json.loads(Path(x.state).read_text());bundle=None
 if x.bundle:
  if not x.bundle_sha:raise SystemExit('explicit published bundle SHA required')
  bundle=read_pinned({'path':x.bundle,'sha256':x.bundle_sha})
 v=verdict(policy,state,x.phase,datetime.datetime.fromisoformat(x.now),bundle,hashlib.sha256(raw).hexdigest());print(json.dumps(v));raise SystemExit(0 if v['allowed'] else 2)
if __name__=='__main__':main()
