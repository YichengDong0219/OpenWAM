"""Independent CPU acceptance of holdout gates and Q5 baseline consistency.
No implementer fixtures, model or real holdout data loaded.
"""
import argparse,copy,json,tempfile
from pathlib import Path
import numpy as np
import torch
import g2_common as C
import g2_q5 as Q5
import g2_q6 as Q6

def rejected(fn):
 try:fn()
 except (SystemExit,ValueError,KeyError,RuntimeError):return True
 return False

def main(out):
 passed=[];blocking=[]
 with tempfile.TemporaryDirectory(prefix='codex_R29_',dir=C.PROBE_ROOT) as td:
  root=Path(td)
  def put(name,d):
   p=root/name;p.write_text(json.dumps(d));return p
  pre=put('pre.json',dict(format='openwam-g2-prereg/1',observations=[],video_seeds=[1001,1002,1003,1004],action_seeds=[2001,2002,2003,2004],pairs=[],gt_eligibility={'eligible_explore_obs_ids':[]}));psha=C.sha256_file(pre)
  appr=put('approval.json',{});approvals=[dict(by=b,path=str(appr),sha256=C.sha256_file(appr)) for b in ['codex','claude']]
  freeze=put('freeze.json',dict(format='openwam-g2-freeze-receipt/1',status='frozen',prereg_sha256=psha,approvals=approvals))
  art=put('art.json',{});exdata=dict(format='openwam-g2-explore-freeze-receipt/1',status='frozen',prereg_sha256=psha,approvals=approvals,artifacts=[dict(role=role,path=str(art),sha256=C.sha256_file(art)) for role in ['analysis_script','interpretation']]);ex=put('ex.json',exdata)
  rec=put('rec.json',{'prompt':'p'});srcpath=root/'source.pt';torch.save({},srcpath)
  obs=dict(obs_id='o',task='t',episode=0,split='holdout',gt={'eligible':True},record=str(rec),record_sha256=C.sha256_file(rec),trace=str(srcpath),trace_sha256=C.sha256_file(srcpath))
  def selection(over=None):return dict(format='openwam-g2-holdout-selection/1',prereg_sha256=psha,freeze_sha256=C.sha256_file(freeze),explore_freeze_sha256=C.sha256_file(ex),observations=[obs],**(over or {}))
  sel=put('sel.json',selection())
  def manifest():return dict(format=C.HOLDOUT_MANIFEST_FORMAT,status='frozen',prereg_sha256=psha,approvals=approvals,n_observations=1,selection={'path':str(sel),'sha256':C.sha256_file(sel)})
  hm=put('hm.json',manifest())
  def gate():return C.check_holdout_gate(pre,freeze,ex,hm)
  assert gate()['observations']==[obs];passed.append('joint frozen gate positive control')
  for typ in ['one_signature','draft','wrong_count','wrong_prereg','wrong_selection_sha','duplicate_obs','empty_obs','wrong_split','wrong_selection_explore','changed_interpretation','changed_approval']:
   ex=put('ex.json',exdata);sel=put('sel.json',selection());md=manifest()
   if typ=='one_signature':md['approvals']=approvals[:1]
   if typ=='draft':md['status']='draft'
   if typ=='wrong_count':md['n_observations']=2
   if typ=='wrong_prereg':md['prereg_sha256']='0'*64
   if typ=='wrong_selection_sha':md['selection']['sha256']='0'*64
   if typ in ['duplicate_obs','empty_obs','wrong_split','wrong_selection_explore']:
    sd=selection()
    if typ=='duplicate_obs':sd['observations']=[obs,obs];md['n_observations']=2
    if typ=='empty_obs':sd['observations']=[];md['n_observations']=0
    if typ=='wrong_split':sd['observations']=[dict(obs,split='explore')]
    if typ=='wrong_selection_explore':sd['explore_freeze_sha256']='0'*64
    sel=put('sel.json',sd);md['selection']['sha256']=C.sha256_file(sel)
   if typ=='changed_interpretation':art.write_text('changed')
   if typ=='changed_approval':appr.write_text('changed')
   hm=put('hm.json',md);assert rejected(gate),typ;passed.append(typ+' rejected')
   art.write_text('{}');appr.write_text('{}')
  # Actual Q5 and Q6 constructors must stop at the gate, before Ledger or model.
  ex=put('ex.json',exdata);sel=put('sel.json',selection());hm=put('hm.json',dict(manifest(),status='draft'))
  for name,module,klass in [('Q5',Q5,Q5.Q5),('Q6',Q6,Q6.Q6)]:
   old=module.check_binding;module.check_binding=lambda *a:{'manifest':{},'runtime_binding':{'sha256':'b'*64}}
   dest=root/name
   args=argparse.Namespace(prereg=str(pre),out=str(dest),config=None,n_obs=1,freeze_receipt=str(freeze),runtime_binding='synthetic',split='holdout',explore_freeze=str(ex),holdout_manifest=str(hm),selection=None)
   def factory(*args):raise AssertionError('model factory reached')
   try:assert rejected(lambda:klass(args,'run',runner_factory=factory));assert not dest.exists();passed.append(name+' bad gate stops before Ledger and model')
   finally:module.check_binding=old
  # Own runner reproduces the observed real source seed42, outside registered grid.
  class Runner:
   def __init__(self):self.calls=[]
   def rerun(self,src,video_seed,action_seed,record=True):
    self.calls.append((video_seed,action_seed));acts=np.ones((32,20),np.float32)
    pt={'schedule':[[1.,1.],[0.,0.]],'active_action_mask':torch.ones(80,dtype=torch.bool),'init':{},'steps':[],'final':{'actions_physical':acts}}
    return {'probe_trace':pt,'actions':acts}
  def makeq(name,seeds):
   q=Q5.Q5.__new__(Q5.Q5);q.split='holdout';q.mode='smoke';q.prereg=json.loads(pre.read_text());q.prereg_sha=psha;q.binding_sha=None;q.runtime={'engine_config_sha256':'x'};q.runtime_code={'commit':'synthetic','dirty':False};q.receipts={};q.runner=Runner();q.ledger=C.Ledger(root/name,{'synthetic':True})
   source={'meta':{'video_seed':seeds[0],'action_seed':seeds[1],'num_frames':9,'action_num_frames':33,'denoise_steps':1,'seed':42},'cond':{},'final':{'actions_physical':np.zeros((32,20),np.float32)}};q.source=lambda _:source;return q,source
  q,source=makeq('outside_grid',(42,42));q.make_baselines(obs)
  if (42,42) not in q.runner.calls:blocking.append({'id':'online_seed_outside_grid_unchecked','evidence':'all16 baselines committed; no source42/42 rerun performed even when own-source rerun would disagree','calls':q.runner.calls,'committed':len(q.ledger.rows)})
  else:passed.append('source outside grid explicitly rerun')
  q,source=makeq('resume',(1001,2001));source['final']['actions_physical']=np.ones((32,20),np.float32);q.make_baselines(obs)
  key='baseline|o|1001-2001';r=q.ledger.rows[key];tr=torch.load(r['path'],weights_only=False);tr['final']['actions_physical']+=1;torch.save(tr,r['path']);r['sha256']=C.sha256_file(r['path']);q.ledger.index.write_text(''.join(json.dumps(row)+'\n' for row in q.ledger.rows.values()));q.ledger=C.Ledger(q.ledger.out,{'synthetic':True})
  if not rejected(lambda:q.make_baselines(obs)):blocking.append({'id':'resumed_online_baseline_not_revalidated','evidence':'source-grid baseline with valid index SHA but changed final actions accepted on resume'})
  else:passed.append('resumed online baseline checked against source')
 result={'format':'codex-g2-holdout-adapter-audit/1','passed':passed,'blocking_findings':blocking,'no_model_loaded':True,'source_sha256':{Path(m.__file__).name:C.sha256_file(m.__file__) for m in [C,Q5,Q6]},'script_sha256':C.sha256_file(__file__)};p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p);print(json.dumps({'passed':len(passed),'blockers':blocking}))
if __name__=='__main__':
 import sys;main(sys.argv[1])
