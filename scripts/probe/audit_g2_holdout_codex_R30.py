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
  # Fresh out-of-grid source consistency and resumed in-grid baseline, independent fixtures.
  def pt(v,a):
   z=torch.zeros(1);acts=np.full((32,20),v*.000001+a*.00000001,np.float32)
   return dict(schedule=[[1.,1.],[0.,0.]],active_action_mask=torch.ones(80,dtype=torch.bool),init={},steps=[dict(i=0,t_v=1.,t_a=1.,video_latents_in=z,action_latents_in=z,video_velocity=z,action_velocity=z)],final=dict(video_latents=z,action_latents=z,actions_normalized=acts.copy(),actions_physical=acts.copy()))
  class Runner:
   def __init__(self):self.calls=[];self.bad=False
   def rerun(self,src,video_seed,action_seed,record=True):
    self.calls.append((video_seed,action_seed));p=pt(video_seed,action_seed)
    if self.bad and (video_seed,action_seed)==(src['meta']['video_seed'],src['meta']['action_seed']):p['final']['actions_physical']+=1
    return {'probe_trace':p,'actions':p['final']['actions_physical']}
  def makeq(name,seeds):
   q=Q5.Q5.__new__(Q5.Q5);q.split='holdout';q.mode='smoke';q.prereg=json.loads(pre.read_text());q.prereg_sha=psha;q.binding_sha=None;q.runtime={'engine_config_sha256':'x'};q.runtime_code={'commit':'synthetic','dirty':False};q.receipts={};q.runner=Runner();q.ledger=C.Ledger(root/name,{'synthetic':True})
   source=pt(*seeds);source.update(meta={'video_seed':seeds[0],'action_seed':seeds[1],'num_frames':9,'action_num_frames':33,'denoise_steps':1,'seed':42},cond={});q.source=lambda _:source;return q
  q=makeq('outside_grid',(42,42));q.runner.bad=True;assert rejected(lambda:q.make_baselines(obs));assert not q.ledger.rows;assert q.runner.calls==[(42,42)];passed.append('out-of-grid42/42 mismatch rejected before any grid commit')
  q.runner.bad=False;q.make_baselines(obs);assert len(q.ledger.rows)==16;assert len(q.runner.calls)==18;passed.append('out-of-grid positive performs source check then16 baselines')
  q.make_baselines(obs);journal=[json.loads(l) for l in (q.ledger.out/'source_checks.jsonl').read_text().splitlines()];assert len(journal)==2 and all(j['result']=='steps_and_final_bitwise_equal' for j in journal);passed.append('resume rechecks source and appends second source journal')
  q.runner.bad=True;before=dict(q.ledger.rows);assert rejected(lambda:q.make_baselines(obs));assert q.ledger.rows==before;passed.append('resume source mismatch rejected even with completed grid')
  q=makeq('in_grid',(1001,2001));q.make_baselines(obs);key='baseline|o|1001-2001';r=q.ledger.rows[key];tr=torch.load(r['path'],weights_only=False);tr['final']['actions_physical']+=1;torch.save(tr,r['path']);r['sha256']=C.sha256_file(r['path']);q.ledger.index.write_text(''.join(json.dumps(row)+'\n' for row in q.ledger.rows.values()));q.ledger=C.Ledger(q.ledger.out,{'synthetic':True});assert rejected(lambda:q.make_baselines(obs));passed.append('in-grid correctSHA bad baseline rejected on resume by source tensor comparison')
 result={'format':'codex-g2-holdout-adapter-audit/2','passed':passed,'blocking_findings':blocking,'no_model_loaded':True,'source_sha256':{Path(m.__file__).name:C.sha256_file(m.__file__) for m in [C,Q5,Q6]},'script_sha256':C.sha256_file(__file__)};p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p);print(json.dumps({'passed':len(passed),'blockers':blocking}))
if __name__=='__main__':
 import sys;main(sys.argv[1])
