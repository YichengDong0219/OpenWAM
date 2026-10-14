"""Result-blind holdout adapter gate/empty-GT end-to-end fixture; no real effects read."""
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
import g2_holdout_analysis_codex as H
R=Path('/home/dongyicheng/OpenWAM');P=R/'outputs/probe/g2_protocol_20261014/G2_PREREG.json';E=R/'outputs/probe/g2_explore_freeze_R28/G2_EXPLORE_FREEZE_RECEIPT_R28.json';F=P.with_name('G2_FREEZE_RECEIPT.json')

def main():
 passed=[]
 with tempfile.TemporaryDirectory(prefix='check_holdout_analysis_',dir=R/'outputs/probe') as td:
  root=Path(td)
  def put(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d));return p
  pre=json.loads(P.read_text());obs=[];pose=np.zeros((32,20),np.float32);pose[:,[3,7,13,17]]=1
  for task in sorted(pre['holdout']['expert_seed_starts']):
   for ep in range(3):
    for slot in range(2):
     oid=f'{task}_fixture_ep{ep}_slot{slot}';rp=put(root/(oid+'.json'),{'actions':pose.tolist()});tp=root/(oid+'.pt');torch.save({},tp)
     obs.append({'obs_id':oid,'task':task,'episode':ep,'split':'holdout','relative_stage':['early','late'][slot],'gt':{'eligible':False},'record':str(rp),'record_sha256':H.F.sha(rp),'trace':str(tp),'trace_sha256':H.F.sha(tp)})
  selection=put(root/'selection.json',{'format':'openwam-g2-holdout-selection/1','prereg_sha256':H.PREREG,'freeze_sha256':H.F.sha(F),'explore_freeze_sha256':H.F.sha(E),'observations':obs})
  approval=put(root/'approval.json',{'synthetic_fixture_only':True})
  receipt=put(root/'receipt.json',{'format':'openwam-g2-holdout-manifest-receipt/1','status':'frozen','prereg_sha256':H.PREREG,'n_observations':18,'approvals':[{'by':b,'path':str(approval),'sha256':H.F.sha(approval)} for b in ['codex','claude']],'selection':{'path':str(selection),'sha256':H.F.sha(selection)}})
  _,_,refs=H.load_inputs(P,F,E,receipt);passed.append('joint frozen fixture accepted without effect reads')
  orig=json.loads(receipt.read_text());bad=dict(orig,status='draft');put(receipt,bad)
  try:H.load_inputs(P,F,E,receipt)
  except AssertionError:passed.append('draft holdout receipt rejected')
  else:raise AssertionError('draft accepted')
  put(receipt,orig);broken=dict(orig,selection={'path':str(selection),'sha256':'0'*64});put(receipt,broken)
  try:H.load_inputs(P,F,E,receipt)
  except AssertionError:passed.append('wrong selection hash rejected')
  else:raise AssertionError('bad SHA accepted')
  put(receipt,orig)
  seeds=[(v,a) for v in pre['video_seeds'] for a in pre['action_seeds']];roots={};binds={};bindhash={}
  for scope in ['Q5','Q6']:
   man=put(root/(scope+'_manifest.json'),{});bind=put(root/(scope+'_binding.json'),{'format':'openwam-runtime-binding/2','scope':'Q5_alpha' if scope=='Q5' else 'Q6_idm','status':'jointly_bound','prereg_sha256':H.PREREG,'signatures':{'codex':'synthetic','claude':'synthetic'},'manifest_path':str(man),'manifest_sha256':H.F.sha(man),'reviews':[{'path':str(approval),'sha256':H.F.sha(approval)}]});binds[scope]=bind;bindhash[scope]=H.F.sha(bind);directory=root/scope;directory.mkdir();roots[scope]=directory;rows=[];keys=H.keys(scope,obs,seeds,pre['pairs'])
   for o in obs:
    oid=o['obs_id'];localrows={}
    def commit(key,tr):
     path=directory/(key.replace('|','_')+'.pt');torch.save(tr,path);rr={'key':key,'path':str(path),'sha256':H.F.sha(path),'kind':key.split('|')[0]};rows.append(rr);localrows[key]=rr
    def ref(k):return {f:localrows[k][f] for f in ['path','sha256']}
    def trace(kind,v,a,donor):
     acts=pose.copy();acts[:,[0,10]]=(v-1001)*.002+(a-2001)*.0001
     return {'meta':{'format':f'openwam-g2-{scope.lower()}/2','kind':kind,'prereg_sha256':H.PREREG,'runtime_binding_sha256':bindhash[scope],'obs_id':oid,'task':o['task'],'episode':o['episode'],'video_seed':v,'action_seed':a,'donor':donor,'source_trace_sha256':o['trace_sha256'],'record_sha256':o['record_sha256']},'steps':[],'final':{'actions_physical':acts}}
    if scope=='Q6':commit('source|'+oid,{})
    for v,a in seeds:commit(f'baseline|{oid}|{v}-{a}',trace('baseline',v,a,None))
    for v,a in seeds:
     tr=trace('identity',v,a,[v,a]);tr['meta']['receiver_baseline']=ref(f'baseline|{oid}|{v}-{a}');commit(f'identity|{oid}|{v}-{a}',tr)
    for pair in pre['pairs']:
     vr,ar=pair['receiver'];vd,ad=pair['donor'];kind='swap' if scope=='Q5' else 'alt';tr=trace(kind,vr,ar,[vd,ad]);tr['final']['actions_physical']=trace(kind,vd,ad,None)['final']['actions_physical'];tr['meta'].update(receiver_baseline=ref(f'baseline|{oid}|{vr}-{ar}'),donor_baseline=ref(f'baseline|{oid}|{vd}-{ad}'));commit(f'{kind}|{oid}|{vr}-{ar}|{vd}-{ad}',tr)
   assert {r['key'] for r in rows}==keys
   (directory/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));put(directory/'RUN_STAMP.json',{'prereg_sha256':H.PREREG,'mode':'run','runtime_binding_sha256':bindhash[scope],'split':'holdout','holdout_refs':refs})
  output=root/'analysis';cmd=[sys.executable,str(Path(H.__file__).resolve()),'--prereg',str(P),'--freeze',str(F),'--explore-freeze',str(E),'--holdout-manifest',str(receipt),'--q5',str(roots['Q5']),'--q6',str(roots['Q6']),'--q5-binding',str(binds['Q5']),'--q6-binding',str(binds['Q6']),'--out',str(output)];subprocess.run(cmd,check=True)
  result=json.loads((output/'analysis.json').read_text());assert result['split']=='holdout' and not result['explore_effects_loaded'];assert result['counts']['Q5_effect']==result['counts']['Q6_effect']==864;assert result['counts']['Q6_action_scatter']==432;passed.append('18-observation full synthetic CLI exact groups/counts')
  for x in result['groups']['GT_command_consistency']['tasks'].values():assert x['n_rows']==x['n_observations_with_rows']==0 and x['metrics']['delta_E_m']['mean'] is None and x['metrics']['delta_E_m']['n_eligible_episodes']==0
  passed.append('zero-GT subset gives null, never zero or silently omitted group')
  for x in result['groups']['Q5_direction']['tasks'].values():assert x['metrics']['p']['mean']==1 and x['metrics']['r_perp_m']['mean']==0
  passed.append('fixture donor direction and frozen metric functions preserved')
  # Orphan rejection must happen before a fresh result directory exists.
  (roots['Q5']/'orphan.pt').write_bytes(b'x');cmd[-1]=str(root/'rejected_out');r=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);assert r.returncode and not (root/'rejected_out').exists();passed.append('orphan unit rejected before analysis output')
 report={'format':'codex-holdout-analysis-fixture/1','passed':passed,'no_real_effects_loaded':True,'adapter_sha256':H.F.sha(H.__file__),'fixture_script_sha256':H.F.sha(__file__),'frozen_metric_sha256':H.ANALYSIS}
 p=Path(sys.argv[1]);p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
