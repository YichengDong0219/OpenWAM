"""Independent GPU3 v4 Q5 engineering smoke on three prereg EXPLORE observations only."""
import argparse,json,hashlib,os,sys,subprocess
from pathlib import Path
import torch
import numpy as np
R=Path('/home/dongyicheng/OpenWAM');sys.path[:0]=[str(R),str(R/'scripts/probe')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 return np.array_equal(a,b)
def compare(a,b):
 return len(a['steps'])==len(b['steps']) and all(eq(s[k],t[k]) for s,t in zip(a['steps'],b['steps']) for k in ['i','t_v','t_a','video_latents_in','action_latents_in','video_velocity','action_velocity']) and all(eq(a['final'][k],b['final'][k]) for k in ['video_latents','action_latents','actions_normalized','actions_physical'])
@torch.no_grad()
def main():
 out=Path(sys.argv[1]);assert not out.exists();out.mkdir(parents=True)
 import g2_q5 as Q;import g2_common as C
 p=R/'outputs/probe/g2_protocol_20261014/G2_PREREG.json';pre=json.loads(p.read_text());by={o['obs_id']:o for o in pre['observations']};obs=[dict(by[v['obs_id']],split='holdout') for v in pre['verification']];sel=out/'engineering_selection.json';sel.write_text(json.dumps({'observations':obs,'note':'EXPLORE relabeling solely for engineering smoke; never a true holdout selection'}))
 args=argparse.Namespace(prereg=str(p),out=str(out/'units'),config=None,n_obs=3,freeze_receipt=None,runtime_binding=None,split='holdout',selection=str(sel),explore_freeze=None,holdout_manifest=None)
 q=Q.Q5(args,'smoke');mp=R/'outputs/probe/g2_runtime_20261014/ALPHA_RUNTIME_MANIFEST_Q5_v4.json';assert sha(mp)=='368aa38954794804acbf3325fdcfdb97e626479dd59d7f371f07567ece861578';man=json.loads(mp.read_text());C.check_runtime(q.runner,man);q.run();reports=[]
 for o in obs:
  source=q.source(o);v,a=source['meta']['video_seed'],source['meta']['action_seed'];assert (v,a)==(42,42);pt=q.runner.rerun(source,video_seed=v,action_seed=a)['probe_trace'];assert compare(pt,source)
  for v in pre['video_seeds']:
   for a in pre['action_seeds']:
    br=q.baseline(o,v,a)['trace'];root=Path(pre['alpha_baseline_sources'][o['source_g1_split']]['root']);grid=torch.load(root/o['task']/o['obs_id']/'grid'/f'v{v}_a{a}.pt',map_location='cpu',weights_only=False);assert compare(br,grid)
  reports.append({'obs_id':o['obs_id'],'independent_direct42_source_steps_final_bitexact':True,'baselines_match16_G1_grid':True})
 before=q.ledger.index.read_bytes();q.run();assert q.ledger.index.read_bytes()==before
 journal=[json.loads(l) for l in (q.ledger.out/'source_checks.jsonl').read_text().splitlines()];assert len(journal)==6 and [j['obs_id'] for j in journal[-3:]]==[o['obs_id'] for o in obs];C.check_imports_bound(man)
 d={'all_passed':True,'results':reports,'n_baseline':48,'n_identity':48,'source_journal_rows_fresh_and_resume':6,'resume_index_unchanged':True,'scope':'engineering EXPLORE observations only; no actual C4 effects generated','prereg_sha256':sha(p),'manifest_sha256':sha(mp),'own_script_sha256':sha(__file__)};temp=out/'summary.json.tmp';temp.write_text(json.dumps(d,indent=2)+'\n');os.replace(temp,out/'summary.json');print(json.dumps(d))
if __name__=='__main__':main()
