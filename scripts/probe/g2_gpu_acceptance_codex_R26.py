"""Independent GPU3 engineering subset, no swaps or GT action overrides.

Execute an identical copy outside the repo so the auditor is not a production dependency.
"""
import argparse,hashlib,importlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
REPO=Path('/home/dongyicheng/OpenWAM')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts/probe'))

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    return np.asarray(a).dtype==np.asarray(b).dtype and np.array_equal(a,b)
def independent_identity(base,tr,scope):
    sk=('i','t_a','action_latents_in','action_velocity') if scope=='IDM' else ('i','t_v','t_a','video_latents_in','action_latents_in','video_velocity','action_velocity')
    fk=('action_latents','actions_normalized','actions_physical','cond_video_latents_used') if scope=='IDM' else ('video_latents','action_latents','actions_normalized','actions_physical')
    return len(base['steps'])==len(tr['steps']) and all(exact(s[k],t[k]) for s,t in zip(base['steps'],tr['steps']) for k in sk) and all(exact(base['final'][k],tr['final'][k]) for k in fk)
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['ALPHA','IDM'],required=True);ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
    C=importlib.import_module('g2_common');M=importlib.import_module('g2_q5' if args.scope=='ALPHA' else 'g2_q6')
    name='ALPHA_RUNTIME_MANIFEST_Q5_v2.json' if args.scope=='ALPHA' else 'IDM_RUNTIME_MANIFEST_Q6_v2.json'
    digest='a449d6700316e8de652ab5eec66d4409847333ca22bea9d56c7f4a5d78912fd7' if args.scope=='ALPHA' else 'fd1404786497238af58908a4013e7d6ba1d2bf560a4fd88c7ef82bb7a564fa8d'
    mp=REPO/'outputs/probe/g2_runtime_20261014'/name;assert sha(mp)==digest;man=json.loads(mp.read_text())
    paths=set();bytes_verified=0
    for a in man['artifacts']:
        p=Path(a['path']);assert str(p) not in paths;paths.add(str(p));assert p.stat().st_size==a['bytes'] and sha(p)==a['sha256'];bytes_verified+=a['bytes']
    assert sha(man['resolved_config_snapshot'])==man['resolved_config_sha256']
    assert C.cfg_sha(C.resolve_config(man['deploy_config'],man['checkpoint_dir']))==man['resolved_config_sha256']
    needed=subprocess.check_output(['git','ls-files','openwam','benchmarks/utils','benchmarks/__init__.py'],cwd=REPO,text=True).splitlines()
    assert all(str(REPO/n) in paths for n in needed if n.endswith('.py'))
    assert not any('/.cache/' in p or '/.range_parts/' in p for p in paths)
    save(out/'manifest_audit.json',dict(manifest_sha256=digest,n_artifacts=len(paths),bytes_verified=bytes_verified,all_match=True,resolved_snapshot_match=True,tracked_dependency_closure=True))
    print(json.dumps({'scope':args.scope,'manifest_artifacts_verified':len(paths)}),flush=True)
    prepath=REPO/'outputs/probe/g2_protocol_20261014/G2_PREREG.json';assert sha(prepath)==man['prereg_sha256'];pre=json.loads(prepath.read_text());by={o['obs_id']:o for o in pre['observations']};obs=[by[v['obs_id']] for v in pre['verification']]
    from offline_runner import OfflineRunner
    runner=OfflineRunner(man['checkpoint_dir'],man['deploy_config']);runtime=C.check_runtime(runner,man)
    runner.precompute_prompts([json.loads(Path(o['record']).read_text())['prompt'] for o in obs])
    q=(M.Q5 if args.scope=='ALPHA' else M.Q6).__new__(M.Q5 if args.scope=='ALPHA' else M.Q6)
    q.mode='smoke';q.prereg=pre;q.prereg_sha=sha(prepath);q.prereg_path=prepath;q.binding_sha=None;q.receipts={};q.manifest=man;q.runtime=runtime;q.runner=runner;q.observations=obs
    from openwam.deploy.engine import _code_version
    q.runtime_code=dict(_code_version());assert not q.runtime_code['dirty']
    q.ledger=C.Ledger(out/'units',{'prereg_sha256':q.prereg_sha,'mode':'independent_engineering_subset','binding_sha256':None,'scope':args.scope})
    q._sources={};summary=[];t0=time.time()
    if args.scope=='ALPHA':
        q.g1_prereg_sha=pre['provenance']['g1_prereg']['sha256'];q.baseline_index=q._baseline_indexes();keys=[]
        for o,v in zip(obs,pre['verification']):
            ok=[]
            for vr,ar in (v['receiver'],v['donor']):
                key=f"identity|{o['obs_id']}|{vr}-{ar}";path=q.ledger.out/o['task']/o['obs_id']/f'{vr}-{ar}.pt'
                q.unit(o,'identity',key,path,vr,ar,vr,ar)
                tr=q.ledger.load(key);b=q.baseline(o,vr,ar)['trace'];assert independent_identity(b,tr,args.scope)
                assert [list(map(float,s)) for s in tr['meta']['schedule']]==runtime['schedule'];keys.append(key);ok.append([vr,ar])
                # A resume must validate persisted tensors and references and preserve the index.
                before=q.ledger.index.read_bytes();q.unit(o,'identity',key,path,vr,ar,vr,ar);assert q.ledger.index.read_bytes()==before
            summary.append({'obs_id':o['obs_id'],'independent_identity_bitexact':True,'verified_receivers':ok,'resume_index_unchanged':True})
            print(json.dumps(summary[-1]),flush=True)
        q.ledger.require_exact(keys)
    else:
        q.gt_ids=set(pre['gt_eligibility']['eligible_explore_obs_ids']);q.records={}
        for o in obs:
            assert sha(o['record'])==o['record_sha256'];q.records[o['obs_id']]=json.loads(Path(o['record']).read_text())
        for o in obs:
            src=q.source(o);baselines=q.baselines(o,src);q.identity(o,src,baselines)
            for (vr,ar),b in baselines.items():
                key=f"identity|{o['obs_id']}|{vr}-{ar}";tr=q.ledger.load(key);assert independent_identity(b['trace'],tr,args.scope)
                assert [list(map(float,s)) for s in tr['meta']['schedule']]==runtime['schedule']
            if o['obs_id'] in q.gt_ids:q.gt_future(o,src)
            before=q.ledger.index.read_bytes();q.identity(o,src,baselines);assert before==q.ledger.index.read_bytes()
            summary.append({'obs_id':o['obs_id'],'independent_identity_bitexact':True,'verified_receivers':len(baselines),'resume_index_unchanged':True,'stage1_steps_all4_action_seeds_bitexact':True,'gt_future_validated':o['obs_id'] in q.gt_ids})
            print(json.dumps(summary[-1]),flush=True)
        q.ledger.require_exact(q.expected_keys())
    C.check_imports_bound(man)
    save(out/'summary.json',dict(scope=args.scope,all_passed=True,results=summary,manifest_sha256=digest,runtime=runtime,
                                 n_units=len(q.ledger.rows),prereg_sha256=q.prereg_sha,seconds_generations=time.time()-t0,
                                 script_sha256=sha(__file__),formal_effects_run=False,swaps=0,GT_action_overrides=0))
    print(json.dumps({'scope':args.scope,'all_passed':True,'n_units':len(q.ledger.rows)}),flush=True)
if __name__=='__main__':main()
