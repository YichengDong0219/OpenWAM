"""Result-audited G1 Q1/Q2 analysis; independent numpy metrics and episode bootstrap.

Only the requested split is read. No model forward, simulator or holdout release.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def lines(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]
def arr(x):return x.detach().float().cpu().numpy().astype(np.float64) if torch.is_tensor(x) else np.asarray(x,dtype=np.float64)
def finite(x):
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,list):return all(finite(v) for v in x)
    return not isinstance(x,float) or np.isfinite(x)


def ss_decomposition(grid):
    y=grid[..., [0,1,2,10,11,12]].reshape(4,4,-1)
    center=np.ones((4,4))/4;contrast=np.eye(4)-center
    project=lambda l,r:np.einsum('ij,jkf,kl->ilf',l,y,r)
    raw={name:float(np.square(project(l,r)).sum()) for name,l,r in
         [('video',contrast,center),('action',center,contrast),('interaction',contrast,contrast)]}
    total=float(np.square(y-y.mean(axis=(0,1),keepdims=True)).sum())
    if not np.isclose(sum(raw.values()),total,rtol=1e-12,atol=1e-15):raise ValueError('nonorthogonal SS')
    return {**{'ss_'+k:v for k,v in raw.items()},'ss_total':total,
            **{k:v/total if total else None for k,v in raw.items()}}


def rotation(r):
    a=r[...,:3];b=r[...,3:];n=np.linalg.norm(a,axis=-1,keepdims=True)
    u=a/np.maximum(n,1e-8);res=b-(u*b).sum(axis=-1,keepdims=True)*u
    m=np.linalg.norm(res,axis=-1,keepdims=True);v=res/np.maximum(m,1e-8)
    valid=(n[...,0]>=1e-6)&(m[...,0]>=1e-6)
    return np.stack((u,v,np.cross(u,v)),axis=-1),valid


def physical_diff(a,b):
    xyz=np.stack((np.linalg.norm(a[:,0:3]-b[:,0:3],axis=1),np.linalg.norm(a[:,10:13]-b[:,10:13],axis=1)))
    r1=np.stack((a[:,3:9],a[:,13:19]));r2=np.stack((b[:,3:9],b[:,13:19]))
    R1,v1=rotation(r1);R2,v2=rotation(r2);valid=v1&v2
    angle=np.degrees(np.arccos(np.clip(((R1*R2).sum(axis=(-2,-1))-1)/2,-1,1)))
    grip=np.abs(a[:,[9,19]]-b[:,[9,19]])
    return dict(xyz_mean_m=float(xyz.mean()),xyz_max_m=float(xyz.max()),
                rotation_mean_deg=float(angle.mean()) if valid.all() else None,
                rotation_max_deg=float(angle.max()) if valid.all() else None,
                degenerate_rot6d=int((~v1).sum()+(~v2).sum()),
                gripper_mean_abs=float(grip.mean()),gripper_max_abs=float(grip.max()))


def means(rows):
    keys=set(rows[0])
    assert all(set(r)==keys for r in rows)
    return {k:float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None for k in sorted(keys)}


def summarize(observations,values,seed=20261014,repetitions=10000):
    groups={}
    for o,v in zip(observations,values):groups.setdefault(o['episode'],[]).append(v)
    episode_values={str(ep):means(v) for ep,v in sorted(groups.items())}
    keys=next(iter(episode_values.values())).keys();result={}
    for key in keys:
        x=[v[key] for v in episode_values.values()]
        if any(v is None for v in x):
            result[key]=dict(mean=None,ci95=None,undefined_episodes=sum(v is None for v in x));continue
        x=np.asarray(x);rng=np.random.default_rng(seed)
        boots=x[rng.integers(0,len(x),size=(repetitions,len(x)))].mean(axis=1)
        result[key]=dict(mean=float(x.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),undefined_episodes=0)
    return dict(n_episodes=len(groups),n_observations=len(values),episode_values=episode_values,metrics=result)


def analyze(prereg,run,split):
    reg=json.loads(prereg.read_text());digest=sha(prereg)
    assert (run/'PREREG_SHA256').read_text().strip()==digest
    obs=[o for o in reg['observations'] if o['split']==split];omap={o['obs_id']:o for o in obs}
    index=lines(run/'index.jsonl');expected={(o['obs_id'],v,a) for o in obs for v in reg['video_seeds'] for a in reg['action_seeds']}
    assert len(index)==len(expected) and {(r['obs_id'],r['video_seed'],r['action_seed']) for r in index}==expected
    indexed={r['path']:r for r in index};q1=[];code_versions=set();configs=set();source_hashes=[]
    for o in obs:
        assert sha(o['trace'])==o['trace_sha256'] and sha(o['record'])==o['record_sha256']
        grid=np.empty((4,4,32,20),np.float64)
        for iv,v in enumerate(reg['video_seeds']):
            for ia,a in enumerate(reg['action_seeds']):
                path=run/o['task']/o['obs_id']/'grid'/f'v{v}_a{a}.pt';row=indexed[str(path)]
                assert row['prereg_sha256']==digest and row['split']==split and sha(path)==row['sha256']
                t=torch.load(path,map_location='cpu',weights_only=False);m=t['meta']
                assert m['prereg_sha256']==digest and m['source_trace_sha256']==o['trace_sha256']
                assert m['video_seed']==v and m['action_seed']==a and not m['runtime_code']['dirty']
                assert len(t['steps'])==10 and np.array_equal(np.flatnonzero(arr(m['active_action_mask'])),reg['metrics']['active_dimensions'])
                grid[iv,ia]=arr(t['final']['actions_physical']);assert np.isfinite(grid[iv,ia]).all()
                code_versions.add(m['runtime_code']['commit']);configs.add(m['runtime_config_sha256'])
                source_hashes.append(row['sha256'])
        scatters={}
        for varying in ('video','action'):
            pairs=[]
            for fixed in range(4):
                for i,j in itertools.combinations(range(4),2):
                    a,b=(grid[i,fixed],grid[j,fixed]) if varying=='video' else (grid[fixed,i],grid[fixed,j])
                    pairs.append(physical_diff(a,b))
            scatters[varying]=dict(pair_count=len(pairs),mean_metrics=means(pairs),pairs=pairs)
        q1.append(dict(obs_id=o['obs_id'],task=o['task'],episode=o['episode'],relative_stage=o['relative_stage'],
                       ss=ss_decomposition(grid),scatter=scatters))
    steps=lines(run/'step_response.jsonl')
    def key(r):return r['obs_id'],tuple(r['receiver']),tuple(r['donor']),r['band'],r['pair_kind']
    expected_steps={(o['obs_id'],tuple(p['receiver']),tuple(p['donor']),b,p['pair_kind']) for o in obs for p in reg['pairs'] for b in o['sigma_steps']}
    assert len(steps)==len(expected_steps) and {key(r) for r in steps}==expected_steps
    for row in steps:
        o=omap[row['obs_id']];assert row['prereg_sha256']==digest and row['split']==split and row['episode']==o['episode'] and row['task']==o['task']
        assert row['step']==o['sigma_steps'][row['band']] and finite(row)
        for field in ('sigma_v','sigma_a','t_v','t_a'):assert row[field]==o['sigma_details'][row['band']][field]
        for field in ('swap_response','noop_vs_recorded','natural_donor_vs_receiver'):
            assert set(row[field])=={'l2','rms','max_abs','base_l2','base_rms','rel_l2'}
            assert all(x is None or x>=0 for x in row[field].values())
            assert (row[field]['rel_l2'] is None)==(row[field]['base_l2']==0)
    isolated=lines(run/'isolated_check.jsonl');wanted={c['obs_id'] for c in reg['isolated_checks']}&set(omap)
    assert {(r['obs_id'],r['band']) for r in isolated}=={(o,b) for o in wanted for b in ('high','mid','low')}
    assert len(isolated)==len(wanted)*3 and all(r['prereg_sha256']==digest and finite(r) for r in isolated)
    proxies=lines(run/'decode_proxy.jsonl');assert len(proxies)==len(expected)
    assert {(r['obs_id'],r['video_seed'],r['action_seed']) for r in proxies}==expected
    slots={'head':(slice(0,256),slice(0,320)),'left_wrist':(slice(256,384),slice(0,160)),'right_wrist':(slice(256,384),slice(160,320))}
    frame_count=0
    for r in proxies:
        o=omap[r['obs_id']];assert r['prereg_sha256']==digest and finite(r) and r['n_frames']==9
        folder=run/o['task']/o['obs_id']/'decoded'/f"v{r['video_seed']}_a{r['action_seed']}"
        assert len(list(folder.glob('frame_*.png')))==9 and (folder/'DONE').exists()
        frames=[np.asarray(Image.open(folder/f'frame_{k}.png').convert('RGB'),dtype=np.float32) for k in range(9)]
        assert all(x.shape==(384,320,3) for x in frames)
        for slot,(rs,cs) in slots.items():
            differences=[float(np.abs(frames[k+1][rs,cs]-frames[k][rs,cs]).mean()) for k in range(8)]
            assert differences==r['slot_framediff'][slot]['framediff_per_step']
            assert float(np.mean(differences))==r['slot_framediff'][slot]['framediff_mean']
        frame_count+=9
    bytask={}; q2_observations=[]
    for task in sorted({o['task'] for o in obs}):
        tobs=[o for o in obs if o['task']==task]; t1=[r for r in q1 if r['task']==task]
        q1values=[]
        for r in t1:
            q1values.append({**r['ss'],**{axis+'_'+k:v for axis in ('video','action') for k,v in r['scatter'][axis]['mean_metrics'].items()}})
        tss=summarize(tobs,q1values,reg['statistics']['bootstrap_seed'],reg['statistics']['bootstrap_repetitions'])
        bands={};stages={}
        for kind in ('primary_same_action','secondary_shifted_action'):
            for band in ('high','mid','low'):
                vals=[]
                for o in tobs:
                    group=[r for r in steps if r['obs_id']==o['obs_id'] and r['pair_kind']==kind and r['band']==band];assert len(group)==48
                    value={**means([r['swap_response'] for r in group]),
                           'nonzero_pair_fraction':float(np.mean([r['swap_response']['max_abs']>r['noop_vs_recorded']['max_abs'] for r in group]))}
                    vals.append(value)
                    q2_observations.append(dict(obs_id=o['obs_id'],task=task,episode=o['episode'],relative_stage=o['relative_stage'],pair_kind=kind,band=band,metrics=value))
                label=kind+'/'+band;bands[label]=summarize(tobs,vals,reg['statistics']['bootstrap_seed'],reg['statistics']['bootstrap_repetitions'])
                for stage in ('early','middle','late'):
                    selected=[(o,v) for o,v in zip(tobs,vals) if o['relative_stage']==stage]
                    if selected:stages[label+'/'+stage]=summarize([o for o,v in selected],[v for o,v in selected],reg['statistics']['bootstrap_seed'],reg['statistics']['bootstrap_repetitions'])
        bytask[task]=dict(Q1=tss,Q2=bands,Q2_by_relative_stage=stages)
    return dict(format='openwam-g1-analysis/1',split=split,prereg_sha256=digest,complete=True,
        input_hashes={n:sha(run/n) for n in ('index.jsonl','step_response.jsonl','isolated_check.jsonl','decode_proxy.jsonl')},
        grid_sha256_manifest=source_hashes,runtime_versions=sorted(code_versions),runtime_config_sha256=sorted(configs),
        audit=dict(grid_files=len(index),step_rows=len(steps),decoded_clips=len(proxies),frames=frame_count,
                   no_op_max_abs=max(r['noop_vs_recorded']['max_abs'] for r in steps),
                   isolated_max_abs=max((r['isolated_mask_response']['max_abs'] for r in isolated),default=None),
                   isolated_rows=len(isolated),holdout_loaded=False),
        tasks=bytask,Q1_observations=q1,Q2_observations=q2_observations,isolated_rows=isolated,
        limitations=reg['limitations'],analysis_script_sha256=sha(__file__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prereg',required=True,type=Path)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--split',required=True,choices=['explore','holdout'])
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    report=analyze(a.prereg.resolve(),a.run.resolve(),a.split)
    a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(a.out)
    print(json.dumps({'complete':report['complete'],'audit':report['audit']}))


if __name__=='__main__':main()
