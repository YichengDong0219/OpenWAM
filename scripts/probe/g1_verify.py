"""Codex independent GPU3 rerun of the preregistered G1 verification subset."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from PIL import Image
from offline_runner import OfflineRunner, load_trace


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def array(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def compare(a, b, errors, prefix=""):
    if isinstance(a, dict):
        if not isinstance(b, dict) or not set(a).issubset(b):
            errors.append(prefix+": missing dictionary fields"); return
        for k in a: compare(a[k], b[k], errors, prefix+"/"+k)
    elif isinstance(a, (list, tuple)):
        if len(a)!=len(b): errors.append(prefix+": length mismatch"); return
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,errors,prefix+f"/{i}")
    elif torch.is_tensor(a) or isinstance(a,np.ndarray):
        x,y=array(a),array(b)
        if x.shape!=y.shape or not np.array_equal(x,y):
            errors.append(prefix+": tensor differs")
    elif a != b:
        errors.append(prefix+": value differs")


def response(swapped, base, dims):
    x=array(swapped).astype(np.float64)[...,dims].ravel()
    y=array(base).astype(np.float64)[...,dims].ravel()
    d=x-y; n=float(np.sqrt(d@d)); b=float(np.sqrt(y@y))
    return dict(l2=n,rms=float(np.sqrt((d@d)/len(d))),max_abs=float(max(abs(d))),
                base_l2=b,base_rms=float(np.sqrt((y@y)/len(y))),rel_l2=n/b if b else None)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prereg',required=True,type=Path);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--ckpt',default='/home/dongyicheng/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full')
    args=p.parse_args(); reg=json.loads(args.prereg.read_text()); digest=sha(args.prereg)
    args.out.mkdir(parents=True,exist_ok=True)
    errors=[]; observations={o['obs_id']:o for o in reg['observations'] if o['split']=='explore'}
    index=[json.loads(l) for l in (args.run/'index.jsonl').read_text().splitlines()]
    expected={(o,v,a) for o in observations for v in reg['video_seeds'] for a in reg['action_seeds']}
    actual={(r['obs_id'],r['video_seed'],r['action_seed']) for r in index}
    assert actual==expected and len(index)==len(expected)==288
    for r in index:
        assert r['prereg_sha256']==digest and r['split']=='explore' and sha(r['path'])==r['sha256']
    rows=[json.loads(l) for l in (args.run/'step_response.jsonl').read_text().splitlines()]
    by_key={(r['obs_id'],tuple(r['receiver']),tuple(r['donor']),r['band']):r for r in rows}
    selected=[observations[c['obs_id']] for c in reg['isolated_checks']]
    for o in selected:
        assert sha(o['trace'])==o['trace_sha256'] and sha(o['record'])==o['record_sha256']
    runner=OfflineRunner(args.ckpt)
    sources={o['obs_id']:load_trace(o['trace']) for o in selected}
    runner.precompute_prompts([s['cond']['prompt'] for s in sources.values()])
    dims=reg['metrics']['active_dimensions']; details=[]
    vr,ar=reg['video_seeds'][0],reg['action_seeds'][0];vd=reg['video_seeds'][1]
    for o in selected:
        src=sources[o['obs_id']]; traces={}; reruns={}; before=len(errors)
        for v in (vr,vd):
            saved=torch.load(args.run/o['task']/o['obs_id']/'grid'/f'v{v}_a{ar}.pt',map_location='cpu',weights_only=False)
            assert saved['meta']['format']=='openwam-g1-generation/1'
            assert saved['meta']['prereg_sha256']==digest and saved['meta']['source_trace_sha256']==o['trace_sha256']
            assert saved['meta']['runtime_code']['dirty'] is False
            new=runner.rerun(src,video_seed=v,action_seed=ar,record=True);pt=new['probe_trace']
            for key in ('init','steps','final'): compare(pt[key],saved[key],errors,o['obs_id']+f'/v{v}/'+key)
            compare(np.asarray(new['actions']),saved['final']['actions_returned'],errors,o['obs_id']+'/returned_actions')
            traces[v]=saved;reruns[v]=new
            torch.save(pt,args.out/f"{o['obs_id']}_v{v}_a{ar}_independent.pt")
        rec,don=traces[vr],traces[vd];inputs=runner._inputs_shared(src);checks=[]
        for band,i in o['sigma_steps'].items():
            sr=rec['steps'][i];sd=don['steps'][i]
            _,base=runner.step_response(rec,i,inputs_shared=inputs)
            _,swap=runner.step_response(rec,i,video_latents=sd['video_latents_in'],inputs_shared=inputs)
            compare(base,sr['action_velocity'],errors,o['obs_id']+'/'+band+'/noop_tensor')
            recorded=by_key[(o['obs_id'],(vr,ar),(vd,ar),band)]
            fresh=response(swap,base,dims)
            for key,val in fresh.items():
                expected_val=recorded['swap_response'][key]
                if val is None or expected_val is None:
                    if val!=expected_val:errors.append(o['obs_id']+'/'+band+'/'+key)
                elif not np.isclose(val,expected_val,rtol=1e-12,atol=1e-12):
                    errors.append(o['obs_id']+'/'+band+'/'+key)
            with runner.attention_mask_mode('isolated'):
                _,b0=runner.step_response(rec,i,inputs_shared=inputs)
                _,b1=runner.step_response(rec,i,video_latents=sd['video_latents_in'],inputs_shared=inputs)
            iso=response(b1,b0,dims)
            if iso['max_abs']!=0:errors.append(o['obs_id']+'/'+band+'/isolated_nonzero')
            checks.append(dict(band=band,step=i,swap_response=fresh,isolated_response=iso))
            torch.save(dict(base=base,swap=swap,isolated_base=b0,isolated_swap=b1),args.out/f"{o['obs_id']}_{band}_responses.pt")
        frames=runner.arch.video_backbone.decode_video(reruns[vr]['probe_trace']['final']['video_latents'].to(runner.arch.device),tiled=True)
        if len(frames)!=9:errors.append(o['obs_id']+'/decode_count')
        for k,frame in enumerate(frames):
            stored=args.run/o['task']/o['obs_id']/'decoded'/f'v{vr}_a{ar}'/f'frame_{k}.png'
            compare(np.asarray(frame.convert('RGB')),np.asarray(Image.open(stored).convert('RGB')),errors,o['obs_id']+f'/decoded/{k}')
        details.append(dict(obs_id=o['obs_id'],passed=len(errors)==before,checks=checks,decoded_frames=len(frames)))
        print(json.dumps({'verified':o['obs_id'],'passed':len(errors)==before}),flush=True)
    report=dict(passed=not errors,errors=errors,prereg_sha256=digest,grid_hashes_checked=len(index),
                full_generations_rerun=6,single_step_pairs=9,decoded_frames_compared=27,details=details,
                runtime_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                scope='6 baselines and 9 local response pairs, not all 5184 pairs; no holdout loaded')
    path=args.out/'g1_verify.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
