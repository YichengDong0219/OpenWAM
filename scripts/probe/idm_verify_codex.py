"""Independent GPU3 V2, engineering invariants only; no Q6 effect estimates."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from offline_runner import OfflineRunner, sha256_file
from benchmarks.utils.action_conversion import eef20d_to_ee16d
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.model.video_backbone.wan import encode as E

def eq(x,y):
    if torch.is_tensor(x) and torch.is_tensor(y):return x.shape==y.shape and x.dtype==y.dtype and torch.equal(x,y)
    return np.array_equal(np.asarray(x),np.asarray(y))

def steps(a,b,keys):
    return len(a)==len(b) and all(x['i']==y['i'] and all(eq(x[k],y[k]) for k in keys) for x,y in zip(a,b))

def save(p,x):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--prereg',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
    repo=Path(__file__).resolve().parents[2];candidate=json.loads(Path(args.candidate).read_text());pr=json.loads(Path(args.prereg).read_text())
    assert sha256_file(args.prereg)=='be91026ca9bfb0ae8a72f1c7390ea8486952371e4661d2ecc4e82b4a0a44de39'
    ck=repo/candidate['checkpoint']['dir'];bindings={}
    for n,h in candidate['checkpoint']['files'].items():
        bindings[str(ck/n)]=sha256_file(ck/n);assert bindings[str(ck/n)]==h['sha256']
    for n,h in candidate['source']['code_sha256'].items():
        bindings[n]=sha256_file(repo/n);assert bindings[n]==h
    config=repo/candidate['deploy_config']['path'];assert sha256_file(config)==candidate['deploy_config']['sha256']
    refs=json.loads((repo/candidate['selfcheck']['summary']).read_text())
    assert sha256_file(repo/candidate['selfcheck']['summary'])==candidate['selfcheck']['sha256']
    byref={r['obs_id']:r for r in refs['results']};byid={r['obs_id']:r for r in pr['observations']}
    obs=[dict(byid[v['obs_id']],**{k:x for k,x in v.items() if k!='obs_id'}) for v in pr['verification']]
    stats=np.load(ck/'normalization_stats.npy',allow_pickle=True).item()
    start=time.time();runner=OfflineRunner(str(ck),str(config));runner.precompute_prompts([json.loads(Path(v['record']).read_text())['prompt'] for v in obs])
    vb=runner.arch.video_backbone;report=[]
    for v in obs:
        recpath=Path(v['record']);assert sha256_file(recpath)==v['record_sha256'];rec=json.loads(recpath.read_text())
        tracepath=Path(byref[v['obs_id']]['trace']['path']);assert sha256_file(tracepath)==byref[v['obs_id']]['trace']['sha256']
        trace=torch.load(tracepath,map_location='cpu',weights_only=False)
        def generate(**over):
            kw=runner.generate_kwargs(trace,video_seed=over.pop('video_seed',None),action_seed=over.pop('action_seed',None))
            return runner.arch.generate(**kw,probe=dict(record=True,**over))['probe_trace']
        a=generate();b=generate(action_seed=v['stage1_independence_action_seeds'][1]);d=generate(video_seed=v['donor'][0])
        own=a['final']['stage1_future_latents'];first=a['cond']['first_frame_latents'];nf=first.shape[2]
        switched=generate(cond_future_override=d['final']['stage1_future_latents']);identity=generate(cond_future_override=own)
        checks={
            'cross_gpu_actions_bitexact':eq(a['final']['actions_physical'],trace['final']['actions_physical']),
            'cross_gpu_stage2_bitexact':steps(a['steps'],trace['steps'],('t_a','action_latents_in','action_velocity')),
            'cross_gpu_stage1_bitexact':steps(a['stage1_steps'],trace['stage1_steps'],('t_v','video_latents_in','video_velocity')),
            'action_noise_changes':not eq(a['init']['action_noise'],b['init']['action_noise']),
            'stage1_action_seed_independent':steps(a['stage1_steps'],b['stage1_steps'],('t_v','video_latents_in','video_velocity')) and eq(own,b['final']['stage1_future_latents']),
            'donor_video_distinct':not eq(own,d['final']['stage1_future_latents']),
            'receiver_stage1_preserved':eq(switched['final']['stage1_future_latents'],own),
            'donor_future_exact':eq(switched['final']['cond_video_latents_used'][:,:,nf:],d['final']['stage1_future_latents'][:,:,nf:]),
            'first_frame_protected':eq(switched['final']['cond_video_latents_used'][:,:,:nf],first),
            'identity_after_swap_actions_bitexact':eq(identity['final']['actions_physical'],a['final']['actions_physical']),
            'identity_after_swap_steps_bitexact':steps(identity['steps'],a['steps'],('t_a','action_latents_in','action_velocity')),
        }
        for name,bad in [('dtype',own.float()),('shape',own[:,:,:-1])]:
            try:generate(cond_future_override=bad)
            except ValueError:checks[name+'_rejected']=True
            else:checks[name+'_rejected']=False
        poisoned=own.clone();poisoned[:,:,:nf]=17
        p=generate(cond_future_override=poisoned)
        checks['poisoned_first_frame_reset']=eq(p['final']['cond_video_latents_used'][:,:,:nf],first)
        assert rec['executed_count']==32 and rec['termination'] in ('none',None)
        offset={x['offset']:x for x in rec['post_obs']}
        pngsets=[rec['pngs']]+[offset[i]['pngs'] for i in range(4,33,4)]
        frames=[]
        for group in pngsets:
            cams={}
            for label,camera in [('head','head_camera'),('left','left_camera'),('right','right_camera')]:
                entry=group[label];assert sha256_file(entry['path'])==entry['sha256'];cams[camera]=Image.open(entry['path']).convert('RGB')
            frames.append(assemble_multiview_layout(cams,['head_camera','left_camera','right_camera'],384,320))
        helper=runner.gt_future_frames(rec)
        checks['all9_gt_canvases_match_training_layout']=all(eq(np.asarray(x),np.asarray(y)) for x,y in zip(frames,helper)) and len(helper)==9
        checks['gt_frame0_trace_matches']=eq(np.asarray(frames[0]),trace['cond']['image'])
        raw=torch.tensor(np.stack([np.asarray(f) for f in frames]).astype(np.float32),device=runner.arch.device,dtype=runner.arch.dtype)
        pixels=(raw*(2/255)-1).permute(3,0,1,2).unsqueeze(0)
        pathpixels=E.preprocess_video(frames,encoder=vb.video_encoder,dtype=runner.arch.dtype,device=runner.arch.device)
        checks['pixel_range_scaling_dtype_exact']=eq(pixels,pathpixels)
        gt=vb.video_encoder.batch_encode(pixels).to(runner.arch.dtype).cpu()
        checks['gt_encoding_helper_exact']=eq(gt,runner.encode_frames(helper).cpu())
        checks['gt_first_latent_exact']=eq(gt[:,:,:nf],first)
        checks['gt_latent_contract']=gt.shape==own.shape and gt.dtype==own.dtype
        rgt=generate(cond_future_override=gt)
        checks['gt_override_firstframe']=eq(rgt['final']['cond_video_latents_used'][:,:,:nf],first)
        checks['gt_override_future']=eq(rgt['final']['cond_video_latents_used'][:,:,nf:],gt[:,:,nf:])
        decoded=vb.video_encoder.decode(gt.to(runner.arch.device),tiled=True)
        checks['gt_reconstruction_finite9frames']=bool(torch.isfinite(decoded).all()) and tuple(decoded.shape)==(1,3,9,384,320)
        st=stats.get('eef_state',stats['eef']);mn=np.asarray(st['min'],np.float32);mx=np.asarray(st['max'],np.float32)
        expected=np.clip((np.asarray(rec['proprio'],np.float32)-mn)/np.maximum(mx-mn,1e-6)*2-1,-1,1)
        checks['independent_proprio_minmax_bf16']=eq(torch.as_tensor(expected).reshape(-1).to(torch.bfloat16),a['cond']['proprio'].reshape(-1))
        checks['normalizer_stats_match_file']=all(eq(runner.arch.normalizer.stats[k],stats['eef'][k]) for k in ('min','max'))
        lo=np.asarray(stats['eef']['min'],np.float32);hi=np.asarray(stats['eef']['max'],np.float32);scale=2/np.maximum(hi-lo,1e-6)
        inverse=a['final']['actions_normalized']/scale+(lo+np.maximum(hi-lo,1e-6)/2)
        checks['action_denormalization_exact']=eq(inverse,a['final']['actions_physical'])
        checks['active20_shape_mask']=tuple(a['final']['actions_physical'].shape)==(32,20) and bool(a['active_action_mask'].all()) and a['active_action_mask'].numel()==20
        commands=np.asarray(rec['actions'],np.float32)
        checks['all32_record_execution_conversions']=len(rec['actions_executed'])==32 and all(eq(eef20d_to_ee16d(commands[e['offset']-1]),e['env_action']) for e in rec['actions_executed'])
        report.append({'obs_id':v['obs_id'],'checks':checks,'source_trace_sha256':sha256_file(tracepath),
                       'record_sha256':sha256_file(recpath),'gt_latents_shape':list(gt.shape),'gt_reconstruction_shape':list(decoded.shape)})
        save(out/'progress.json',report);print(json.dumps({'obs':v['obs_id'],'passes':sum(checks.values()),'total':len(checks)}),flush=True)
    summary=dict(format='openwam-codex-idm-V2/1',scope='GPU3 independent engineering; no Q6 effects',
                 all_passed=all(all(x['checks'].values()) for x in report),results=report,bindings=bindings,
                 prereg_sha256=sha256_file(args.prereg),candidate_sha256=sha256_file(args.candidate),
                 own_script_sha256=sha256_file(__file__),seconds=time.time()-start,
                 runtime_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
    save(out/'summary.json',summary);print(json.dumps({'all_passed':summary['all_passed'],'seconds':summary['seconds']}),flush=True)

if __name__=='__main__':main()
