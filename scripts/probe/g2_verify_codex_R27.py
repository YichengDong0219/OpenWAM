"""Independent direct architecture reruns of prereg V2; never calls Q5/Q6 wrappers."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
REPO=Path('/home/dongyicheng/OpenWAM');sys.path[:0]=[str(REPO),str(REPO/'scripts/probe')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
 a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
def steps(a,b,keys):return len(a)==len(b) and all(eq(x['i'],y['i']) and all(eq(x[k],y[k]) for k in keys) for x,y in zip(a,b))
def save(p,x):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['Q5','Q6'],required=True);ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
 prepath=REPO/'outputs/probe/g2_protocol_20261014/G2_PREREG.json';assert sha(prepath)=='be91026ca9bfb0ae8a72f1c7390ea8486952371e4661d2ecc4e82b4a0a44de39';pre=json.loads(prepath.read_text());obs={o['obs_id']:o for o in pre['observations']}
 scope=args.scope;root=REPO/'outputs/probe'/('g2_q5_explore_20261014_1750' if scope=='Q5' else 'g2_q6_explore_20261014_1750');rows={r['key']:r for r in map(json.loads,(root/'index.jsonl').read_text().splitlines())}
 def formal(k):
  r=rows[k];assert sha(r['path'])==r['sha256'];return load(r['path'])
 bindpath=REPO/'outputs/probe/g2_runtime_20261014'/('RUNTIME_BINDING_Q5_ALPHA_v2.json' if scope=='Q5' else 'RUNTIME_BINDING_Q6_IDM_v2.json');bind=json.loads(bindpath.read_text());assert bind['status']=='jointly_bound';assert sha(bind['manifest_path'])==bind['manifest_sha256'];man=json.loads(Path(bind['manifest_path']).read_text())
 # Actual runtime assertion uses previously cross-reviewed runtime comparator; generations do not use wrappers.
 from offline_runner import OfflineRunner
 from g2_common import check_runtime,check_imports_bound
 runner=OfflineRunner(man['checkpoint_dir'],man['deploy_config']);runtime=check_runtime(runner,man)
 runner.precompute_prompts([json.loads(Path(obs[v['obs_id']]['record']).read_text())['prompt'] for v in pre['verification']]);report=[];started=time.time()
 for v in pre['verification']:
  o=obs[v['obs_id']];oid=o['obs_id'];vr,ar=v['receiver'];vd,ad=v['donor'];rs=f'{vr}-{ar}';ds=f'{vd}-{ad}';checks={};traces={}
  if scope=='Q5':
   src=load(o['trace']);assert sha(o['trace'])==o['trace_sha256']
   def base(vs,aseed):return load(Path(pre['alpha_baseline_sources'][o['source_g1_split']]['root'])/o['task']/oid/'grid'/f'v{vs}_a{aseed}.pt')
   br,bd=base(vr,ar),base(vd,ad)
   calls=[('baseline',None,br),('identity',br,formal(f'identity|{oid}|{rs}')),('swap',bd,formal(f'swap|{oid}|{rs}|{ds}'))]
   for kind,replay,ref in calls:
    pt=runner.rerun(src,video_seed=vr,action_seed=ar,replay=replay)['probe_trace'];traces[kind]=pt
    checks[kind+'_steps_bitexact']=steps(pt['steps'],ref['steps'],('t_v','t_a','video_latents_in','action_latents_in','video_velocity','action_velocity'))
    checks[kind+'_final_bitexact']=all(eq(pt['final'][k],ref['final'][k]) for k in ('video_latents','action_latents','actions_normalized','actions_physical'))
    checks[kind+'_firstframe_preserved']=all(eq(s['video_latents_in'][:,:,:1],src['cond']['first_frame_latents']) for s in pt['steps'])
  else:
   src=formal(f'source|{oid}');br=formal(f'baseline|{oid}|{rs}');bd=formal(f'baseline|{oid}|{ds}')
   fresh_r=runner.rerun(src,video_seed=vr,action_seed=ar)['probe_trace'];fresh_d=runner.rerun(src,video_seed=vd,action_seed=ad)['probe_trace'];fresh_a=runner.rerun(src,video_seed=vr,action_seed=v['stage1_independence_action_seeds'][1])['probe_trace']
   for kind,pt,ref in [('baseline',fresh_r,br),('donor_baseline',fresh_d,bd)]:
    checks[kind+'_steps_bitexact']=steps(pt['steps'],ref['steps'],('t_a','action_latents_in','action_velocity'))
    checks[kind+'_stage1_cross_gpu_bitexact']=steps(pt['stage1_steps'],ref['stage1_steps'],('t_v','video_latents_in','video_velocity'))
    checks[kind+'_final_bitexact']=all(eq(pt['final'][k],ref['final'][k]) for k in ('stage1_future_latents','action_latents','actions_normalized','actions_physical'))
   checks['stage1_actionseed_independent']=steps(fresh_r['stage1_steps'],fresh_a['stage1_steps'],('t_v','video_latents_in','video_velocity')) and eq(fresh_r['final']['stage1_future_latents'],fresh_a['final']['stage1_future_latents'])
   # GT construction independent of OfflineRunner.gt_future_frames/encode_frames.
   from openwam.dataloader.transforms.multiview import assemble_multiview_layout
   rec=json.loads(Path(o['record']).read_text());assert sha(o['record'])==o['record_sha256'];post={p['offset']:p for p in rec['post_obs']};frames=[]
   for ps in [rec['pngs']]+[post[i]['pngs'] for i in range(4,33,4)]:
    cams={}
    for label,cam in [('head','head_camera'),('left','left_camera'),('right','right_camera')]:
     e=ps[label];assert sha(e['path'])==e['sha256'];cams[cam]=Image.open(e['path']).convert('RGB')
    frames.append(assemble_multiview_layout(cams,['head_camera','left_camera','right_camera'],384,320))
   pixels=torch.tensor(np.stack([np.asarray(f) for f in frames]),device=runner.arch.device,dtype=runner.arch.dtype);pixels=(pixels*(2/255)-1).permute(3,0,1,2).unsqueeze(0);vb=runner.arch.video_backbone
   gt=(vb.video_encoder.batch_encode(pixels) if vb.video_encoder is not None else vb.vae.batch_encode(pixels,device=pixels.device)).to(runner.arch.dtype).cpu()
   gref=formal(f'gt_future|{oid}');checks['gt_latents_independent_bitexact']=eq(gt,gref['latents']);checks['gt_frame0_exact']=eq(np.asarray(frames[0]),src['cond']['image'])
   calls=[('identity',fresh_r['final']['stage1_future_latents'],formal(f'identity|{oid}|{rs}')),('alt',fresh_d['final']['stage1_future_latents'],formal(f'alt|{oid}|{rs}|{ds}')),('gt',gt,formal(f'gt|{oid}|{rs}'))]
   for kind,future,ref in calls:
    pt=runner.rerun(src,video_seed=vr,action_seed=ar,cond_future_override=future)['probe_trace'];traces[kind]=pt
    checks[kind+'_steps_bitexact']=steps(pt['steps'],ref['steps'],('t_a','action_latents_in','action_velocity'))
    checks[kind+'_final_bitexact']=all(eq(pt['final'][k],ref['final'][k]) for k in ('stage1_future_latents','cond_video_latents_used','action_latents','actions_normalized','actions_physical'))
    checks[kind+'_firstframe_preserved']=eq(pt['final']['cond_video_latents_used'][:,:,:1],src['cond']['first_frame_latents'])
    checks[kind+'_receiver_stage1_preserved']=eq(pt['final']['stage1_future_latents'],fresh_r['final']['stage1_future_latents'])
   checks['identity_after_overrides_exact']=eq(runner.rerun(src,video_seed=vr,action_seed=ar,cond_future_override=fresh_r['final']['stage1_future_latents'])['probe_trace']['final']['actions_physical'],br['final']['actions_physical'])
  for name,pt in traces.items():torch.save(pt,out/f'{oid}_{name}.pt')
  assert all(checks.values()),checks
  report.append({'obs_id':oid,'checks':checks});save(out/'progress.json',report);print(oid,len(checks),'passed',flush=True)
 check_imports_bound(man);save(out/'summary.json',dict(scope=scope,all_passed=True,results=report,seconds=time.time()-started,script_sha256=sha(__file__),prereg_sha256=sha(prepath),binding_sha256=sha(bindpath),runtime=runtime,formal_index_sha256=sha(root/'index.jsonl')))
if __name__=='__main__':main()
