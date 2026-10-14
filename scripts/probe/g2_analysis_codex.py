"""G2 A3: independent raw-file audit, physical metrics and episode bootstrap.
No imports from g2 wrappers, g1_metrics, or other analysis implementations.
Units metres internally. Nulls only permitted for registered direction/rotation cases.
"""
import argparse,collections,datetime,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
PRE_SHA='be91026ca9bfb0ae8a72f1c7390ea8486952371e4661d2ecc4e82b4a0a44de39'
FREEZE_SHA='626288e15018202756b8c6f468e1d87004a60a2d38500f902ee210aa0c82d3f6'
BIND={'Q5':'02f22b08ff9d8965357f8f01fa370074bcb60c04c3bad61d83f235ce1763dab0','Q6':'efd9dfdf129d68299a5342348d576e6fb80897b0e93984fbb54a9700f7f45b33'}
INDEX={'Q5':'dcac4ffcd7cc2d989fb14e27793aecf28e55aa91e538a6b6931cd2afd6ad5885','Q6':'50181f9cdd2c52c607fd907b958516d12924796bafd9bb778664b421fea31a06'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def eq(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
def step_eq(a,b,keys):return len(a)==len(b) and all(eq(s['i'],t['i']) and all(eq(s[k],t[k]) for k in keys) for s,t in zip(a,b))
def xyz(a):return np.stack([a[:,0:3],a[:,10:13]],axis=1)
def rotation(a):
 r=np.stack([a[:,3:9],a[:,13:19]],axis=1);x,y=r[...,:3],r[...,3:];nx=np.linalg.norm(x,axis=-1);b1=x/np.maximum(nx[...,None],1e-8);yo=y-(b1*y).sum(-1,keepdims=True)*b1;ny=np.linalg.norm(yo,axis=-1);b2=yo/np.maximum(ny[...,None],1e-8);b3=np.cross(b1,b2)
 return np.stack([b1,b2,b3],axis=-1),(nx<1e-6)|(ny<1e-6)
def physical(a,b):
 a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64);assert a.shape==b.shape==(32,20) and np.isfinite(a).all() and np.isfinite(b).all()
 dist=np.linalg.norm(xyz(a)-xyz(b),axis=-1);grip=np.abs(a[:,[9,19]]-b[:,[9,19]]);ra,ia=rotation(a);rb,ib=rotation(b);valid=~(ia|ib);c=(np.einsum('...ij,...ij->...',ra,rb)-1)/2;angle=np.rad2deg(np.arccos(np.clip(c,-1,1)))[valid]
 return {'D_xyz_m':float(dist.mean()),'max_xyz_m':float(dist.max()),'rotation_mean_deg':float(angle.mean()) if len(angle) else None,'rotation_max_deg':float(angle.max()) if len(angle) else None,'rotation_invalid_A':int(ia.sum()),'rotation_invalid_B':int(ib.sum()),'rotation_valid_comparisons':int(valid.sum()),'gripper_mean_abs':float(grip.mean()),'gripper_max_abs':float(grip.max())}
def direction(out,receiver,donor):
 delta=(xyz(out)-xyz(receiver)).reshape(-1);d=(xyz(donor)-xyz(receiver)).reshape(-1);nd=np.linalg.norm(d);dn=np.linalg.norm(delta);rd=float(nd/8);result={'r_d_m':rd,'eligible':rd>=.001}
 if not result['eligible']:return dict(result,p=None,s_m=None,r_perp_m=None,cosine=None,positive_p=None,negative_p=None,zero_p=None)
 p=float(np.dot(delta,d)/np.dot(d,d));s=float(np.dot(delta,d)/nd/8);perp=float(np.linalg.norm(delta-p*d)/8);cos=float(np.dot(delta,d)/dn/nd) if dn else None
 return dict(result,p=p,s_m=s,r_perp_m=perp,cosine=cos,positive_p=float(p>0),negative_p=float(p<0),zero_p=float(p==0))
def hierarchy(rows,metrics,observations):
 by=collections.defaultdict(list)
 for r in rows:by[r['obs_id']].append(r)
 obsout={}
 for oid,rr in sorted(by.items()):
  vals={}
  for m in metrics:
   vs=[r['metrics'][m] for r in rr if r['metrics'][m] is not None]
   assert all(np.isfinite(v) for v in vs);vals[m]=float(np.mean(vs,dtype=np.float64)) if vs else None
  obsout[oid]={'n_rows':len(rr),'valid_rows_by_metric':{m:sum(r['metrics'][m] is not None for r in rr) for m in metrics},'metrics':vals}
 tasks={}
 for task in sorted({o['task'] for o in observations}):
  oo=[o for o in observations if o['task']==task];eps=sorted({o['episode'] for o in oo});epout={}
  for ep in eps:
   ids=[o['obs_id'] for o in oo if o['episode']==ep];epout[str(ep)]={}
   for m in metrics:
    vs=[obsout[i]['metrics'][m] for i in ids if i in obsout and obsout[i]['metrics'][m] is not None];epout[str(ep)][m]={'n_obs':len(vs),'mean':float(np.mean(vs)) if vs else None}
  aggregate={}
  for m in metrics:
   vals=np.array([epout[str(e)][m]['mean'] for e in eps if epout[str(e)][m]['mean'] is not None],dtype=np.float64)
   if len(vals):
    # Reset same prereg RNG for each task/metric. Same eligible episode set uses same resamples.
    sampled=np.random.default_rng(20261018).integers(0,len(vals),size=(10000,len(vals)));ci=np.quantile(vals[sampled].mean(1),[.025,.975],method='linear');mean=float(vals.mean());ci=ci.tolist()
   else:mean=None;ci=[None,None]
   aggregate[m]={'mean':mean,'ci95':ci,'n_eligible_episodes':len(vals),'n_total_episodes':len(eps),'empty_episodes':[e for e in eps if epout[str(e)][m]['mean'] is None]}
  tasks[task]={'n_rows':sum(len(by[o['obs_id']]) for o in oo),'n_observations_with_rows':sum(o['obs_id'] in by for o in oo),'episode':epout,'metrics':aggregate}
 return {'observations':obsout,'tasks':tasks}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',default='/home/dongyicheng/OpenWAM');ap.add_argument('--out',required=True);args=ap.parse_args();repo=Path(args.repo);out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
 prepath=repo/'outputs/probe/g2_protocol_20261014/G2_PREREG.json';assert sha(prepath)==PRE_SHA;pre=json.loads(prepath.read_text());fp=prepath.with_name('G2_FREEZE_RECEIPT.json');assert sha(fp)==FREEZE_SHA;fr=json.loads(fp.read_text());assert fr['status']=='frozen' and fr['prereg_sha256']==PRE_SHA
 write(out/'START.json',{'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(__file__),'prereg_sha256':PRE_SHA,'registered_metrics':pre['metrics'],'registered_statistics':pre['statistics'],'bootstrap_rng_detail':'reset default_rng(20261018) each task/metric; ordered eligible episodes; exact subset replay'})
 seeds=[(v,a) for v in pre['video_seeds'] for a in pre['action_seeds']];gtids=set(pre['gt_eligibility']['eligible_explore_obs_ids']);pairs=pre['pairs'];observations=pre['observations'];roots={s:repo/'outputs/probe'/f'g2_{s.lower()}_explore_20261014_1750' for s in ['Q5','Q6']}
 indexes={};audits={};bindings={};g1idx={};cache={}
 for s,root in roots.items():
  assert sha(root/'index.jsonl')==INDEX[s];data=[json.loads(x) for x in (root/'index.jsonl').read_text().splitlines()];index={r['key']:r for r in data};assert len(index)==len(data)
  expected=set()
  for o in observations:
   oid=o['obs_id']
   if s=='Q6':expected.add('source|'+oid)
   if s=='Q6' and oid in gtids:expected.add('gt_future|'+oid)
   for v,a in seeds:
    expected.add(f'identity|{oid}|{v}-{a}')
    if s=='Q6':expected.add(f'baseline|{oid}|{v}-{a}')
    if s=='Q6' and oid in gtids:expected.add(f'gt|{oid}|{v}-{a}')
   for p in pairs:expected.add(f"{'swap' if s=='Q5' else 'alt'}|{oid}|{p['receiver'][0]}-{p['receiver'][1]}|{p['donor'][0]}-{p['donor'][1]}")
  assert set(index)==expected;paths={Path(r['path']).resolve() for r in data};assert len(paths)==len(data);assert all(p.is_relative_to(root.resolve()) for p in paths)
  files={p.resolve() for p in root.rglob('*') if p.is_file()};assert files==paths|{(root/'index.jsonl').resolve(),(root/'RUN_STAMP.json').resolve()};assert not (root/'failures.jsonl').exists()
  stamp=json.loads((root/'RUN_STAMP.json').read_text());assert stamp=={'prereg_sha256':PRE_SHA,'mode':'run','runtime_binding_sha256':BIND[s]}
  bpath=repo/'outputs/probe/g2_runtime_20261014'/('RUNTIME_BINDING_Q5_ALPHA_v2.json' if s=='Q5' else 'RUNTIME_BINDING_Q6_IDM_v2.json');assert sha(bpath)==BIND[s];binding=json.loads(bpath.read_text());assert binding['status']=='jointly_bound';assert sha(binding['manifest_path'])==binding['manifest_sha256'];bindings[s]=json.loads(Path(binding['manifest_path']).read_text())
  for r in data:assert sha(r['path'])==r['sha256']
  indexes[s]=index;audits[s]={'n_units':len(data),'counts':dict(collections.Counter(r['kind'] for r in data)),'index_sha256':sha(root/'index.jsonl'),'stamp_sha256':sha(root/'RUN_STAMP.json'),'exact_keyset':True,'all_unit_sha_match':True,'no_orphans_duplicates_failures':True}
 for split,src in pre['alpha_baseline_sources'].items():
  p=Path(src['root'])/'index.jsonl';assert sha(p)==src['index_sha256']
  for r in map(json.loads,p.read_text().splitlines()):
   if r['kind']=='grid':g1idx[r['path']]=r['sha256']
 def get(s,key):
  if (s,key) not in cache:cache[s,key]=torch.load(indexes[s][key]['path'],map_location='cpu',weights_only=False)
  return cache[s,key]
 def ref(s,key):return {k:indexes[s][key][k] for k in ['path','sha256']}
 def g1(o,v,a):
  p=Path(pre['alpha_baseline_sources'][o['source_g1_split']]['root'])/o['task']/o['obs_id']/'grid'/f'v{v}_a{a}.pt';assert sha(p)==g1idx[str(p)];tr=torch.load(p,map_location='cpu',weights_only=False);assert tr['meta']['source_trace_sha256']==o['trace_sha256'];assert tr['meta']['prereg_sha256']==pre['provenance']['g1_prereg']['sha256'];return tr,{'path':str(p),'sha256':g1idx[str(p)]}
 def check(s,tr,o,kind,extra):
  m=tr['meta'];expected=dict(format=f'openwam-g2-{s.lower()}/2',kind=kind,prereg_sha256=PRE_SHA,runtime_binding_sha256=BIND[s],obs_id=o['obs_id'],task=o['task'],episode=o['episode'],**extra)
  for k,v in expected.items():assert m[k]==v,(s,o['obs_id'],kind,k)
  if kind!='gt_future':
   rt=bindings[s]['runtime_description'];assert [[float(x),float(y)] for x,y in m['schedule']]==rt['schedule'];assert m['runtime_engine_config_sha256']==rt['engine_config_sha256'];assert m['runtime_code']['commit'].startswith('f01665b') and not m['runtime_code']['dirty'];assert m['receipts']['freeze_receipt']['sha256']==FREEZE_SHA
   assert tr['init']['action_noise'].shape[1]==32;
   for st in tr['steps']:
    for value in st.values():
     if torch.is_tensor(value):assert torch.isfinite(value).all()
   assert len(tr['steps'])==10;assert [t['i'] for t in tr['steps']]==list(range(10));assert eq(tr['final']['actions_returned'],tr['final']['actions_physical']);actions=tr['final']['actions_physical'];assert np.shape(actions)==(32,20) and np.isfinite(actions).all()
  return tr
 def row(group,o,key,m):return {'group':group,'obs_id':o['obs_id'],'task':o['task'],'episode':o['episode'],'stage':o.get('relative_stage'),'key':key,'metrics':m}
 rows=[];engineering=collections.Counter();coverage=[]
 for o in observations:
  cache.clear();oid=o['obs_id'];assert sha(o['record'])==o['record_sha256'] and sha(o['trace'])==o['trace_sha256'];rec=json.loads(Path(o['record']).read_text());source=torch.load(o['trace'],map_location='cpu',weights_only=False);qsrc=get('Q6','source|'+oid)
  assert qsrc['meta']['deploy_config_sha256']==bindings['Q6']['runtime_description']['engine_config_sha256'];assert [[float(x),float(y)] for x,y in qsrc['meta']['schedule']]==bindings['Q6']['runtime_description']['schedule'];assert qsrc['cond']['prompt']==rec['prompt'];assert qsrc['meta']['video_seed']==1001 and qsrc['meta']['action_seed']==2001
  a1={};r1={};a2={};r2={}
  for v,a in seeds:
   a1[v,a],r1[v,a]=g1(o,v,a);key=f'baseline|{oid}|{v}-{a}';a2[v,a]=check('Q6',get('Q6',key),o,'baseline',dict(record_sha256=o['record_sha256'],video_seed=v,action_seed=a,donor=None,source=ref('Q6','source|'+oid)));r2[v,a]=ref('Q6',key)
   for s,base,refs,sk,fk in [('Q5',a1,r1,('t_v','t_a','video_latents_in','action_latents_in','video_velocity','action_velocity'),('video_latents','action_latents','actions_normalized','actions_physical')),('Q6',a2,r2,('t_a','action_latents_in','action_velocity'),('action_latents','actions_normalized','actions_physical','cond_video_latents_used'))]:
    key=f'identity|{oid}|{v}-{a}';extra=dict(video_seed=v,action_seed=a,donor=[v,a],receiver_baseline=refs[v,a]);extra.update(dict(source_trace_sha256=o['trace_sha256'],donor_baseline=refs[v,a]) if s=='Q5' else dict(record_sha256=o['record_sha256'],source=ref('Q6','source|'+oid)))
    tr=check(s,get(s,key),o,'identity',extra);assert step_eq(tr['steps'],base[v,a]['steps'],sk);assert all(eq(tr['final'][k],base[v,a]['final'][k]) for k in fk);engineering[s+'_identity_bitexact']+=1
   if (v,a)==(1001,2001):assert eq(a2[v,a]['final']['actions_physical'],qsrc['final']['actions_physical']);engineering['IDM_source_baseline_bitexact']+=1
  for v in pre['video_seeds']:
   for a in pre['action_seeds'][1:]:assert step_eq(a2[v,2001]['stage1_steps'],a2[v,a]['stage1_steps'],('t_v','video_latents_in','video_velocity')) and eq(a2[v,2001]['final']['stage1_future_latents'],a2[v,a]['final']['stage1_future_latents']);engineering['IDM_stage1_actionseed_comparisons']+=1
  for s,base,refs,kind in [('Q5',a1,r1,'swap'),('Q6',a2,r2,'alt')]:
   for p in pairs:
    rv,ra=p['receiver'];dv,da=p['donor'];key=f'{kind}|{oid}|{rv}-{ra}|{dv}-{da}';extra=dict(video_seed=rv,action_seed=ra,donor=[dv,da],receiver_baseline=refs[rv,ra],donor_baseline=refs[dv,da]);extra.update(dict(source_trace_sha256=o['trace_sha256']) if s=='Q5' else dict(record_sha256=o['record_sha256'],source=ref('Q6','source|'+oid)))
    tr=check(s,get(s,key),o,kind,extra);br,bd=base[rv,ra],base[dv,da];outa=np.asarray(tr['final']['actions_physical'],np.float64);receiver=np.asarray(br['final']['actions_physical'],np.float64);donor=np.asarray(bd['final']['actions_physical'],np.float64)
    assert eq(tr['init']['action_noise'],br['init']['action_noise'])
    if s=='Q5':
     for st,dt in zip(tr['steps'],bd['steps']):assert eq(st['video_latents_in'][:,:,1:],dt['video_latents_in'][:,:,1:]) and eq(st['video_latents_in'][:,:,:1],source['cond']['first_frame_latents'])
    else:
     ff=qsrc['cond']['first_frame_latents'];assert eq(tr['final']['stage1_future_latents'],br['final']['stage1_future_latents']);assert eq(tr['final']['cond_video_latents_used'][:,:,1:],bd['final']['stage1_future_latents'][:,:,1:]);assert eq(tr['final']['cond_video_latents_used'][:,:,:1],ff)
    engineering[s+'_intervention_controls']+=1;engineering[s+'_posthoc_donor_final_all20_bitexact']+=int(eq(tr['final']['actions_physical'],bd['final']['actions_physical']));engineering[s+'_posthoc_donor_action_steps_bitexact']+=int(step_eq(tr['steps'],bd['steps'],('t_a','action_latents_in','action_velocity')));rows.append(row(s+'_effect',o,key,physical(outa,receiver)));rows.append(row(s+'_natural',o,key,physical(donor,receiver)));dire=direction(outa,receiver,donor);rows.append(row(s+'_direction_coverage',o,key,{'eligible_fraction':float(dire['eligible'])}))
    if dire.pop('eligible'):rows.append(row(s+'_direction',o,key,dire))
  for p in pre['idm_action_scatter_pairs']:
   v=p['video_seed'];a,b=p['action_seed_pair'];rows.append(row('Q6_action_scatter',o,f'{v}-{a}-{b}',physical(a2[v,a]['final']['actions_physical'],a2[v,b]['final']['actions_physical'])))
  if oid in gtids:
   assert rec['executed_count']==32 and len(rec['actions'])==32 and len(rec['actions_executed'])==32;assert [e['offset'] for e in rec['actions_executed']]==list(range(1,33));assert all(e['count_delta']==1 and not e['exception'] for e in rec['actions_executed']);assert {p['offset'] for p in rec['post_obs']}==set(range(4,33,4));assert len(o['pngs'])==27
   used={(e['path'],e['sha256']) for e in rec['pngs'].values()}|{(e['path'],e['sha256']) for post in rec['post_obs'] for e in post['pngs'].values()};assert used=={(e['path'],e['sha256']) for e in o['pngs']}
   for p in o['pngs']:assert sha(p['path'])==p['sha256']
   g=check('Q6',get('Q6','gt_future|'+oid),o,'gt_future',dict(record_sha256=o['record_sha256'],source=ref('Q6','source|'+oid)));assert eq(g['latents'][:,:,:1],qsrc['cond']['first_frame_latents'])
   for v,a in seeds:
    key=f'gt|{oid}|{v}-{a}';tr=check('Q6',get('Q6',key),o,'gt',dict(record_sha256=o['record_sha256'],video_seed=v,action_seed=a,donor='gt',source=ref('Q6','source|'+oid),receiver_baseline=r2[v,a],gt_future=ref('Q6','gt_future|'+oid)))
    br=a2[v,a];assert eq(tr['final']['stage1_future_latents'],br['final']['stage1_future_latents']);assert eq(tr['final']['cond_video_latents_used'][:,:,1:],g['latents'][:,:,1:]);assert eq(tr['final']['cond_video_latents_used'][:,:,:1],qsrc['cond']['first_frame_latents']);assert eq(tr['init']['action_noise'],br['init']['action_noise'])
    ga,ba=tr['final']['actions_physical'],br['final']['actions_physical'];mg,ms=physical(ga,rec['actions']),physical(ba,rec['actions']);m=physical(ga,ba);m.update(E_gt_m=mg['D_xyz_m'],E_self_m=ms['D_xyz_m'],delta_E_m=mg['D_xyz_m']-ms['D_xyz_m']);rows.append(row('GT_command_consistency',o,key,m));engineering['GT_intervention_controls']+=1
  coverage.append({'obs_id':oid,'task':o['task'],'episode':o['episode'],'eligible_GT':oid in gtids,'executed_count':rec['executed_count'],'planner_failures':sum(p.get('status')!='Success' for e in rec['actions_executed'] for p in e.get('planner',[]))})
  print(oid,'audit and metrics complete',flush=True)
 counts=collections.Counter(r['group'] for r in rows);assert counts['Q5_effect']==counts['Q5_natural']==counts['Q6_effect']==counts['Q6_natural']==1440;assert counts['Q6_action_scatter']==720 and counts['GT_command_consistency']==416
 write(out/'audit.json',{'status':'passed','audits':audits,'engineering_counts':dict(engineering),'source_record_coverage':coverage,'own_script_sha256':sha(__file__),'prereg_sha256':PRE_SHA,'no_missing_expected_rows':True})
 temp=out/'rows.jsonl.tmp';temp.write_text(''.join(json.dumps(r,allow_nan=False,sort_keys=True)+'\n' for r in rows));os.replace(temp,out/'rows.jsonl')
 summary={}
 for group in sorted(counts):
  rr=[r for r in rows if r['group']==group];metrics=list(rr[0]['metrics']);assert all(list(r['metrics'])==metrics for r in rr);summary[group]=hierarchy(rr,metrics,observations)
 write(out/'analysis.json',{'format':'openwam-codex-g2-A3/1','status':'submitted_pending_claude_independent_review','split':'explore','all_G1_sources_now_explore':True,'prereg_sha256':PRE_SHA,'script_sha256':sha(__file__),'counts':dict(counts),'groups':summary,'statistics':pre['statistics'],'bootstrap_rng_detail':'reset per task/metric; sorted eligible episodes; default_rng(20261018)','first_metrics_parse':json.loads((out/'START.json').read_text())['started_utc'],'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows_sha256':sha(out/'rows.jsonl'),'audit_sha256':sha(out/'audit.json'),'units':'metres internally, report mm; max/rotation/degenerate/grip are pair metrics averaged through the same hierarchy','holdout_loaded':False})
 print('A3 submitted',dict(counts),flush=True)
if __name__=='__main__':main()
