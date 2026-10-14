"""Independent R26 synthetic acceptance; never imports implementer's test fixtures."""
import argparse,copy,contextlib,io,json,tempfile
from pathlib import Path
import numpy as np
import torch
import g2_common as C
import g2_q5 as A
import g2_q6 as B

def rejected(fn):
    try:fn()
    except (SystemExit,ValueError,KeyError,RuntimeError):return True
    return False
def main():
    passed=[]
    with tempfile.TemporaryDirectory(prefix='codex_R26_',dir=C.PROBE_ROOT) as td:
        root=Path(td);obs=dict(obs_id='o',task='t',episode=0,trace_sha256='a'*64,record_sha256='b'*64)
        z=torch.zeros(1);act=np.zeros((32,20),np.float32)
        def pt():
            return dict(schedule=[[1.,1.],[0.,0.]],active_action_mask=torch.ones(20,dtype=torch.bool),cond={},init={},
                        steps=[dict(i=0,t_v=1.,t_a=1.,video_latents_in=z,action_latents_in=z,video_velocity=z,action_velocity=z)],
                        final=dict(stage1_future_latents=z,cond_video_latents_used=z,video_latents=z,action_latents=z,actions_normalized=act,actions_physical=act))
        class Runner:
            bad=True;calls=0
            def rerun(self,*a,**kw):
                self.calls+=1;p=copy.deepcopy(pt())
                if self.bad:p['final']['actions_physical']=act+1
                return {'probe_trace':p,'actions':p['final']['actions_physical']}
        stamp={'synthetic':True}
        for kind,cls in [('Q5',A.Q5),('Q6',B.Q6)]:
            q=cls.__new__(cls);q.mode='smoke';q.prereg_sha='c'*64;q.binding_sha=None;q.runtime={'engine_config_sha256':'d'*64};q.runtime_code={};q.receipts={};q.runner=Runner()
            q.ledger=C.Ledger(root/kind,stamp)
            base={'trace':pt(),'ref':{'path':'bound-baseline.pt','sha256':'e'*64}}
            if kind=='Q5':
                q.source=lambda _: {'cond':{}}
                q.baseline=lambda *a: base
                call=lambda:q.unit(obs,'identity','identity|o|1001-2001',q.ledger.out/'id.pt',1001,2001,1001,2001)
            else:
                q.ledger.commit('source|o',q.ledger.out/'src.pt',{'synthetic':True})
                call=lambda:q.override_unit(obs,{},'identity','identity|o|1001-2001',q.ledger.out/'id.pt',base,1001,2001,z,[1001,2001],{})
            assert rejected(call)
            assert 'identity|o|1001-2001' not in q.ledger.rows and not (q.ledger.out/'id.pt').exists()
            assert q.ledger.failures.exists();passed.append(kind+' failed identity has journal but no committed output')
            q.ledger=C.Ledger(q.ledger.out,stamp);assert rejected(call);passed.append(kind+' failed identity rejected after fresh ledger restart')
            q.runner.bad=False;call();before=q.runner.calls
            q.ledger=C.Ledger(q.ledger.out,stamp);call();assert q.runner.calls==before;passed.append(kind+' valid resume checks identity without new generation')
            row=q.ledger.rows['identity|o|1001-2001'];original=torch.load(row['path'],weights_only=False)
            def forge(obj):
                torch.save(obj,row['path']);row['sha256']=C.sha256_file(row['path']);q.ledger.index.write_text(''.join(json.dumps(r)+'\n' for r in q.ledger.rows.values()));q.ledger=C.Ledger(q.ledger.out,stamp)
            bad=copy.deepcopy(original);bad['final']['actions_physical']=act+1;forge(bad);assert rejected(call);passed.append(kind+' correct SHA but corrupt identity tensors rejected')
            bad=copy.deepcopy(original);bad['meta']['obs_id']='foreign';forge(bad);assert rejected(call);passed.append(kind+' correct SHA but wrong observation rejected')
            bad=copy.deepcopy(original);bad['meta']['receiver_baseline']['sha256']='f'*64;forge(bad);assert rejected(call);passed.append(kind+' wrong baseline reference rejected')
            forge(original);assert rejected(lambda:q.ledger.require_exact([]));q.ledger.require_exact(q.ledger.rows);passed.append(kind+' exact key set rejects extra units')
        for name in ('missing_stamp','duplicate_key','orphan'):
            directory=root/name;l=C.Ledger(directory,stamp);l.commit('k',directory/'x.pt',{'x':1})
            if name=='missing_stamp':l.stamp_path.unlink()
            elif name=='duplicate_key':l.index.write_text(l.index.read_text()*2)
            else:(directory/'orphan.pt').write_bytes(b'orphan')
            assert rejected(lambda:C.Ledger(directory,stamp));passed.append(name+' ledger rejected')
        cfg=C.REPO/'configs/deploy_probe.yaml';other=root/'other.yaml';other.write_text(cfg.read_text().replace('denoise_steps: 10','denoise_steps: 9'))
        prereg=root/'pre.json';prereg.write_text(json.dumps({'format':'openwam-g2-prereg/1','observations':[],'gt_eligibility':{'eligible_explore_obs_ids':[]},'provenance':{'g1_prereg':{'sha256':'a'*64}},'alpha_baseline_sources':{}}))
        freeze=root/'freeze.json';freeze.write_text(json.dumps({'status':'frozen','prereg_sha256':C.sha256_file(prereg)}))
        review=root/'review.json';review.write_text('{}')
        for name,cls,ckpt,scope in [('Q5',A.Q5,A.ALPHA_CKPT,'Q5_alpha'),('Q6',B.Q6,B.IDM_CKPT,'Q6_idm')]:
            man=dict(format=C.MANIFEST_FORMAT,scope=scope,prereg_sha256=C.sha256_file(prereg),runtime_commit='synthetic',checkpoint_dir=str(ckpt.resolve()),deploy_config=str(cfg.resolve()),resolved_config_sha256=C.cfg_sha(C.resolve_config(str(cfg),str(ckpt))),artifacts=[{'path':str(cfg.resolve()),'sha256':C.sha256_file(cfg)}])
            mp=root/(name+'_man.json');mp.write_text(json.dumps(man))
            bind=dict(format=C.BINDING_FORMAT,scope=scope,status='jointly_bound',prereg_sha256=C.sha256_file(prereg),manifest_path=str(mp),manifest_sha256=C.sha256_file(mp),signatures={'claude':'synthetic','codex':'synthetic'},reviews=[{'path':str(review),'sha256':C.sha256_file(review)}])
            bp=root/(name+'_bind.json')
            def factory(*a):raise AssertionError('model factory reached')
            for variation in ('unbound_config','wrong_prereg','resolved_config','missing_signature'):
                b=copy.deepcopy(bind);m=copy.deepcopy(man)
                if variation=='wrong_prereg':b['prereg_sha256']='0'*64
                if variation=='missing_signature':b['signatures']['codex']=None
                if variation=='resolved_config':m['resolved_config_sha256']='0'*64
                mp.write_text(json.dumps(m));b['manifest_sha256']=C.sha256_file(mp);bp.write_text(json.dumps(b));dest=root/(name+variation)
                args=argparse.Namespace(prereg=str(prereg),freeze_receipt=str(freeze),runtime_binding=str(bp),out=str(dest),config=str(other if variation=='unbound_config' else cfg),n_obs=1)
                with contextlib.redirect_stdout(io.StringIO()):assert rejected(lambda:cls(args,'run',runner_factory=factory))
                assert not dest.exists();passed.append(name+' '+variation+' rejected before model and output')
            mp.write_text(json.dumps(man));bp.write_text(json.dumps(bind));args.config=str(cfg);args.runtime_binding=str(bp);args.out=str(root/(name+'positive'))
            with contextlib.redirect_stdout(io.StringIO()):
                try:cls(args,'run',runner_factory=factory)
                except AssertionError as e:assert str(e)=='model factory reached'
                else:raise AssertionError('valid control did not reach model factory')
            passed.append(name+' valid preflight reaches factory (positive control)')
    result=dict(format='openwam-codex-G2-v2-synthetic-audit/1',passed=passed,blocking_findings=[],no_model_loaded=True,
                source_sha256={Path(m.__file__).name:C.sha256_file(m.__file__) for m in (C,A,B)},script_sha256=C.sha256_file(__file__))
    import sys
    out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True);t=out.with_name(out.name+'.tmp');t.write_text(json.dumps(result,indent=2)+'\n');t.replace(out);print(json.dumps({'checks':len(passed),'blocking_findings':0}))
if __name__=='__main__':main()
