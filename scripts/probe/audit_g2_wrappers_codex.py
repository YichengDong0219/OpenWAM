"""Independent R25 fail-path audit; synthetic tensors only, no model or effect run."""
import argparse,copy,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import g2_q5 as A
import g2_q6 as B

def main():
    failures=[];passed=[]
    obs=dict(obs_id='synthetic',task='t',episode=0,record_sha256='a'*64)
    tensor=torch.zeros(1)
    def trace():
        return dict(schedule=[(1,1)],cond={},init={},active_action_mask=torch.ones(20,dtype=torch.bool),
                    steps=[dict(i=0,t_a=1.,action_latents_in=tensor,action_velocity=tensor)],
                    final=dict(stage1_future_latents=tensor,cond_video_latents_used=tensor,
                               action_latents=tensor,actions_normalized=np.zeros((1,20),np.float32),actions_physical=np.zeros((1,20),np.float32)))
    base=trace();bad=trace();bad['final']['actions_physical']=np.ones((1,20),np.float32)
    assert B.override_mismatches(base,bad)==['final.actions_physical'];passed.append('IDM identity helper detects wrong action')
    with tempfile.TemporaryDirectory(prefix='codex_g2_R25_') as td:
        root=Path(td)
        q=B.Q6.__new__(B.Q6);q.out=root/'q6';q.out.mkdir();q.index=q.out/'index.jsonl';q.done={};q.prereg_sha='b'*64;q.runtime_code={};q.runtime_config_sha256='c'*64;q.receipts={}
        class Runner:
            calls=0
            def rerun(self,*a,**kw):self.calls+=1;return {'probe_trace':copy.deepcopy(bad),'actions':bad['final']['actions_physical']}
        runner=Runner();q.runner=runner
        baselines={(1001,2001):dict(trace=base,path='synthetic-base.pt',sha256='d'*64)}
        try:q.identity(obs,{},dict(path='source.pt',sha256='e'*64),baselines)
        except RuntimeError:pass
        else:raise AssertionError('wrong identity did not stop')
        p=q.odir(obs)/'identity/r1001-2001.pt'
        assert p.exists() and q.index.exists() and q.reuse(p)
        calls=runner.calls;q.identity(obs,{},dict(path='source.pt',sha256='e'*64),baselines)
        assert runner.calls==calls
        failures.append({'case':'Q6_failed_identity_is_committed_then_skipped_on_resume','failed_identity_saved':True,'index_has_failed_identity':True,'second_identity_call_skips_validation':True})
        q5=A.Q5.__new__(A.Q5);q5.out=root/'q5';q5.out.mkdir();q5.done={};q5.prereg={'video_seeds':[1001],'action_seeds':[2001]};q5.source=lambda _:{}
        p=q5.out/'t/synthetic/identity/r1001-2001.pt';h=A.atomic_torch_save({'meta':{'kind':'swap','obs_id':'wrong'},'final':'not a baseline'},p);q5.done[str(p)]=h
        q5.baseline=lambda *a: (_ for _ in ()).throw(AssertionError('baseline should have been validated'))
        q5.identity(obs)
        failures.append({'case':'Q5_resume_accepts_valid_hash_with_wrong_identity_and_format','validation_skipped':True})
        conf=root/'frozen.yaml';conf.write_text('denoise_steps: 10\n');other=root/'unbound.yaml';other.write_text('denoise_steps: 9\n')
        man=root/'manifest.json';man.write_text(json.dumps(dict(prereg_sha256='b'*64,runtime_commit='synthetic',artifacts=[dict(path=str(conf),sha256=A.sha256_file(conf))])))
        freeze=root/'freeze.json';freeze.write_text(json.dumps(dict(status='frozen',prereg_sha256='b'*64)))
        accepted=[]
        for cls,scope in [(A.Q5,'Q5_alpha'),(B.Q6,'Q6_idm')]:
            bind=root/(scope+'.json');bind.write_text(json.dumps(dict(status='jointly_bound',scope=scope,signatures={'claude':'synthetic','codex':'synthetic'},manifest_path=str(man),manifest_sha256=A.sha256_file(man),prereg_sha256='f'*64)))
            obj=cls.__new__(cls);obj.prereg_sha='b'*64
            args=SimpleNamespace(freeze_receipt=str(freeze),runtime_binding=str(bind),config=str(other))
            obj._receipts(args);accepted.append(scope)
        failures.append({'case':'both_preflights_accept_unbound_config_and_mismatched_binding_prereg','scopes':accepted,'configs_have_different_contents':True})
    report=dict(format='openwam-codex-G2-wrapper-audit/1',synthetic_only=True,no_model_loaded=True,
                passed=passed,blocking_findings=failures,real_Q5_Q6_effects=False,
                reviewed_sha256={'g2_q5.py':A.sha256_file(A.__file__),'g2_q6.py':A.sha256_file(B.__file__)})
    out=Path(__import__('sys').argv[1]);out.parent.mkdir(parents=True,exist_ok=True);t=out.with_name(out.name+'.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(out)
    print(json.dumps(report))
if __name__=='__main__':main()
