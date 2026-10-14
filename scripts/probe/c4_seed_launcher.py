"""C4: delegate to the unchanged G0 wrapper, substituting only registered expert start seed.
The G0 CLI exposes no start-seed flag. No expert validity, RNG or instruction logic is changed.
"""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmarks.robotwin import eval_policy_wrapper as W

def install_registered_start(module, prereg):
    W._install_manifest_original(module)
    frozen_manifest=module.eval_policy
    starts=prereg['holdout']['expert_seed_starts']
    def call(task_name,task_env,args,model,st_seed,test_num=100,video_size=None,instruction_type=None):
        assert test_num==3 and task_name in starts
        print(f'[C4] registered expert start task={task_name} seed={starts[task_name]} (default was {st_seed})',flush=True)
        return frozen_manifest(task_name,task_env,args,model,starts[task_name],test_num=test_num,video_size=video_size,instruction_type=instruction_type)
    module.eval_policy=call

def main():
    p=Path(os.environ['C4_PREREG']);assert hashlib.sha256(p.read_bytes()).hexdigest()=='be91026ca9bfb0ae8a72f1c7390ea8486952371e4661d2ecc4e82b4a0a44de39'
    pre=json.loads(p.read_text());W._install_manifest_original=W._install_manifest
    W._install_manifest=lambda module:install_registered_start(module,pre)
    return W.main()
if __name__=='__main__':raise SystemExit(main())
