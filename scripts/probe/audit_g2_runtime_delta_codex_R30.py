"""Independent full-byte v3 manifest audit and v2 dependency delta; no model loaded."""
import hashlib,json,os,sys
from pathlib import Path
R=Path('/home/dongyicheng/OpenWAM');D=R/'outputs/probe/g2_runtime_20261014'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()
report={}
for scope,stem,expected,changed in [('alpha','ALPHA_RUNTIME_MANIFEST_Q5','368aa38954794804acbf3325fdcfdb97e626479dd59d7f371f07567ece861578',['g2_common.py','g2_q5.py']),('IDM','IDM_RUNTIME_MANIFEST_Q6','d62e5e2f6d6b9c3bd8eb08d1711f2ffb977335dde631edc3214cd968de2e2efd',['g2_common.py','g2_q5.py'])]:
 old=json.loads((D/(stem+'_v3.json')).read_text());p=D/(stem+'_v4.json');assert sha(p)==expected;new=json.loads(p.read_text());o={a['path']:a for a in old['artifacts']};n={a['path']:a for a in new['artifacts']};assert len(n)==len(new['artifacts']) and set(o)==set(n)
 diff=[k for k in n if o[k]['sha256']!=n[k]['sha256']];assert sorted(Path(k).name for k in diff)==sorted(changed)
 assert old['runtime_description']==new['runtime_description'] and old['resolved_config_sha256']==new['resolved_config_sha256']
 for a in n.values():assert Path(a['path']).stat().st_size==a['bytes'] and sha(a['path'])==a['sha256']
 assert sha(new['resolved_config_snapshot'])==new['resolved_config_sha256']
 report[scope]={'manifest_sha256':expected,'n_artifacts':len(n),'all_bytes_hash_match':True,'exact_changed_artifacts':diff,'runtime_description_unchanged':True,'resolved_config_unchanged':True}
result={'format':'codex-g2-v4-runtime-byte-audit/1','scopes':report,'own_script_sha256':sha(__file__),'no_model_loaded':True,'acceptance':'byte audit only; byte audit; wrapper acceptance recorded separately'}
p=Path(sys.argv[1]);assert not p.exists();t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(result,indent=2)+'\n');os.replace(t,p);print(json.dumps(report))
