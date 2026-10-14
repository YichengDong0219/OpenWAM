"""Independent byte-level runtime manifest audit; no model loads or file deletion."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.root);out=Path(a.out);assert not out.exists()
    expected={'ALPHA_RUNTIME_MANIFEST_Q5.json':'b841c7413eac52ddd8407aa304b7ba0d187419d7fa044d93dbfef972ea714ce2','IDM_RUNTIME_MANIFEST_Q6.json':'4502137802b9f3a54ecd070e5648335f1c0568735949dab2aad06cdd18b2feda'}
    results=[];cache={};start=time.time()
    for name,digest in expected.items():
        p=root/'outputs/probe/g2_runtime_20261014'/name;assert sha(p)==digest;man=json.loads(p.read_text());issues=[];paths=set();nbytes=0
        for art in man['artifacts']:
            f=Path(art['path']);key=str(f)
            if key in paths:issues.append('duplicate path '+key)
            paths.add(key)
            if not f.is_file():issues.append('missing '+key);continue
            if f.stat().st_size!=art['bytes']:issues.append('size '+key)
            if key not in cache:cache[key]=sha(f)
            if cache[key]!=art['sha256']:issues.append('hash '+key)
            nbytes+=f.stat().st_size
        tracked=subprocess.check_output(['git','ls-files','openwam'],cwd=root,text=True).splitlines()
        missing=[f for f in tracked if f.endswith('.py') and str(root/f) not in paths]
        results.append(dict(name=name,sha256=digest,n_artifacts=len(man['artifacts']),verified_bytes=nbytes,issues=issues,missing_tracked_openwam_py=missing,
                            tokenizer_files=[f for f in paths if '/tokenizer/' in f],
                            missing_required_external_helpers=[f for f in ['benchmarks/utils/client.py','benchmarks/utils/__init__.py'] if str(root/f) not in paths],
                            resolved_config_artifact_present=any('resolved_config' in f for f in paths),
                            unused_download_parts_bytes=sum(art['bytes'] for art in man['artifacts'] if '/.range_parts/' in art['path'])))
        print(json.dumps({'manifest':name,'n':len(man['artifacts']),'hash_issues':len(issues)}),flush=True)
    report=dict(format='openwam-codex-G2-manifest-audit/1',all_listed_artifacts_match=all(not r['issues'] and not r['missing_tracked_openwam_py'] for r in results),results=results,unique_files_read=len(cache),seconds=time.time()-start,no_model_loaded=True,script_sha256=sha(__file__))
    out.parent.mkdir(parents=True,exist_ok=True);t=out.with_name(out.name+'.tmp');t.write_text(json.dumps(report,indent=2)+'\n');os.replace(t,out)
    print(json.dumps({'all_listed_artifacts_match':report['all_listed_artifacts_match'],'seconds':report['seconds']}),flush=True)
if __name__=='__main__':main()
