"""Independent R23 acceptance. Every fixture is synthetic; label-load spy proves ordering."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import rating_compare_run as D

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p, obj): p.write_text(json.dumps(obj))

def main():
    passed=[]
    here=Path(D.__file__).parent
    with tempfile.TemporaryDirectory(prefix='codex_R23_synthetic_') as t:
        root=Path(t)
        for sub in ('private','sealed_human','sealed_ai'): (root/sub).mkdir()
        mapping=[]
        for i in range(30):
            for j, role in enumerate(('receiver','donor')):
                mapping.append(dict(pack='C',clip_id=f'K{2*i+j+1:03d}',obs_id=f'o{i:02d}',
                                    task=f't{i//10}',episode=i%3,role=role))
        mp=root/'private/mapping.json'; put(mp,{'clips':mapping})
        rows=[dict(pack='C',clip_id=k,active_arm='left',legibility='clear',uncertain_reason='',
                   grasp_frame_interval='none',release_frame_interval='none',handover_frame_interval='none') for k in D.CLIPS]
        files={n:root/rel for n,(rel,_) in D.SEALED.items()}
        for n,p in files.items(): p.write_text(''.join(json.dumps(r)+'\n' for r in (rows[:-1] if n=='ai2' else rows)))
        hashes={n:sha(p) for n,p in files.items()}
        seal={'human':{'C':{'sha256':hashes['human']}},'ai':{f'rater_{n}':{'sha256':hashes[n]} for n in ('ai1','ai2')}}
        sp=root/'seal.json';put(sp,seal)
        doc=root/'rules.md';doc.write_text('synthetic rules')
        candidate=root/'candidate.json';put(candidate,{'synthetic':True})
        receipt=dict(format='openwam-human-ai-comparison-freeze/1',status='jointly_frozen',
                     signatures={'claude':'synthetic','codex':'synthetic'},
                     code_sha256={n:sha(here/n) for n in ('rating_compare_run.py','rating_compare.py','rating_rules.py','rating_rules_v2.py')},
                     rules_doc_sha256=sha(doc),candidate_json_sha256=sha(candidate),seal_manifest_sha256=sha(sp),
                     mapping_sha256=sha(mp),sealed_inputs=hashes)
        fp=root/'freeze.json';put(fp,receipt)
        out=root/'result.json';journal=out.with_suffix('.journal.json')
        args=argparse.Namespace(pack_root=str(root),freeze_receipt=str(fp),seal_manifest=str(sp),
                                rules_doc=str(doc),candidate=str(candidate),out=str(out))
        argv=['driver','--pack-root',str(root),'--freeze-receipt',str(fp),'--seal-manifest',str(sp),
              '--rules-doc',str(doc),'--candidate',str(candidate),'--out',str(out)]
        original=D.load_rater
        loads=[]
        def spy(p,h):
            j=json.loads(journal.read_text())
            assert j['status']=='started' and j['label_read_started_at'].endswith('Z')
            loads.append(str(p))
            return original(p,h)
        with patch.object(sys,'argv',argv),patch.object(D,'load_rater',spy),contextlib.redirect_stdout(io.StringIO()): D.main()
        assert len(loads)==3
        res=json.loads(out.read_text());j=json.loads(journal.read_text())
        assert j['status']=='finished' and j['result_sha256']==sha(out)
        assert j['label_read_started_at']<=j['analysis_finished_at']
        assert res['provenance']['label_read_started_at']==j['label_read_started_at']
        s=res['rater_shape']['ai2']
        assert s['n_rows']==59 and s['n_missing_rows']==1 and s['n_ids']==60
        for f in ('legibility','active_arm'): assert sum(s[f].values())==60 and s[f]['missing_row']==1
        for f in D.EVENTS: assert s[f]['kinds']=={'none':59,'missing_row':1} and s[f]['n_intervals']==0
        for cmp in res['comparisons'].values():
            for mode in cmp.values():
                for tag,r in mode.items():
                    nc,np=(60,30) if tag=='all' else (59,29)
                    for f,c in r['clip_level'].items():
                        assert c['denominator']==nc
                        assert sum(v['pooled_counts']['denominator'] for v in r['clip_level_by_task'][f].values())==nc
                    assert r['pair_level']['counts']['denominator']==np
        passed.append('valid full CLI, missing_row fixed60, journal-before-load, 60/59 and 30/29')
        def refuse(name, mutate=None, action=None):
            put(fp,receipt)
            if mutate:
                r=copy.deepcopy(receipt);mutate(r);put(fp,r)
            if action: action()
            sentinel=AssertionError('rating parse attempted before refusal')
            with patch.object(sys,'argv',argv),patch.object(D,'load_rater',side_effect=sentinel):
                try: D.main()
                except (SystemExit,ValueError,KeyError): pass
                else: raise AssertionError(name+' accepted')
            passed.append(name+' refused with zero rating-load calls')
        before=journal.read_bytes();refuse('existing output/journal');assert journal.read_bytes()==before
        out.unlink();journal.unlink()
        cases=[('wrong seal',lambda r:r.update(seal_manifest_sha256='0'*64)),
               ('wrong mapping',lambda r:r.update(mapping_sha256='0'*64)),
               ('wrong candidate',lambda r:r.update(candidate_json_sha256='0'*64)),
               ('wrong rules',lambda r:r.update(rules_doc_sha256='0'*64)),
               ('empty code',lambda r:r.update(code_sha256={})),
               ('missing code',lambda r:r['code_sha256'].pop('rating_rules.py')),
               ('traversal code',lambda r:r['code_sha256'].update({'../x.py':'0'*64})),
               ('invalid code SHA',lambda r:r['code_sha256'].update({'rating_rules.py':'abc'})),
               ('wrong code SHA',lambda r:r['code_sha256'].update({'rating_rules.py':'0'*64})),
               ('missing signature',lambda r:r['signatures'].update(codex=None)),
               ('pending status',lambda r:r.update(status='pending')),
               ('seal input disagreement',lambda r:r['sealed_inputs'].update(ai1='0'*64)),
               ('extra sealed input',lambda r:r['sealed_inputs'].update(extra='0'*64))]
        for name,mutate in cases:
            refuse(name,mutate);assert not journal.exists()
        p=files['ai2'];raw=p.read_bytes();p.write_bytes(raw+b'{}\n')
        refuse('changed raw rating');assert not journal.exists();p.write_bytes(raw)
        for name,mutate in [('extra mapping row',lambda m:m.append(dict(m[0],role='extra'))),
                            ('wrong role',lambda m:m[0].update(role='extra')),
                            ('duplicate clip',lambda m:m[1].update(clip_id=m[0]['clip_id'])),
                            ('wrong pack',lambda m:m[0].update(pack='A')),
                            ('cross episode',lambda m:m[1].update(episode=99)),
                            ('bool episode',lambda m:m[0].update(episode=True))]:
            m=copy.deepcopy(mapping);mutate(m);put(mp,{'clips':m})
            refuse(name,lambda r:r.update(mapping_sha256=sha(mp)));assert not journal.exists()
        put(mp,{'clips':mapping});put(fp,receipt)
        p=files['human'];p.write_text('{bad-json}\n')
        broken=copy.deepcopy(seal);broken['human']['C']['sha256']=sha(p);put(sp,broken)
        r=copy.deepcopy(receipt);r['sealed_inputs']['human']=sha(p);r['seal_manifest_sha256']=sha(sp);put(fp,r)
        with patch.object(sys,'argv',argv):
            try:D.main()
            except SystemExit:pass
            else:raise AssertionError('malformed label accepted')
        failure=json.loads(journal.read_text())
        assert failure['status']=='failed' and failure['error_type']=='JSONDecodeError'
        assert not out.exists() and '{bad-json}' not in journal.read_text()
        passed.append('post-read failure journal contains error type/time and no label values')
    h=D.hierarchy(['agreement','disagreement','disagreement','indeterminate','indeterminate','indeterminate'],
                  [('task',0),('task',1),('task',1),('task',1),('task',2),('task',2)])['task']
    assert abs(h['eq_weight_agreement_all']-1/3)<1e-12 and abs(h['eq_weight_coverage']-5/9)<1e-12
    assert h['eq_weight_agreement_decidable']==.5 and h['episodes_with_decidable']==2
    passed.append('independent unequal-episode equal-weight arithmetic')
    report=dict(format='openwam-codex-driver-v2-audit/1',synthetic_only=True,real_scores_read=False,
                driver_sha256=sha(Path(D.__file__)),audit_sha256=sha(Path(__file__)),passed_cases=passed,blocking_findings=[])
    dest=Path(sys.argv[1]);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.tmp');put(tmp,report);tmp.replace(dest)
    print(json.dumps({'passes':len(passed),'blocking_findings':0,'report':str(dest)}))

if __name__=='__main__':main()
