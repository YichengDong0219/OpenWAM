"""Codex independent synthetic review of Claude's driver; no real labels loaded."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import rating_compare_run as driver


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def record(cid):
    return dict(clip_id=cid,pack='C',active_arm='left',grasp_frame_interval='none',
                release_frame_interval='none',handover_frame_interval='none',legibility='clear',uncertain_reason='')


def must_reject(fn):
    try: fn()
    except (ValueError,KeyError,SystemExit): return True
    return False


def main():
    passed=[];failures=[]
    h=driver.hierarchy(['agreement','disagreement','disagreement','indeterminate','indeterminate','indeterminate'],
                       [('task',0),('task',1),('task',1),('task',1),('task',2),('task',2)])['task']
    assert abs(h['eq_weight_agreement_all']-1/3)<1e-12
    assert abs(h['eq_weight_coverage']-5/9)<1e-12
    assert h['eq_weight_agreement_decidable']==0.5 and h['episodes_with_decidable']==2
    assert h['pooled_counts']['agreement']==1 and h['pooled_counts']['denominator']==6
    passed.append('unequal episode sizes and undefined conditional mean independently verified')
    with tempfile.TemporaryDirectory(prefix='codex_rating_driver_synthetic_') as temporary:
        root=Path(temporary);(root/'sealed_human').mkdir();(root/'sealed_ai').mkdir();(root/'private').mkdir()
        rows=[record(f'K{i:03d}') for i in range(1,61)]
        for relative in ['sealed_human/rater_human_C.jsonl','sealed_ai/rater_ai1.jsonl','sealed_ai/rater_ai2.jsonl']:
            (root/relative).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        short=root/'short.jsonl';short.write_text('\n'.join(json.dumps(r) for r in rows[:-1])+'\n')
        assert len(driver.load_rater(short,digest(short)))==59
        passed.append('missing row accepted by loader without label imputation')
        shape=driver.rater_shape(driver.load_rater(short,digest(short)))
        if sum(shape['grasp_frame_interval']['kinds'].values())!=60 or shape['grasp_frame_interval']['kinds'].get('missing')!=1:
            failures.append({'case':'missing-rater-descriptive-denominator','expected':{'total':60,'missing':1},
                             'actual':shape['grasp_frame_interval']['kinds']})
        bad=root/'bad.jsonl'
        for name, contents in [('duplicate',rows+[rows[0]]),('unknown',[record('K061')]),('old practice',[dict(record('K001'),pack='A')]),
                               ('illegal interval',[dict(record('K001'),grasp_frame_interval='8-2')])]:
            bad.write_text('\n'.join(json.dumps(r) for r in contents)+'\n')
            assert must_reject(lambda:driver.load_rater(bad,digest(bad))),name
            passed.append('loader rejects '+name)
        bad.write_text(json.dumps(rows[0])+'\n\n')
        assert must_reject(lambda:driver.load_rater(bad,digest(bad)))
        passed.append('loader rejects blank row')
        mapping=[]
        for i in range(30):
            for offset,role in [(1,'receiver'),(2,'donor')]:
                mapping.append(dict(pack='C',clip_id=f'K{2*i+offset:03d}',obs_id=f'obs{i:02d}',
                                    role=role,task=f'task{i//10}',episode=(i%10)//2))
        mp=root/'private/mapping.json';mp.write_text(json.dumps({'clips':mapping}))
        pairs,units=driver.load_pairs(mp)
        assert len(pairs)==30 and {p['pair_id'] for p in pairs}=={f'obs{i:02d}' for i in range(30)}
        passed.append('pair IDs and episode identity match frozen v2 contract')
        rs={'human':driver.load_rater(short,digest(short)),'ai1':{r['clip_id']:r for r in rows},'ai2':{r['clip_id']:r for r in rows}}
        analysed=driver.analyse(rs,pairs,units)
        primary=analysed['comparisons']['human-ai1']['clear_primary']
        assert primary['all']['clip_level']['active_arm']['denominator']==60
        assert primary['all']['clip_level']['active_arm']['indeterminate']==1
        assert primary['exclude_K001']['clip_level']['active_arm']['denominator']==59
        assert primary['all']['pair_level']['counts']['denominator']==30
        assert primary['exclude_K001']['pair_level']['counts']['denominator']==29
        passed.append('driver end-to-end synthetic missing/all/exclusion denominators verified')
        seal={'human':{'C':{'sha256':digest(root/'sealed_human/rater_human_C.jsonl')}},
              'ai':{'rater_ai1':{'sha256':digest(root/'sealed_ai/rater_ai1.jsonl')},
                    'rater_ai2':{'sha256':digest(root/'sealed_ai/rater_ai2.jsonl')}}}
        sp=root/'seal.json';sp.write_text(json.dumps(seal))
        here=Path(driver.__file__).parent
        code={n:digest(here/n) for n in ('rating_compare_run.py','rating_compare.py','rating_rules.py','rating_rules_v2.py')}
        receipt={'format':'openwam-human-ai-comparison-freeze/1','status':'jointly_frozen','code_sha256':code,
                 'seal_manifest_sha256':'0'*64,'mapping_sha256':'0'*64}
        fp=root/'freeze.json';fp.write_text(json.dumps(receipt))
        def cli(out):
            return subprocess.run([sys.executable,str(here/'rating_compare_run.py'),'--pack-root',str(root),
                '--freeze-receipt',str(fp),'--seal-manifest',str(sp),'--out',str(out)],capture_output=True,text=True)
        result=cli(root/'result.json')
        if result.returncode==0:
            failures.append({'case':'seal-and-mapping-digest-mismatch-accepted','expected':'reject before parsing ratings','actual':'output written despite both wrong frozen hashes'})
        receipt['code_sha256']={};fp.write_text(json.dumps(receipt))
        result=cli(root/'bypassed-code.json')
        if result.returncode==0:
            failures.append({'case':'empty-code-freeze-map-accepted','expected':'require exact4 code hashes','actual':'all code checking bypassed'})
        receipt['code_sha256']=code;fp.write_text(json.dumps(receipt))
        enlarged=mapping+[dict(mapping[0],role='extra',task='other',episode=99)]
        mp.write_text(json.dumps({'clips':enlarged}))
        try:
            _,altered_units=driver.load_pairs(mp)
            failures.append({'case':'extra-mapping-row-accepted','expected':'reject non60/unknown-role mapping',
                             'actual_K001_unit':altered_units['K001']})
        except (ValueError,KeyError):pass
    report={'format':'openwam-codex-synthetic-driver-audit/1','driver_sha256':digest(Path(driver.__file__)),
            'synthetic_only':True,'real_scores_read':False,'passed_cases':passed,'blocking_findings':failures,
            'source_review_notes':['output existence checked only after parsing/analyzing ratings',
                                   'no first label parse UTC recorded in provenance',
                                   'mapping SHA only recorded after analysis; frozen seal SHA not checked']}
    out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_name(out.name+'.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
    print(json.dumps({'passed_cases':len(passed),'blocking_findings':len(failures),'driver_sha256':report['driver_sha256'],
                      'report':str(out)}))


if __name__=='__main__':main()
