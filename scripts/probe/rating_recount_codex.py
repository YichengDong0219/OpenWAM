"""R24 independent recount from frozen rules; imports no rating modules or driver."""
import argparse
import hashlib
import json
import os
import re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

FIELDS=('active_arm','grasp_frame_interval','release_frame_interval','handover_frame_interval')
IDS=tuple(f'K{i:03d}' for i in range(1,61))
NAMES=('human','ai1','ai2')
DECISIONS=('visible_label_divergence','compatible_labels','indeterminate')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parse(x,field):
    if x is None or str(x).strip()=='':return 'unknown',None
    x=str(x).strip().lower()
    if x=='uncertain':return 'unknown',None
    if field=='active_arm':
        assert x in {'left','right','both','neither'},'illegal arm'
        return 'known',x
    if x=='none':return 'known',None
    m=re.fullmatch(r'([0-8])\s*-\s*([0-8])',x);assert m and int(m[1])<=int(m[2]),'illegal event'
    return 'known',(int(m[1]),int(m[2]))
def compare(a,b,field,partial):
    allowed={'clear','partial'} if partial else {'clear'}
    if a is None or b is None or a.get('legibility') not in allowed or b.get('legibility') not in allowed:return 'indeterminate'
    sx,x=parse(a.get(field),field);sy,y=parse(b.get(field),field)
    if sx!='known' or sy!='known':return 'indeterminate'
    if field=='active_arm':agree=x==y
    elif x is None or y is None:agree=x is None and y is None
    else:agree=max(x[0],y[0])<=min(x[1],y[1])
    return 'agreement' if agree else 'disagreement'
def decision(a,b,partial):
    fields={f:compare(a,b,f,partial) for f in FIELDS}
    if 'disagreement' in fields.values():d='visible_label_divergence'
    elif set(fields.values())=={'agreement'}:d='compatible_labels'
    else:d='indeterminate'
    return d,fields
def count(values):
    c=Counter(values);n=len(values);a=c['agreement'];d=c['disagreement'];i=c['indeterminate'];assert a+d+i==n
    return dict(agreement=a,disagreement=d,indeterminate=i,denominator=n,coverage=(a+d)/n,
                agreement_all=a/n,agreement_decidable=a/(a+d) if a+d else None)
def hierarchy(status,units):
    groups=defaultdict(list)
    for x,u in zip(status,units):groups[u].append(x)
    tasks=defaultdict(dict)
    for (t,e),v in groups.items():tasks[t][str(e)]=count(v)
    out={}
    for t,eps in tasks.items():
        c=list(eps.values());dec=[v['agreement_decidable'] for v in c if v['agreement_decidable'] is not None]
        out[t]=dict(episodes=eps,n_episodes=len(c),episodes_with_decidable=len(dec),
                    eq_weight_agreement_all=sum(v['agreement_all'] for v in c)/len(c),
                    eq_weight_coverage=sum(v['coverage'] for v in c)/len(c),
                    eq_weight_agreement_decidable=sum(dec)/len(dec) if dec else None,
                    pooled_counts=count([s for (task,_),vals in groups.items() if task==t for s in vals]))
    return out
def match(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(match(v,b[k]) for k,v in a.items())
    if isinstance(a,float):return isinstance(b,(int,float)) and abs(a-b)<1e-12
    return a==b
def selfcheck():
    r=dict(legibility='clear',active_arm='left',grasp_frame_interval='2-3',release_frame_interval='none',handover_frame_interval='none')
    assert compare(r,dict(r,grasp_frame_interval='3-4'),FIELDS[1],False)=='agreement'
    assert compare(r,dict(r,grasp_frame_interval='4-5'),FIELDS[1],False)=='disagreement'
    assert compare(r,dict(r,grasp_frame_interval='none'),FIELDS[1],False)=='disagreement'
    assert compare(r,dict(r,legibility='partial'),FIELDS[0],False)=='indeterminate'
    assert compare(r,dict(r,legibility='partial'),FIELDS[0],True)=='agreement'
    assert decision(dict(r,release_frame_interval='uncertain'),dict(r,active_arm='right'),False)[0]=='visible_label_divergence'
    assert compare(None,None,FIELDS[0],False)=='indeterminate'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--freeze',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--journal',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    selfcheck();root=Path(args.root);out=Path(args.out);assert not out.exists()
    freeze=json.loads(Path(args.freeze).read_text());assert sha(args.freeze)=='2908554db17971f85041ec941370f494eb8a3f043437e187698eb776c67258d6' and freeze['status']=='jointly_frozen'
    assert sha(args.reference)=='d829d6752ff2e6e0c46f0d3632fd1084d6631cd1351c16056ba1513a6a706fed'
    j=json.loads(Path(args.journal).read_text());assert j['status']=='finished' and j['result_sha256']==sha(args.reference) and j['freeze_receipt_sha256']==sha(args.freeze)
    assert datetime.fromisoformat(j['label_read_started_at'].replace('Z','+00:00'))>=datetime.fromisoformat(freeze['frozen_at_utc'])
    mapping=root/'private/mapping.json';assert sha(mapping)==freeze['mapping_sha256']
    seal=root/'sealed_human/SEAL_MANIFEST.json';assert sha(seal)==freeze['seal_manifest_sha256']
    paths={'human':root/'sealed_human/rater_human_C.jsonl','ai1':root/'sealed_ai/rater_ai1.jsonl','ai2':root/'sealed_ai/rater_ai2.jsonl'}
    assert all(sha(p)==freeze['sealed_inputs'][n] for n,p in paths.items())
    started=datetime.now(timezone.utc).isoformat();raters={}
    for n,p in paths.items():
        rows={}
        for line in p.read_text().splitlines():
            assert line.strip();row=json.loads(line);cid=row['clip_id'].removeprefix('C-');assert cid in IDS and cid not in rows
            assert row.get('pack','C')=='C' and row.get('doc_id','C-'+cid)=='C-'+cid
            assert row['legibility'] in ('clear','partial','illegible')
            for f in FIELDS:parse(row.get(f),f)
            if row['legibility']=='illegible' or any(str(row.get(f,'')).strip().lower()=='uncertain' for f in FIELDS):assert str(row.get('uncertain_reason','')).strip()
            rows[cid]=row
        assert set(rows)==set(IDS);raters[n]=rows
    clips=json.loads(mapping.read_text())['clips'];assert len(clips)==60 and {c['clip_id'] for c in clips}==set(IDS)
    pairs=defaultdict(dict);units={}
    for c in clips:
        assert c['role'] in ('receiver','donor') and c['role'] not in pairs[c['obs_id']]
        pairs[c['obs_id']][c['role']]=c['clip_id'];units[c['clip_id']]=(c['task'],c['episode'])
    assert len(pairs)==30 and all(set(v)=={'receiver','donor'} for v in pairs.values())
    excluded=next(o for o,p in pairs.items() if 'K001' in p.values())
    calculated={'comparisons':{},'pair_decisions':{},'descriptives':{},'within_pair_fields':{}}
    for n,rows in raters.items():
        shape={'legibility':dict(Counter(r['legibility'] for r in rows.values())),'active_arm':dict(Counter(r['active_arm'] for r in rows.values()))}
        for f in FIELDS[1:]:
            c=Counter();width=Counter()
            for r in rows.values():
                s,v=parse(r.get(f),f);kind='uncertain' if s=='unknown' else ('none' if v is None else 'interval');c[kind]+=1
                if kind=='interval':width[str(v[1]-v[0])]+=1
            shape[f]={'kinds':dict(c),'widths':dict(width)}
        calculated['descriptives'][n]=shape
    for mode,partial in [('clear_primary',False),('partial_sensitivity',True)]:
        decisions={n:{o:decision(rows[p['receiver']],rows[p['donor']],partial) for o,p in pairs.items()} for n,rows in raters.items()}
        calculated['pair_decisions'][mode]={}
        calculated['within_pair_fields'][mode]={}
        for n,d in decisions.items():
            calculated['pair_decisions'][mode][n]={tag:dict(Counter(x[0] for o,x in d.items() if tag=='all' or o!=excluded)) for tag in ('all','exclude_K001')}
            calculated['within_pair_fields'][mode][n]={f:count([x[1][f] for x in d.values()]) for f in FIELDS}
        for a,b in [('human','ai1'),('human','ai2'),('ai1','ai2')]:
            key=a+'-'+b;calculated['comparisons'].setdefault(key,{})[mode]={}
            for tag in ('all','exclude_K001'):
                ids=[k for k in IDS if tag=='all' or k!='K001'];obs=[o for o in pairs if tag=='all' or o!=excluded]
                fields={f:[compare(raters[a][k],raters[b][k],f,partial) for k in ids] for f in FIELDS}
                matrix=Counter((decisions[a][o][0],decisions[b][o][0]) for o in obs)
                statuses=['indeterminate' if 'indeterminate' in (x,y) else 'agreement' if x==y else 'disagreement' for x,y in [(decisions[a][o][0],decisions[b][o][0]) for o in obs]]
                calculated['comparisons'][key][mode][tag]={'clip_level':{f:count(s) for f,s in fields.items()},
                    'clip_level_by_task':{f:hierarchy(s,[units[k] for k in ids]) for f,s in fields.items()},
                    'pair_level':{'counts':count(statuses),'raw_3x3':{x+'|'+y:matrix[(x,y)] for x in DECISIONS for y in DECISIONS},'raw_same_category_count':sum(matrix[(x,x)] for x in DECISIONS)},
                    'pair_level_by_task':hierarchy(statuses,[units[pairs[o]['receiver']] for o in obs])}
    # All calculations above precede accessing the reference result's count values.
    reference=json.loads(Path(args.reference).read_text());issues=[];checks=0
    for key,modes in calculated['comparisons'].items():
        for mode,tags in modes.items():
            for tag,sections in tags.items():
                for section,value in sections.items():
                    checks+=1
                    if not match(value,reference['comparisons'][key][mode][tag][section]):issues.append('/'.join((key,mode,tag,section)))
    for mode,names in calculated['pair_decisions'].items():
        for n,tags in names.items():
            for tag,c in tags.items():
                refs=reference['pair_decisions'][mode][n];actual=refs['main' if tag=='all' else 'exclude_potential_unblinding']['counts']
                checks+=1
                if any(c.get(d,0)!=actual[d] for d in DECISIONS):issues.append('pair_decisions/'+mode+'/'+n+'/'+tag)
    human=calculated['within_pair_fields']['clear_primary']['human']
    report={'format':'openwam-codex-independent-rating-recount/1','imports_no_rating_modules':True,'script_sha256':sha(__file__),
            'started_at_utc':started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'reference_sha256':sha(args.reference),
            'freeze_sha256':sha(args.freeze),'journal_sha256':sha(args.journal),'sealed_inputs':{n:sha(p) for n,p in paths.items()},
            'checks':checks,'issues':issues,'actual_labels_read':True,'calculated':calculated,
            'Q7_scope':'exact30 alpha pairs only; compatible does not establish semantic equality'}
    out.parent.mkdir(parents=True,exist_ok=True);t=out.with_name(out.name+'.tmp');t.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');os.replace(t,out)
    print(json.dumps({'checks':checks,'issues':issues,'human_primary_pair':calculated['pair_decisions']['clear_primary']['human']['all'],'human_field_counts':human}))
    assert not issues
if __name__=='__main__':main()
