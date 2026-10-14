"""Pure human/AI agreement functions. No real-label or database/file access."""
from collections import Counter
from rating_rules import FIELDS, arm, event, validate


def clip_comparison(first, second, *, include_partial=False):
    allowed={'clear','partial'} if include_partial else {'clear'}
    readable=bool(first and second and first.get('legibility') in allowed and second.get('legibility') in allowed)
    errors=[validate(row) if row is not None else [] for row in (first,second)]
    out={}
    for field in FIELDS:
        parse=arm if field=='active_arm' else event
        s1,v1=parse((first or {}).get(field));s2,v2=parse((second or {}).get(field))
        if not readable or any(errors) or s1!='known' or s2!='known': status='indeterminate'
        elif field=='active_arm':status='agreement' if v1==v2 else 'disagreement'
        elif v1 is None or v2 is None:status='agreement' if v1 is None and v2 is None else 'disagreement'
        else:status='agreement' if max(v1[0],v2[0])<=min(v1[1],v2[1]) else 'disagreement'
        out[field]=status
    return {'fields':out,'validation_errors':errors}


def counts(statuses):
    c=Counter(statuses);n=len(statuses);a=c['agreement'];d=c['disagreement']
    return {'denominator':n,'agreement':a,'disagreement':d,'indeterminate':c['indeterminate'],
            'coverage':(a+d)/n if n else None,'agreement_all':a/n if n else None,
            'agreement_decidable':a/(a+d) if a+d else None}


def clip_summary(first, second, *, include_partial=False, exclude_k001=False):
    expected={f'K{i:03d}' for i in range(1,61)}
    if (set(first)|set(second))-expected:raise ValueError('unknown clip identity')
    rows=[clip_comparison(first.get(k),second.get(k),include_partial=include_partial)
          for k in sorted(expected) if not(exclude_k001 and k=='K001')]
    if any(any(row['validation_errors']) for row in rows):raise ValueError('invalid schema; do not silently drop')
    return {f:counts([r['fields'][f] for r in rows]) for f in FIELDS}


def pair_summary(first, second, *, excluded_pair_ids=()):
    """Input decisions from frozen v2 classifier, with identical30 private pair IDs."""
    if set(first)!=set(second) or len(first)!=30:raise ValueError('expected same30 pair identities')
    allowed={'visible_label_divergence','compatible_labels','indeterminate'}
    if any(v not in allowed for v in list(first.values())+list(second.values())):raise ValueError('invalid decision')
    if set(excluded_pair_ids)-set(first):raise ValueError('unknown excluded pair')
    keys=sorted(set(first)-set(excluded_pair_ids));status=[];matrix=Counter()
    for key in keys:
        a,b=first[key],second[key];matrix[a+'|'+b]+=1
        status.append('indeterminate' if 'indeterminate' in (a,b) else 'agreement' if a==b else 'disagreement')
    return {'counts':counts(status),'raw_3x3':{a+'|'+b:matrix[a+'|'+b] for a in sorted(allowed) for b in sorted(allowed)},
            'raw_same_category_count':sum(first[k]==second[k] for k in keys)}
