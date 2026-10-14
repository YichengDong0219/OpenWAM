"""Pure fixed-denominator pack C aggregation; no real score/file/database I/O."""
from collections import Counter

from rating_rules import classify_pair, validate


def summarize(pairs, rows, *, include_partial=False):
    if len(pairs) != 30: raise ValueError('expected30 controlled pairs')
    keys = [tuple(pair[role]) for pair in pairs for role in ('receiver', 'donor')]
    expected = {('C', f'K{i:03d}') for i in range(1,61)}
    if set(keys) != expected or len(set(keys)) != 60:
        raise ValueError('invalid/repeated pack C identities')
    if set(rows)-expected: raise ValueError('non-C/unregistered identity; practice scores are excluded')
    for key, row in rows.items():
        if row is not None and validate(row): raise ValueError('invalid score schema: '+str(key))
    decisions = []
    for pair in pairs:
        first=rows.get(tuple(pair['receiver']));second=rows.get(tuple(pair['donor']))
        result = classify_pair(first or {},second or {},include_partial=include_partial)
        decisions.append({'pair_id':pair['pair_id'],'task':pair.get('task'),'episode':pair.get('episode'),
                          'potential_unblinding':('C','K001') in (tuple(pair['receiver']),tuple(pair['donor'])),
                          **result})

    def counts(values):
        categories=Counter(v['decision'] for v in values)
        n=len(values);positive=categories['visible_label_divergence'];compatible=categories['compatible_labels']
        return {'pair_denominator':n,'visible_label_divergence':positive,'compatible_labels':compatible,
                'indeterminate':categories['indeterminate'],'decidable':positive+compatible,
                'decidable_coverage':(positive+compatible)/n if n else None,
                'divergence_all_pairs':positive/n if n else None,
                'divergence_decidable':positive/(positive+compatible) if positive+compatible else None}
    subset=[v for v in decisions if not v['potential_unblinding']]
    if len(subset)!=29: raise ValueError('expected one potentially unblinded pair')
    return {'mode':'partial_sensitivity' if include_partial else 'clear_primary',
            'clip_denominator':60,'clips_submitted':sum(rows.get(k) is not None for k in expected),
            'main':counts(decisions),'exclude_potential_unblinding':counts(subset),'pairs':decisions}
