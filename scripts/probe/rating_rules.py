"""Score-blind pair classification for the single-human rating-v1 protocol.

Pure functions only. No artifact/database or real rating file access is implemented.
"""
import re

FIELDS = ('active_arm', 'grasp_frame_interval', 'release_frame_interval', 'handover_frame_interval')
ARMS = {'left', 'right', 'both', 'neither'}


def event(value):
    if value is None or str(value).strip() == '': return ('missing', None)
    value = str(value).strip().lower()
    if value == 'uncertain': return ('uncertain', None)
    if value == 'none': return ('known', None)
    match = re.fullmatch(r'([0-8])\s*-\s*([0-8])', value)
    if not match or int(match[1]) > int(match[2]): return ('invalid', None)
    return ('known', (int(match[1]), int(match[2])))


def arm(value):
    if value is None or str(value).strip() == '': return ('missing', None)
    value = str(value).strip().lower()
    if value == 'uncertain': return ('uncertain', None)
    return ('known', value) if value in ARMS else ('invalid', None)


def validate(row):
    errors = []
    legibility = row.get('legibility')
    if legibility not in ('clear', 'partial', 'illegible'): errors.append('invalid_legibility')
    for field in FIELDS:
        status, _ = (arm if field == 'active_arm' else event)(row.get(field))
        if status == 'invalid': errors.append('invalid_'+field)
    if (legibility == 'illegible' or any(str(row.get(f, '')).strip().lower() == 'uncertain' for f in FIELDS)) and not str(row.get('uncertain_reason', '')).strip():
        errors.append('missing_uncertain_reason')
    return errors


def classify_pair(first, second, *, include_partial=False):
    allowed = {'clear', 'partial'} if include_partial else {'clear'}
    readable = first.get('legibility') in allowed and second.get('legibility') in allowed
    errors = [validate(first), validate(second)]
    fields = {}
    for field in FIELDS:
        parse = arm if field == 'active_arm' else event
        st1, v1 = parse(first.get(field)); st2, v2 = parse(second.get(field))
        if not readable or any(errors) or st1 != 'known' or st2 != 'known':
            fields[field] = 'indeterminate'
        elif field == 'active_arm':
            fields[field] = 'different_arm_label' if v1 != v2 else 'compatible'
        elif (v1 is None) != (v2 is None):
            fields[field] = 'different_visible_event_presence'
        elif v1 is not None and (v1[1] < v2[0] or v2[1] < v1[0]):
            fields[field] = 'disjoint_event_intervals'
        else:
            fields[field] = 'compatible'
    different = [f for f, status in fields.items() if status.startswith('different_') or status == 'disjoint_event_intervals']
    decision = 'visible_label_divergence' if different else (
        'compatible_labels' if all(s == 'compatible' for s in fields.values()) else 'indeterminate')
    return {'decision': decision, 'field_decisions': fields, 'divergent_fields': different,
            'both_readable': readable, 'validation_errors': errors,
            'mode': 'partial_sensitivity' if include_partial else 'clear_primary'}
