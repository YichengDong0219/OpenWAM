"""Codex independent audit of controlled pack C; never reads annotations."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import stat

from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, protocol):
    p = json.loads(protocol.read_text())
    private = root/'private/mapping.json'
    public = root/'public/clips_c.json'
    mapping = json.loads(private.read_text())
    clips = mapping['clips']
    pub = json.loads(public.read_text())
    issues = []
    if len(clips) != 60 or len(pub) != 60:
        raise ValueError('expected60 clips')
    if mapping['prereg_sha256'] != sha(protocol): issues.append('prereg SHA')
    if stat.S_IMODE(private.stat().st_mode) != 0o600: issues.append('private mapping permissions')
    if stat.S_IMODE(private.parent.stat().st_mode) != 0o700: issues.append('private directory permissions')
    items = [(o['obs_id'], role) for o in p['observations'] for role in ('receiver', 'donor')]
    if len(set(items)) != 60: raise ValueError('nonunique protocol observations')
    rng = random.Random(20261019)
    draws = 0
    while True:
        draws += 1
        order = list(range(60)); rng.shuffle(order)
        positions = {}
        for slot, index in enumerate(order): positions.setdefault(items[index][0], []).append(slot)
        gaps = [abs(a-b) for a,b in positions.values()]
        if min(gaps) >= 10: break
        if draws > 1000000: raise RuntimeError('no accepted order within diagnostic limit')
    if mapping['draws_needed'] != draws: issues.append('first qualifying draw number')
    if [(c['obs_id'], c['role']) for c in clips] != [items[i] for i in order]: issues.append('seeded display order')
    indexed = {}
    for source in p['alpha_baseline_sources'].values():
        index_path = Path(source['root'])/'index.jsonl'
        if sha(index_path) != source['index_sha256']: issues.append('G1 source index changed')
        for line in index_path.read_text().splitlines():
            row = json.loads(line)
            if row['kind'] == 'grid': indexed[row['path']] = row['sha256']
    observations = {o['obs_id']: o for o in p['observations']}
    frame_count = exact_jpegs = 0
    for number, (c, u) in enumerate(zip(clips, pub)):
        cid = f'K{number+1:03d}'
        if any(c[k] != value or u[k] != value for k,value in {'pack':'C','clip_id':cid,'order':number}.items()):
            issues.append('public/private identity:'+cid)
        if set(u) != {'pack','clip_id','order','sheet','frame_w','frame_h'}: issues.append('public extra fields:'+cid)
        o = observations[c['obs_id']]
        v = 1001 if c['role'] == 'receiver' else 1002
        expected_root = Path(p['alpha_baseline_sources'][o['source_g1_split']]['root'])/o['task']/o['obs_id']
        grid = expected_root/'grid'/f'v{v}_a2001.pt'
        if (c['video_seed'],c['action_seed']) != (v,2001): issues.append('seed identity:'+cid)
        if Path(c['grid_trace']) != grid or sha(grid) != c['grid_trace_sha256'] or indexed[str(grid)] != c['grid_trace_sha256']:
            issues.append('trace provenance:'+cid)
        if len(c['frames']) != 9 or len(c['frame_sha256']) != 9: raise ValueError('frame inventory:'+cid)
        ims = []
        for k, (fn, digest) in enumerate(zip(c['frames'],c['frame_sha256'])):
            frame = expected_root/'decoded'/f'v{v}_a2001'/f'frame_{k}.png'
            if Path(fn) != frame or sha(frame) != digest: issues.append('frame provenance:'+cid+':'+str(k))
            ims.append(Image.open(frame).convert('RGB'));frame_count += 1
        w,h=ims[0].size
        sheet = Image.new('RGB',(w*9,h))
        for k,im in enumerate(ims): sheet.paste(im,(w*k,0))
        buffer=io.BytesIO();sheet.save(buffer,format='JPEG',quality=92)
        path = root/'public'/u['sheet']
        if u['sheet'] != f'sheets/C-{cid}.jpg' or (u['frame_w'],u['frame_h']) != (w,h): issues.append('public layout:'+cid)
        if sha(path) != c['sheet_sha256']: issues.append('sheet receipt:'+cid)
        if hashlib.sha256(buffer.getvalue()).hexdigest() != sha(path): issues.append('sheet reconstruction:'+cid)
        else: exact_jpegs += 1
        if Image.open(path).getexif(): issues.append('unexpected EXIF:'+cid)
    expected_files = {public.resolve()} | {(root/'public'/u['sheet']).resolve() for u in pub}
    if {f.resolve() for f in (root/'public').rglob('*') if f.is_file()} != expected_files: issues.append('public extra/missing files')
    return {'issues':issues,'passed':not issues,'observations':30,'clips':60,'controlled_pairs':30,
            'grid_trace_hashes_checked':60,'frame_hashes_checked':frame_count,'exact_jpeg_reconstructions':exact_jpegs,
            'order_seed':20261019,'draws_needed':draws,'minimum_position_difference':min(gaps),
            'mapping_sha256':sha(private),'public_list_sha256':sha(public),'protocol_sha256':sha(protocol),
            'verifier_sha256':sha(__file__),'scope':'CPU material/identity/order audit only; no rating access or new GPU decode',
            'prior_decode_validation':'G1 V1 already independently decoded27 frames on3 registered receiver clips; no new decode claimed'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root,args.protocol)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(result,indent=2)+'\n'
    if args.out.exists() and args.out.read_text()!=data: raise ValueError('refusing differing audit overwrite')
    tmp=args.out.with_name(args.out.name+'.tmp');tmp.write_text(data);tmp.replace(args.out)
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)
