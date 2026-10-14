"""Package only preregistered clips from one released split; never assign labels."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import random
from PIL import Image


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prereg',required=True,type=Path)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--split',required=True,choices=['explore','holdout'])
    p.add_argument('--explore-freeze',type=Path);a=p.parse_args()
    reg=json.loads(a.prereg.read_text());digest=sha(a.prereg)
    if a.split=='holdout':
        if a.explore_freeze is None:raise SystemExit('holdout rating pack requires --explore-freeze')
        receipt=json.loads(a.explore_freeze.read_text())
        if receipt.get('prereg_sha256')!=digest or receipt.get('explore_analysis_frozen') is not True:
            raise SystemExit('explore freeze receipt does not cover this prereg')
    assert (a.run/'PREREG_SHA256').read_text().strip()==digest
    obs={o['obs_id']:o for o in reg['observations'] if o['split']==a.split}
    clips=[r for r in reg['verification']['rating_clips'] if r['split']==a.split]
    assert len(clips)==2*len(obs)
    random.Random(20261016).shuffle(clips)  # presentation order only, never changes selection
    if a.out.exists():raise SystemExit('output exists; do not overwrite an issued rating pack')
    public=a.out/'public';private=a.out/'private';public.mkdir(parents=True);private.mkdir(mode=0o700)
    mapping=[];manifest=[];sections=[]
    for i,c in enumerate(clips,1):
        o=obs[c['obs_id']];cid=f'C{i:03d}'
        src=a.run/o['task']/o['obs_id']/'decoded'/f"v{c['video_seed']}_a{c['action_seed']}"
        assert (src/'DONE').exists() and len(list(src.glob('frame_*.png')))==9
        dest=public/cid;dest.mkdir();images=[];hashes={}
        for k in range(9):
            path=dest/f'frame_{k}.png'
            with Image.open(src/f'frame_{k}.png') as image:
                clean=Image.new('RGB',image.size);clean.paste(image.convert('RGB'));clean.save(path)
            assert clean.size==(320,384);images.append(clean);hashes[path.name]=sha(path)
        gif=dest/'preview.gif';images[0].save(gif,save_all=True,append_images=images[1:],duration=250,loop=0)
        manifest.append(dict(clip_id=cid,frames=9,frame_sha256=hashes,preview_sha256=sha(gif)))
        mapping.append(dict(clip_id=cid,**c,task=o['task'],episode=o['episode'],trace_sha256=o['trace_sha256'],source=str(src)))
        thumbs=''.join(f'<img width="160" src="{cid}/frame_{k}.png" alt="frame {k}">' for k in range(9))
        sections.append(f'<section><h2>{cid}</h2><img src="{cid}/preview.gif" alt="{cid}"><details><summary>Frames 0–8</summary>{thumbs}</details></section>')
    with (public/'ratings.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['clip_id','active_arm','release_frame_interval','handover_frame_interval','uncertain_reason','notes'])
        writer.writerows([[r['clip_id'],'','','','',''] for r in manifest])
    (public/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Anonymous future clips</title><style>body{font-family:sans-serif}section{border-bottom:1px solid #bbb;padding:1em}img{margin:.2em}</style><h1>Anonymous future clips</h1><p>Display speed is illustrative. Open ratings.csv separately. Frames are numbered 0–8.</p>'+''.join(sections))
    (public/'README.md').write_text('匿名预测视频评审包。尚未指定评审人，也未生成任何标签。\n\n只看图像，不查看 private、动作、种子、结果或对方评分。active_arm 可填 left/right/both/neither/uncertain；依据头部视角中的可见运动，腕部槽位仅辅助。release/handover 仅在可见且可判定时填写帧区间（0–8），否则留空并说明不确定。不能由图像断言物理任务成功。GIF 每帧250ms仅用于展示，不代表仿真速度；以原始PNG查看细节。最终人工/模型评审安排由用户决定。\n')
    (public/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (private/'mapping.json').write_text(json.dumps(dict(prereg_sha256=digest,split=a.split,presentation_seed=20261016,mapping=mapping),indent=2)+'\n')
    os.chmod(private/'mapping.json',0o600)
    report=dict(prereg_sha256=digest,split=a.split,n_clips=len(clips),n_frames=9*len(clips),
                public_path=str(public),private_mapping=str(private/'mapping.json'),
                public_manifest_sha256=sha(public/'manifest.json'),rating_pack_script_sha256=sha(__file__),
                labels_generated=0,holdout_loaded=a.split=='holdout')
    (a.out/'pack_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
