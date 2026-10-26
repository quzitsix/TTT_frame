#!/usr/bin/env python3
"""Compare Idea5 source times with old full ICL, explicitly measuring pixel mismatch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.evaluate_forecast_mcq import file_hash
from scripts.run_idea5_acquisition import atomic_json

FIELDS=('context_requested_seconds','context_actual_seconds','recent_requested_seconds',
        'recent_actual_seconds','source_media')


def compare(new,old):
    if len({r['id'] for r in new})!=len(new) or len({r['id'] for r in old})!=len(old):
        raise ValueError('duplicate IDs')
    old={r['id']:r for r in old}
    if {r['id'] for r in new}!=set(old):raise ValueError('case coverage mismatch')
    result=[]
    for row in new:
        sid=row['id'];previous=old[sid]
        if any(row['sampling'][key]!=previous['sampling'][key] for key in FIELDS):
            raise ValueError(f'source/timestamp mismatch: {sid}')
        result.append({'id':sid,'source_times_equal':True,
                       'mid224_pixels_equal':row['arms']['mid224']['selected_pixel_sha256']==previous['sampling']['frame_bundle_sha256']})
    return {'cases':len(result),'source_times_equal':len(result),
            'mid224_pixels_equal':sum(x['mid224_pixels_equal'] for x in result),'per_case':result,
            'limitation':'15 segments vs one video, plus 448->224 vs direct224 resizing; not a pure packing ablation'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--baseline-dir',type=Path,default=Path('runs/forecast_bench_v0/mcq_baseline_v1/eval'))
    args=p.parse_args(argv);path=args.run_dir/'scores.jsonl'
    old=[];files=sorted(args.baseline_dir.glob('qwen_shard*.jsonl'))
    for f in files:
        old.extend(x for x in map(json.loads,f.read_text().splitlines()) if x['arm']=='qwen_full_icl')
    result=compare(list(map(json.loads,path.read_text().splitlines())),old)
    if result['cases']!=200:raise ValueError('complete benchmark required')
    result.update(schema='idea05_input_audit_v1',answer_files_read=False,scores_sha256=file_hash(path),
                  baseline_shards={str(f):file_hash(f) for f in files},source_sha256=file_hash(Path(__file__)))
    atomic_json(args.run_dir/'input_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_case','baseline_shards']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
