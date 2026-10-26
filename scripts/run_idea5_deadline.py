#!/usr/bin/env python3
"""Post-hoc fixed-latest attribution control, separately frozen before scoring.

Added after inspecting motion policy locations in the six-arm preflight. This
is explicitly exploratory, not a seventh preregistered configuration.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_idea5_acquisition import validate,atomic_json
from scripts.evaluate_forecast_mcq import QwenEngine,label_tokens,file_hash,fingerprint
from scripts.evaluate_idea5_acquisition import paired,describe
from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.forecast_mcq import load_frame_bundle,build_mcq_prompt,frame_bundle_hash
from ttt_frame.idea5_acquisition import answer,thumbnail

SIDES=[448 if i in (3,7,11,14) else 112 for i in range(15)]


def run_worker(args):
    import torch
    p=validate(args.run_dir/'protocol.json');bp=p['payload']['baseline_protocol']['payload']
    torch.set_num_threads(2);torch.manual_seed(0)
    for name,digest in bp['base_artifact_sha256'].items():
        if file_hash(Path(bp['model_path'])/name)!=digest:raise ValueError('base model changed')
    engine=QwenEngine(bp['model_path'],args.device)
    from ttt_frame.vision_patch_embed import LinearVisionPatchEmbed
    if bp['vision_linear_patch']:engine.vision_patch_embed=LinearVisionPatchEmbed(engine.model)
    if bp['vision_batched_sdpa']:
        from ttt_frame.vision_attention import EqualLengthVisionSDPA
        engine.vision_attention_patch=EqualLengthVisionSDPA(engine.model)
    token_ids=label_tokens(engine.processor.tokenizer)
    cases=load_forecast_cases(Path(bp['root']))[args.shard_index::4]
    out=args.run_dir/'deadline'/f'scores_{args.shard_index}.jsonl'
    with out.open('x') as f:
        for case in cases:
            start=time.perf_counter();bundle=load_frame_bundle(case,max_side=448)
            decoded=time.perf_counter()-start
            images=bundle['context_images']+bundle['recent_images'];times=bundle['context_actual_seconds']+bundle['recent_actual_seconds']
            requested=bundle['context_requested_seconds']+bundle['recent_requested_seconds']
            chunks=[images[i:i+6] for i in range(0,90,6)];chunk_times=[times[i:i+6] for i in range(0,90,6)]
            result=answer(engine,chunks,chunk_times,SIDES,build_mcq_prompt(case),token_ids)
            chosen=[thumbnail(im,s) for ch,s in zip(chunks,SIDES) for im in ch]
            row={'id':case.id,'arm':'deadline_high','protocol_fingerprint':p['protocol_fingerprint'],
                 'result':result,'source_frame_sha256':bundle['frame_bundle_sha256'],
                 'selected_pixel_sha256':frame_bundle_hash(chosen,times,requested),
                 'resolutions':SIDES,'decode_seconds':decoded,'episode_seconds':time.perf_counter()-start,
                 'answer_keys_used':False,'future_media_used':False}
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({'id':case.id,'prediction':result['prediction']}),flush=True)


def validate_rows(rows,expected,main):
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=set(expected):raise ValueError('incomplete/duplicate IDs')
    for r in rows:
        if (r['resolutions']!=SIDES or r['result']['prediction'] not in list('ABCDEF')
                or r['result']['status']!='ok' or r['answer_keys_used'] is not False
                or r['future_media_used'] is not False
                or r['source_frame_sha256']!=main[r['id']]['sampling']['frame_bundle_sha256']):
            raise ValueError('invalid control result or changed pixels')


def launch(args):
    p=validate(args.run_dir/'protocol.json');out=args.run_dir/'deadline';out.mkdir()
    payload={'base_protocol_fingerprint':p['protocol_fingerprint'],'resolutions':SIDES,
             'source_sha256':file_hash(Path(__file__)),'main_scores_sha256':file_hash(args.run_dir/'scores.jsonl'),
             'selection':'fixed latest chunk in each budget cell; added after six-arm scoring',
             'test_status':'post-hoc exploratory attribution control, no threshold fitting',
             'created_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(out/'protocol.json',{'payload':payload,'protocol_fingerprint':fingerprint(payload)})
    import torch
    for d in ['cuda:4','cuda:5']:
        if torch.cuda.mem_get_info(d)[0]<16*2**30:raise ValueError('insufficient free memory')
    processes=[];logs=[];start=time.perf_counter()
    try:
        for i,d in enumerate(['cuda:4','cuda:4','cuda:5','cuda:5']):
            log=(out/f'worker_{i}.log').open('x');logs.append(log)
            cmd=[sys.executable,'-u',str(Path(__file__)),'--stage','worker','--run-dir',str(args.run_dir),
                 '--device',d,'--shard-index',str(i)]
            processes.append(subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('worker failure')
            time.sleep(2)
        if any(p.returncode for p in processes):raise RuntimeError('worker failure')
        rows=[]
        for i in range(4):rows.extend(map(json.loads,(out/f'scores_{i}.jsonl').read_text().splitlines()))
        expected=p['payload']['baseline_protocol']['payload']['case_ids']
        main={r['id']:r for r in map(json.loads,(args.run_dir/'scores.jsonl').read_text().splitlines())}
        validate_rows(rows,expected,main)
        rows.sort(key=lambda r:r['id'])
        (out/'scores.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        atomic_json(out/'freeze.json',{'complete':True,'cases':len(rows),'elapsed_seconds':time.perf_counter()-start,
            'scores_sha256':file_hash(out/'scores.jsonl'),'protocol_sha256':file_hash(out/'protocol.json'),
            'source_sha256':file_hash(Path(__file__)),'answer_keys_read':False})
    finally:
        for child in processes:
            if child.poll() is None:child.terminate()
        for child in processes:
            if child.poll() is None:
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()


def score(args):
    out=args.run_dir/'deadline';frozen=json.loads((out/'freeze.json').read_text())
    if frozen['scores_sha256']!=file_hash(out/'scores.jsonl') or frozen['source_sha256']!=file_hash(Path(__file__)):
        raise ValueError('frozen result/source changed')
    if (out/'summary.json').exists():raise FileExistsError('already scored')
    main=list(map(json.loads,(args.run_dir/'evaluation/per_case.jsonl').read_text().splitlines()))
    scores={r['id']:r for r in map(json.loads,(out/'scores.jsonl').read_text().splitlines())}
    rows=[]
    for r in main:
        pred=scores[r['id']]['result']['prediction']
        rows.append({**r,'predictions':{**r['predictions'],'deadline_high':pred},
            'prediction_text':{**r['prediction_text'],'deadline_high':r['options'][pred]},
            'correct':{**r['correct'],'deadline_high':pred==r['gold']}})
    metrics={};pairs={}
    for split in ['train','validation','test']:
        selected=[r for r in rows if r['research_split']==split];c=sum(r['correct']['deadline_high'] for r in selected)
        metrics[split]={'n':len(selected),'correct':c,'accuracy':c/len(selected)}
        pairs[split]={b:paired(selected,'deadline_high',b) for b in ['motion_high','uniform_high','high448','low112','mid224']}
    summary={'metrics':metrics,'paired':pairs,'timing':{k:describe([r['result'][k] for r in scores.values()])
        for k in ['elapsed_seconds','forward_seconds','input_tokens','visual_tokens']},
        'status':'post-hoc fixed latest control, predictions independently frozen before grading',
        'source_freeze_sha256':file_hash(out/'freeze.json')}
    atomic_json(out/'summary.json',summary)
    (out/'per_case.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps(summary['metrics'],indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['launch','worker','score'],required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--device',default='cuda:4');p.add_argument('--shard-index',type=int,default=0)
    args=p.parse_args(argv)
    if args.stage=='launch':launch(args)
    elif args.stage=='worker':run_worker(args)
    else:score(args)
    return 0

if __name__=='__main__':raise SystemExit(main())
