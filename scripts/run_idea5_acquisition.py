#!/usr/bin/env python3
"""Past-only low/high-resolution and causal acquisition preflight on bench_v0."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.evaluate_forecast_mcq import file_hash, fingerprint, read_protocol, QwenEngine, label_tokens
from scripts.run_forecast_residual import read_split
from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.forecast_mcq import build_mcq_prompt, load_frame_bundle, frame_bundle_hash
from ttt_frame.idea5_acquisition import ARMS, chunk_motion, resolution_schedules, thumbnail, answer


def atomic_json(path, value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def sources():
    return {name:file_hash(REPO/name) for name in ["scripts/run_idea5_acquisition.py",
            "ttt_frame/idea5_acquisition.py", "scripts/run_forecast_residual.py"]}


def prepare(args):
    if args.protocol.exists(): raise FileExistsError(args.protocol)
    baseline=read_protocol(args.baseline_protocol); split=read_split(args.split_manifest)
    payload={"schema":"idea05_acquisition_preflight_v1","baseline_protocol":baseline,
             "split":split,"sources":sources(),"arms":list(ARMS),"seed":0,
             "frames":{"context":30,"recent":60,"decode_max_side":448,"chunk_size":6},
             "resolutions":[112,224,448],"budget":"one high chunk per cells [0,4),[4,8),[8,12),[12,15)",
             "motion_threshold":0.1,"future_media_used":False,"answer_keys_used":False,
             "query_used_by_selector":False,"model_training":False,
             "scope":"frozen Qwen perceptual/budget preflight; no learned memory-conditioned selector",
             "packing":"15 ordered six-frame video segments, identical for all new arms",
             "feedback":"none; policy settings predeclared, no label selection"}
    if set().union(*map(set,split["splits"].values())) != set(baseline["payload"]["case_ids"]):
        raise ValueError("split coverage mismatch")
    if args.protocol.resolve().is_relative_to(Path(baseline['payload']['root'])):
        raise ValueError("output inside source benchmark")
    rec={"payload":payload,"protocol_fingerprint":fingerprint(payload),
         "created_at":datetime.now(timezone.utc).isoformat(),
         "git_revision":subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    args.protocol.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.protocol,rec)
    return rec


def validate(path):
    rec=json.loads(path.read_text()); p=rec['payload']; b=p['baseline_protocol']
    from scripts.evaluate_forecast_mcq import source_hashes
    if (fingerprint(p)!=rec['protocol_fingerprint'] or p['sources']!=sources()
            or fingerprint(b['payload'])!=b['protocol_fingerprint']
            or b['payload']['source_hashes']!=source_hashes()):
        raise ValueError('protocol/source changed')
    if file_hash(Path(b['payload']['root'])/'inputs_mcq.jsonl')!=b['payload']['inputs_sha256']:
        raise ValueError('public inputs changed')
    return rec


def completed(path,ids,digest):
    result=set()
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if (row.get('id') not in ids or row['id'] in result or row.get('protocol_fingerprint')!=digest
                    or row.get('status')!='ok' or set(row.get('arms',{}))!=set(ARMS)
                    or any(x.get('prediction') not in 'ABCDEF' or x.get('status')!='ok'
                           for x in row['arms'].values())):
                raise ValueError('foreign, failed or incomplete previous row')
            result.add(row['id'])
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--baseline-protocol',type=Path,default=Path('runs/forecast_bench_v0/mcq_baseline_v1/protocol.json'))
    p.add_argument('--split-manifest',type=Path,default=Path('runs/forecast_bench_v0/research_day_split.json'))
    p.add_argument('--output',type=Path);p.add_argument('--device',default='cuda:4')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--ids',nargs='+')
    p.add_argument('--resume',action='store_true');args=p.parse_args(argv)
    if args.prepare: print(json.dumps(prepare(args),indent=2));return 0
    rec=validate(args.protocol); bp=rec['payload']['baseline_protocol']['payload']
    if not args.output or args.output.resolve().is_relative_to(Path(bp['root'])):
        raise ValueError('output required outside source data')
    if not 0<=args.shard_index<args.num_shards: raise ValueError('invalid shard')
    cases=load_forecast_cases(Path(bp['root']))
    if args.ids: cases=[c for c in cases if c.id in args.ids]
    cases=cases[args.shard_index::args.num_shards]
    ids={c.id for c in cases}
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    done=completed(args.output,ids,rec['protocol_fingerprint']); pending=[c for c in cases if c.id not in done]
    import torch
    torch.set_num_threads(args.threads);torch.manual_seed(0)
    for name,expected in bp['base_artifact_sha256'].items():
        if file_hash(Path(bp['model_path'])/name)!=expected: raise ValueError('model changed')
    engine=QwenEngine(bp['model_path'],args.device)
    if bp.get('vision_batched_sdpa'):
        from ttt_frame.vision_attention import EqualLengthVisionSDPA
        engine.vision_attention_patch=EqualLengthVisionSDPA(engine.model)
    if bp.get('vision_linear_patch'):
        from ttt_frame.vision_patch_embed import LinearVisionPatchEmbed
        engine.vision_patch_embed=LinearVisionPatchEmbed(engine.model)
    tokens=label_tokens(engine.processor.tokenizer)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('a' if args.resume else 'x') as out:
        for case in pending:
            started=time.perf_counter();bundle=load_frame_bundle(case,max_side=448)
            decode=time.perf_counter()-started
            images=bundle['context_images']+bundle['recent_images']
            times=bundle['context_actual_seconds']+bundle['recent_actual_seconds']
            requested=bundle['context_requested_seconds']+bundle['recent_requested_seconds']
            chunks=[images[i:i+6] for i in range(0,90,6)]
            chunk_times=[times[i:i+6] for i in range(0,90,6)]
            policy_start=time.perf_counter();motions=[chunk_motion(x) for x in chunks]
            schedules=resolution_schedules(case.id,motions)
            policy_seconds=time.perf_counter()-policy_start
            arms={}
            for arm in ARMS:
                sides=schedules[arm]
                result=answer(engine,chunks,chunk_times,sides,build_mcq_prompt(case),tokens)
                chosen=[thumbnail(im,side) for ch,side in zip(chunks,sides) for im in ch]
                result.update(resolutions=sides,high_frames=sum(s==448 for s in sides)*6,
                              low_thumbnail_frames=90 if arm.endswith('_high') else 0,
                              selected_pixel_sha256=frame_bundle_hash(chosen,times,requested))
                arms[arm]=result
                print(json.dumps({'id':case.id,'arm':arm,'prediction':result['prediction'],
                                  'tokens':result['input_tokens'],'seconds':result['elapsed_seconds']}),flush=True)
            row={'schema':'idea05_acquisition_scores_v1','id':case.id,'status':'ok',
                 'protocol_fingerprint':rec['protocol_fingerprint'],'device':args.device,
                 'question':case.question,'options':{o.label:o.text for o in case.options},
                 'sampling':{k:v for k,v in bundle.items() if not k.endswith('_images')},
                 'decode_seconds':decode,'policy_seconds':policy_seconds,'motion':motions,
                 'arms':arms,'episode_seconds':time.perf_counter()-started,
                 'answer_keys_used':False,'future_media_used':False}
            out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');out.flush()
    assert completed(args.output,ids,rec['protocol_fingerprint'])==ids
    atomic_json(args.output.with_suffix('.completion.json'),{'complete':True,'cases':len(ids),
                 'ids':sorted(ids),'protocol_fingerprint':rec['protocol_fingerprint'],
                 'scores_sha256':file_hash(args.output)})
    return 0

if __name__=='__main__':raise SystemExit(main())
