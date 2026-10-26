#!/usr/bin/env python3
"""Launch explicit GPU workers and merge only complete Idea5 score shards."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_idea5_acquisition import validate,completed,atomic_json
from scripts.evaluate_forecast_mcq import file_hash


def merge(protocol,output,workers):
    rec=validate(protocol); ids=rec['payload']['baseline_protocol']['payload']['case_ids']
    rows={}
    for i in range(workers):
        path=output/f'scores_{i}.jsonl'; expected=set(ids[i::workers])
        sidecar=json.loads(path.with_suffix('.completion.json').read_text())
        if (not sidecar.get('complete') or set(sidecar['ids'])!=expected
                or sidecar['scores_sha256']!=file_hash(path)
                or completed(path,expected,rec['protocol_fingerprint'])!=expected):
            raise ValueError('incomplete or modified shard')
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['id'] in rows:raise ValueError('duplicate case')
            rows[row['id']]=row
    if set(rows)!=set(ids):raise ValueError('incomplete merged cases')
    path=output/'scores.jsonl'
    if path.exists():raise FileExistsError(path)
    path.write_text(''.join(json.dumps(rows[i],ensure_ascii=False,allow_nan=False)+'\n' for i in ids))
    atomic_json(output/'scores.completion.json',{'complete':True,'cases':len(ids),
        'protocol_fingerprint':rec['protocol_fingerprint'],'scores_sha256':file_hash(path),
        'shards':{f'scores_{i}.jsonl':file_hash(output/f'scores_{i}.jsonl') for i in range(workers)}})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--devices',nargs='+',required=True)
    p.add_argument('--threads',type=int,default=2);args=p.parse_args(argv)
    rec=validate(args.protocol);args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'launch.json').exists():raise FileExistsError('choose new launch dir')
    import torch
    for device in set(args.devices):
        free,_=torch.cuda.mem_get_info(device)
        if free < args.devices.count(device)*8*2**30:raise ValueError('insufficient free device memory')
    workers=[];children=[];logs=[];start=time.perf_counter()
    for i,device in enumerate(args.devices):
        cmd=[sys.executable,'-u',str(REPO/'scripts/run_idea5_acquisition.py'),
             '--protocol',str(args.protocol),'--output',str(args.output_dir/f'scores_{i}.jsonl'),
             '--device',device,'--num-shards',str(len(args.devices)),'--shard-index',str(i),
             '--threads',str(args.threads)]
        workers.append({'index':i,'device':device,'command':cmd})
    atomic_json(args.output_dir/'launch.json',{'workers':workers,'protocol_fingerprint':rec['protocol_fingerprint']})
    try:
        for worker in workers:
            log=(args.output_dir/f"worker_{worker['index']}.log").open('x');logs.append(log)
            children.append(subprocess.Popen(worker['command'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT))
        while any(child.poll() is None for child in children):
            failed=[i for i,c in enumerate(children) if c.poll() not in (None,0)]
            if failed:raise RuntimeError(f'workers failed {failed}; inspect logs')
            time.sleep(2)
        if any(c.returncode for c in children):raise RuntimeError('worker failure')
        merge(args.protocol,args.output_dir,len(args.devices))
        atomic_json(args.output_dir/'launch.completion.json',{'complete':True,'elapsed_seconds':time.perf_counter()-start,
            'workers':len(workers),'devices':args.devices,'scores_sha256':file_hash(args.output_dir/'scores.jsonl'),
            'launcher_sha256':file_hash(Path(__file__))})
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            if c.poll() is None:
                try:c.wait(timeout=5)
                except subprocess.TimeoutExpired:c.kill();c.wait()
        for log in logs:log.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
