#!/usr/bin/env python3
"""Isolated single-case 20-write module timing using one fixed train trajectory."""
from __future__ import annotations
import argparse,json,sys,time,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_idea4_response import context,collect,get_model,sha,save_json
from ttt_frame.idea4_response_memory import initial_teacher,teacher_write


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--device',default='cuda:3')
    a=p.parse_args();a.output=a.run_dir;torch.set_num_threads(2)
    protocol,split,baseline,pdata=context(a);selection=json.loads((a.output/'selection.json').read_text());sid=split['splits']['train'][0]
    data=collect(a,[sid],a.device);models={k:get_model(a,name,pdata) for k,name in selection['selected'].items()};results={}
    for arm in ['teacher','fixed','conditioned']:
        timings=[];peak=[]
        for repeat in range(7):
            if arm=='teacher':wi,wo=initial_teacher(1,4,256,128,a.device)
            else:state=torch.zeros(1,4,len(models[arm].probes),256,device=a.device)
            torch.cuda.synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device);start=time.perf_counter()
            for step in range(20):
                if arm=='teacher':teacher_write(wi,wo,data['keys'][:,step],data['targets'][:,step],torch.ones(1,4,1,1,device=a.device))
                else:state=models[arm](state,data['keys'][:,step],data['targets'][:,step])
            torch.cuda.synchronize(a.device);elapsed=time.perf_counter()-start
            if repeat>=2:timings.append(elapsed/20);peak.append(torch.cuda.max_memory_allocated(a.device))
        results[arm]={'seconds_per_four_horizon_write':statistics.median(timings),'repeat_means_seconds':timings,'process_peak_allocated_bytes':max(peak)}
    save_json(a.output/'memory_benchmark.json',{'schema':'idea04_module_timing_v1','device':a.device,'gpu':torch.cuda.get_device_name(a.device),
        'case_id':sid,'source_sha256':sha(Path(__file__)),'results':results,'answer_files_read':False,
        'scope':'one case, four horizons, eight simultaneous tokens, 20 real mature writes, zero initial states, two warmup + five measured repeats, synchronized wall time; all three modules resident; excludes video/baseline/query/teacher training/trajectory generation',
        'speed_claim':'local unfused implementations only; process peaks include co-resident models and trace data; no end-to-end latency claim'})
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
