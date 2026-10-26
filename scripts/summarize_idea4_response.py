#!/usr/bin/env python3
"""Archive completed Idea4 prediction/score/model artifacts without raw video."""
from __future__ import annotations
import argparse,csv,json,shutil,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.audit_idea4_response import audit
from scripts.run_idea4_response import load,sha,save_json


def archive(run:Path,out:Path):
    if out.exists():raise ValueError('archive destination must be new')
    check=audit(run,'score');rows=[json.loads(x) for x in (run/'per_case.jsonl').read_text().splitlines()]
    if len(rows)!=check['cases'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('incomplete per-case scores')
    out.mkdir(parents=True)
    names=['protocol.json','probes.pt','selection.json','teacher_completion.json','forecast_completion.json',
           'predictions.jsonl','per_case.jsonl','diagnostics.json','summary.json',
           'audit_before_fit.json','audit_before_forecast.json','audit_before_score.json']
    names += [name for name in ['teacher.log','fit.log','forecast.log','memory_benchmark.json'] if (run/name).exists()]
    for name in names:shutil.copyfile(run/name,out/name)
    shutil.copytree(run/'models',out/'models')
    arms=list(rows[0]['predictions'])
    with (out/'per_case.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','research_split']+[f'{a}_{suffix}' for a in arms for suffix in ['prediction','correct']],lineterminator='\n');writer.writeheader()
        for r in rows:writer.writerow({'id':r['id'],'research_split':r['research_split'],**{f'{a}_prediction':r['predictions'][a] for a in arms},**{f'{a}_correct':r['correct'][a] for a in arms}})
    errors=[]
    for r in rows:
        forecasts=load(run/'forecasts'/f"{r['id']}.pt");teacher=forecasts['teacher']
        errors.append({'id':r['id'],'research_split':r['research_split'],'mse_to_teacher':{a:float((v-teacher).square().mean()) for a,v in forecasts.items()}})
    (out/'forecast_errors.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in errors))
    metrics=json.loads((out/'summary.json').read_text())['metrics']
    md=['# Idea4 finite-response diagnostic','', 'This is an Idea1 residual-MLP teacher prototype, not the complete Idea4 or official Spatial-TTT.',
        'Warm20/copy/oracle require teacher responses; only the full arms start from zero and operate without a teacher at deployment.','',
        '| Arm | Validation | Exploratory test |','|---|---:|---:|']
    for a,m in metrics.items():md.append(f"| {a} | {m['validation']['correct']}/32 | {m['test']['correct']}/57 |")
    md += ['', 'All200 forecasts are frozen before a separate score stage opens silver answer labels.',
           'Teacher trajectories (2.5GB) and full forecast tensors stay in the source run; their SHA256 values are archived.',
           'The probe/model checkpoints are included. See docs/IDEA04_BENCH_V0_RESULTS.md for limitations and reproduction.','']
    (out/'results.md').write_text('\n'.join(md))
    files={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
    save_json(out/'manifest.json',{'schema':'idea04_response_archive_v1','source_run':str(run.resolve()),'cases':len(rows),
        'answer_files_read_by_archiver':False,'sources':{'scripts/summarize_idea4_response.py':sha(Path(__file__)),'scripts/audit_idea4_response.py':sha(ROOT/'scripts/audit_idea4_response.py')},'files':files})
    print('Archived',len(rows),'cases and',len(files),'files')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);archive(a.run_dir,a.output_dir)
