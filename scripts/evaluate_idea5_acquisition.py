#!/usr/bin/env python3
"""Freeze all predictions without labels, then score and archive the Idea5 preflight."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import statistics

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_idea5_acquisition import validate,completed,atomic_json
from scripts.evaluate_forecast_mcq import file_hash
from ttt_frame.idea5_acquisition import ARMS,CELLS


def read_scores(path,protocol):
    p=protocol['payload']; ids=p['baseline_protocol']['payload']['case_ids']
    if completed(path,set(ids),protocol['protocol_fingerprint'])!=set(ids):
        raise ValueError('score coverage incomplete')
    rows={x['id']:x for x in map(json.loads,path.read_text().splitlines())}
    for sid,row in rows.items():
        if (row.get('schema')!='idea05_acquisition_scores_v1' or row.get('answer_keys_used') is not False
                or row.get('future_media_used') is not False):raise ValueError('invalid inference boundary')
        s=row['sampling']
        if (s['sampling']['context_frames'],s['sampling']['recent_frames'],s['sampling']['max_side'])!=(30,60,448):
            raise ValueError('unexpected sampling')
        for role,duration,offset in [('context',1800,0),('recent',30,1800)]:
            requested=s[f'{role}_requested_seconds'];actual=s[f'{role}_actual_seconds']
            if len(actual)!=len(requested) or any(a>r+1e-9 or a<offset or r>=offset+duration for a,r in zip(actual,requested)):
                raise ValueError('future or invalid sample time')
        for arm,record in row['arms'].items():
            probs=record['label_probabilities'];logits=record['label_logits'];res=record['resolutions']
            if (len(probs)!=6 or len(logits)!=6 or any(not math.isfinite(v) for v in probs+logits)
                    or min(probs)<0 or abs(sum(probs)-1)>1e-5
                    or record['prediction']!='ABCDEF'[max(range(6),key=lambda i:logits[i])]
                    or len(res)!=15):raise ValueError('invalid answer scores')
            if arm.endswith('_high'):
                if any(res[start:stop].count(448)!=1 for start,stop in CELLS):
                    raise ValueError('budget mismatch')
            else:
                side=int(arm[-3:])
                if res!=[side]*15:raise ValueError('uniform resolution mismatch')
    return rows


def freeze(scores,protocol_path,out):
    protocol=validate(protocol_path);rows=read_scores(scores,protocol)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    preds=[]
    split=protocol['payload']['split']
    for role,ids in split['splits'].items():
        for sid in ids:
            preds.append({'id':sid,'research_split':role,
                          'predictions':{a:rows[sid]['arms'][a]['prediction'] for a in ARMS}})
    (out/'predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in preds))
    shutil.copyfile(protocol_path,out/'protocol.json')
    atomic_json(out/'prediction_freeze.json',{'schema':'idea05_frozen_predictions_v1',
        'cases':len(preds),'arms':list(ARMS),'answer_keys_read':False,
        'configuration_selection':'none; all six fixed configurations reported',
        'protocol_fingerprint':protocol['protocol_fingerprint'],'scores_sha256':file_hash(scores),
        'predictions_sha256':file_hash(out/'predictions.jsonl'),
        'evaluator_sha256':file_hash(Path(__file__)),'created_at':datetime.now(timezone.utc).isoformat()})
    return len(preds)


def describe(values):
    values=sorted(values);n=len(values)
    return {'n':n,'mean':statistics.mean(values),'min':values[0],'max':values[-1],
            'p50':statistics.median(values),'p95':values[min(n-1,math.ceil(n*.95)-1)]}


def paired(rows,arm,base):
    gains=[r['id'] for r in rows if r['correct'][arm] and not r['correct'][base]]
    losses=[r['id'] for r in rows if r['correct'][base] and not r['correct'][arm]]
    return {'n':len(rows),'gains':len(gains),'losses':len(losses),'net':len(gains)-len(losses),
            'gain_ids':gains,'loss_ids':losses,
            'prediction_disagreements':sum(r['predictions'][arm]!=r['predictions'][base] for r in rows)}


def score(scores,out):
    frozen=json.loads((out/'prediction_freeze.json').read_text())
    if (file_hash(scores)!=frozen['scores_sha256'] or file_hash(out/'predictions.jsonl')!=frozen['predictions_sha256']
            or file_hash(Path(__file__))!=frozen['evaluator_sha256']):raise ValueError('frozen artifact changed')
    if (out/'summary.json').exists():raise FileExistsError('already scored')
    protocol=validate(out/'protocol.json');raw=read_scores(scores,protocol)
    root=Path(protocol['payload']['baseline_protocol']['payload']['root'])
    answers={x['id']:x['correct_answer'] for x in map(json.loads,(root/'answers.jsonl').read_text().splitlines())}
    baseline=REPO/'data/bench_v0_baseline_v1/qwen_full_icl_predictions.jsonl'
    old={x['id']:x['prediction'] for x in map(json.loads,baseline.read_text().splitlines())}
    if set(answers)!=set(raw) or set(old)!=set(raw):raise ValueError('answer or baseline coverage')
    cases=[]
    for row in map(json.loads,(out/'predictions.jsonl').read_text().splitlines()):
        sid=row['id'];g=answers[sid]
        if g not in 'ABCDEF' or len(g)!=1:raise ValueError('invalid answer')
        preds={**row['predictions'],'full_icl_old':old[sid]}
        cases.append({**row,'predictions':preds,'gold':g,'question':raw[sid]['question'],
            'options':raw[sid]['options'],'correct':{a:v==g for a,v in preds.items()},
            'prediction_text':{a:raw[sid]['options'][v] for a,v in preds.items()},
            'gold_text':raw[sid]['options'][g]})
    metrics={};pairs={};runtime={}
    for arm in (*ARMS,'full_icl_old'):
        metrics[arm]={};pairs[arm]={}
        for role in ['train','validation','test']:
            subset=[r for r in cases if r['research_split']==role]
            c=sum(r['correct'][arm] for r in subset)
            result={'n':len(subset),'correct':c,'accuracy':c/len(subset)}
            if arm in ARMS:
                result['mean_nll']=statistics.mean(-math.log(max(raw[r['id']]['arms'][arm]['label_probabilities']['ABCDEF'.index(r['gold'])],1e-30)) for r in subset)
            metrics[arm][role]=result
        for base in ['low112','mid224','high448','uniform_high','full_icl_old']:
            pairs[arm][base]={role:paired([r for r in cases if r['research_split']==role],arm,base)
                             for role in ['train','validation','test']}
        if arm in ARMS:
            runtime[arm]={field:describe([r['arms'][arm][field] for r in raw.values()]) for field in
                ['input_tokens','visual_tokens','preprocess_seconds','forward_seconds','elapsed_seconds','cuda_peak_allocated_bytes','high_frames']}
    summary={'schema':'idea05_acquisition_results_v1','metrics':metrics,'paired':pairs,'runtime':runtime,
        'shared_runtime':{field:describe([r[field] for r in raw.values()]) for field in ['decode_seconds','policy_seconds','episode_seconds']},
        'test_status':'exploratory reused silver labels, overlapping windows; no IID significance claim',
        'selection':'none; every policy fixed before label scoring','protocol_fingerprint':protocol['protocol_fingerprint'],
        'prediction_freeze_sha256':file_hash(out/'prediction_freeze.json'),'answers_sha256':file_hash(root/'answers.jsonl'),
        'baseline_sha256':file_hash(baseline),'scope':'perception and acquisition budget preflight, not complete learned memory interaction'}
    atomic_json(out/'summary.json',summary)
    (out/'per_case.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in cases))
    return summary


def archive(run,out):
    if out.exists():raise FileExistsError(out)
    eval_dir=run/'evaluation';summary=json.loads((eval_dir/'summary.json').read_text())
    frozen=json.loads((eval_dir/'prediction_freeze.json').read_text())
    if file_hash(run/'scores.jsonl')!=frozen['scores_sha256']:raise ValueError('score hash mismatch')
    out.mkdir(parents=True)
    for name in ['protocol.json','prediction_freeze.json','predictions.jsonl','summary.json','per_case.jsonl']:
        shutil.copyfile(eval_dir/name,out/name)
    shutil.copyfile(run/'scores.jsonl',out/'scores.jsonl')
    prov=out/'provenance';prov.mkdir()
    for name in ['launch.json','launch.completion.json','scores.completion.json','launcher.log','input_audit.json']:
        shutil.copyfile(run/name,prov/name)
    for i in range(4):
        for name in [f'scores_{i}.completion.json',f'worker_{i}.log']:
            shutil.copyfile(run/name,prov/name)
    lines=['# Idea5 acquisition preflight results','',
           'All policies fixed before scoring; reused silver-label test results are exploratory.','',
           '| Arm | Validation | Test | Accuracy |','|---|---:|---:|---:|']
    for arm,metric in summary['metrics'].items():
        v,t=metric['validation'],metric['test']
        lines.append(f"| {arm} | {v['correct']}/{v['n']} | {t['correct']}/{t['n']} | {t['accuracy']:.2%} |")
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    (out/'.gitattributes').write_text('* -text\n')
    files={str(f.relative_to(out)):{'sha256':file_hash(f),'bytes':f.stat().st_size} for f in sorted(out.rglob('*')) if f.is_file()}
    atomic_json(out/'manifest.json',{'schema':'idea05_acquisition_archive_v1','files':files,
        'protocol_fingerprint':summary['protocol_fingerprint'],'cases':200})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','score','archive'],required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path)
    args=p.parse_args(argv);run=args.run_dir
    if args.stage=='freeze':print('Frozen',freeze(run/'scores.jsonl',run/'protocol.json',run/'evaluation'),'cases')
    elif args.stage=='score':
        summary=score(run/'scores.jsonl',run/'evaluation')
        print(json.dumps({k:v['test'] for k,v in summary['metrics'].items()},indent=2))
    else:
        if not args.output_dir:raise ValueError('--output-dir required')
        archive(run,args.output_dir);print('Archived',args.output_dir)
    return 0

if __name__=='__main__':raise SystemExit(main())
