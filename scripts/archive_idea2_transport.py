#!/usr/bin/env python3
"""Verify complete Idea2 extraction/analysis provenance and archive small outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run=args.run_dir;out=args.output_dir
    torch.set_num_threads(2)
    if out.exists():raise ValueError('archive output already exists')
    protocol=load(run/'protocol.json');split=protocol['split'];expected={sid for ids in split['splits'].values() for sid in ids}
    if len(expected)!=200:raise ValueError('expected full 200-case research split')
    if out.resolve().is_relative_to(Path(protocol['root']).resolve()):raise ValueError('archive may not write benchmark')
    for filename,expected_sha in protocol['sources'].items():
        if sha(ROOT/filename)!=expected_sha:raise ValueError(f'source mismatch {filename}')
    if sha(Path(protocol['root'])/'inputs_mcq.jsonl')!=protocol['public_inputs_sha256']:raise ValueError('public inputs changed')
    files={p.stem:p for p in (run/'features').glob('*.pt')}
    if set(files)!=expected:raise ValueError('cache ID coverage mismatch')
    ref=load(ROOT/'runs/forecast_bench_v0/idea01_mcq_v1/features/cache_protocol.json')['protocol']['model_fingerprint']
    covered=set()
    for i in range(4):
        worker=load(run/f'worker_{i}.json');completion=load(run/f'worker_{i}.completion.json')
        if worker['model']!=ref:raise ValueError('visual model differs from frozen Idea1 encoder')
        if worker['protocol_sha256']!=sha(run/'protocol.json'):raise ValueError('worker protocol mismatch')
        if not completion['complete'] or completion['count']!=len(worker['cases']):raise ValueError('worker incomplete')
        if covered.intersection(worker['cases']):raise ValueError('overlapping workers')
        covered.update(worker['cases'])
    if covered!=expected or load(run/'launch.completion.json')['codes']!=[0]*4:raise ValueError('launch coverage incomplete')
    records=[]
    for sid,path in sorted(files.items()):
        cache=torch.load(path,map_location='cpu',weights_only=True)
        if cache['id']!=sid or cache['protocol_sha256']!=sha(run/'protocol.json'):raise ValueError('cache identity mismatch')
        if len(cache['features'])!=24:raise ValueError('wrong frame count')
        for x in cache['features']:
            if x.ndim!=2 or x.shape[1]!=256 or not torch.isfinite(x).all():raise ValueError('invalid features')
        for a in cache['sampling']:
            requested=[q[key] for q in protocol['schedule'] if q['role']==a['role'] for key in ('old_seconds','new_seconds')]
            if requested!=a['requested_seconds']:raise ValueError('sampling changed')
            if any(t<0 or t>r+1e-9 for t,r in zip(a['actual_seconds'],requested)):raise ValueError('noncausal timestamps')
            filename='context.mp4' if a['role']=='context' else 'input.mp4'
            expected_path=Path(protocol['root'])/'samples'/sid/filename
            if Path(a['path'])!=expected_path:raise ValueError('unexpected media path')
            stat=expected_path.stat()
            if (stat.st_size,stat.st_mtime_ns)!=(a['source_size'],a['source_mtime_ns']):raise ValueError('media stat changed')
        records.append({'id':sid,'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size,'token_counts':[len(x) for x in cache['features']]})
    case_rows=[json.loads(x) for x in (run/'per_case.jsonl').read_text().splitlines()]
    pairs=[json.loads(x) for x in (run/'pairs.jsonl').read_text().splitlines()]
    if len(case_rows)!=200 or {r['id'] for r in case_rows}!=expected:raise ValueError('analysis case coverage')
    keys={(r['id'],r['pair'],r['arm']) for r in pairs}
    desired={(sid,i,arm) for sid in expected for i in range(12) for arm in ('matched','permuted','wrong_video')}
    if len(pairs)!=7200 or keys!=desired:raise ValueError('analysis pair coverage')
    if sha(run/'train_pca.pt')!=load(run/'selection.json')['pca_sha256']:raise ValueError('PCA changed')
    out.mkdir(parents=True)
    for name in ['protocol.json','selection.json','summary.json','per_case.jsonl','pairs.jsonl','train_pca.pt']:
        shutil.copyfile(run/name,out/name)
    provenance=out/'provenance';provenance.mkdir()
    names=['launch.json','launch.completion.json','analysis.log']
    names += [f'worker_{i}{suffix}' for i in range(4) for suffix in ('.json','.completion.json','.log')]
    for name in names:shutil.copyfile(run/name,provenance/name)
    (out/'feature_manifest.json').write_text(json.dumps({'files':records,'model_fingerprint':ref,'complete':True,'source_media_check':'size and mtime match; decoded frame pixel hashes in per_case sampling','answers_read':False},indent=2)+'\n')
    manifest={'schema':'idea02_transport_archive_v1','cases':200,'natural_adjacent_pairs':2400,'arm_pair_rows':7200,'archive_script_sha256':sha(__file__),
              'source_run':str(run.resolve()),'files':{str(f.relative_to(out)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(out.rglob('*')) if f.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Archived {len(manifest["files"])} files, verified all 200 feature caches and 7200 arm/pair rows')


if __name__=='__main__':main()
