#!/usr/bin/env python3
"""Validate frozen Idea4 stage artifacts and all-case old-teacher parity.

Use before fit, forecast, and score. This independent audit was added after
teacher protocol freeze; it never changes the source/teacher protocol or reads
answer labels. Exact dependency hashes are preserved in a separate report.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_idea4_response import context,sha,load,save_json,BASE,ids_for
from ttt_frame.idea1_mcq_readout import SharedRidgeReadout,pool_forecasts


def audit(directory:Path,stage:str):
    args=argparse.Namespace(output=directory)
    protocol,split,baseline,pdata=context(args);ids=ids_for(split)
    teacher=json.loads((directory/'teacher_completion.json').read_text())
    if not teacher['complete'] or set(teacher['files'])!=set(ids):raise ValueError('incomplete teacher files')
    readout=SharedRidgeReadout.from_dict(load(BASE/'mcq/shared_readout.pt'));text=load(BASE/'text_features.pt')['records']
    rows=[];dependencies={'protocol.json':sha(directory/'protocol.json'),'teacher_completion.json':sha(directory/'teacher_completion.json')}
    for sid in ids:
        file=directory/'trajectories'/f'{sid}.pt'
        if sha(file)!=teacher['files'][sid]:raise ValueError(f'trajectory hash mismatch: {sid}')
        new=load(file);oldfile=BASE/'forecasts'/f'{sid}_forecasts.pt';old=load(oldfile)
        if new['id']!=sid or new['protocol_sha256']!=dependencies['protocol.json']:raise ValueError('trajectory provenance mismatch')
        a=new['teacher_forecast'];b=old['last_forecasts']['residual']
        if not torch.allclose(a,b,rtol=1e-4,atol=1e-5):raise ValueError(f'teacher replay differs: {sid}')
        recent=new['baseline']*baseline.std+baseline.mean;oldrecent=old['last_forecasts']['recent_only']
        if not torch.allclose(recent,oldrecent,rtol=1e-4,atol=1e-5):raise ValueError(f'recent replay differs: {sid}')
        features=pool_forecasts(torch.stack([a,b,recent,oldrecent])).unsqueeze(0)
        pred=readout.predict_scores(features,text[sid]['embeddings'].unsqueeze(0))[0].argmax(-1)
        if pred[0]!=pred[1] or pred[2]!=pred[3]:raise ValueError('replay prediction mismatch')
        rows.append({'id':sid,'teacher_max_abs':float((a-b).abs().max()),'recent_max_abs':float((recent-oldrecent).abs().max()),
                     'prediction_equal':True,'trajectory_sha256':teacher['files'][sid],'old_forecast_sha256':sha(oldfile)})
    if stage in ['forecast','score']:
        selection=json.loads((directory/'selection.json').read_text());dependencies['selection.json']=sha(directory/'selection.json')
        for record in selection['records']:
            path=directory/'models'/f"{record['name']}.pt"
            if sha(path)!=record['sha256']:raise ValueError('model hash mismatch')
            model=load(path)
            if model['train_ids']!=split['splits']['train'] or model['protocol_sha256']!=dependencies['protocol.json']:raise ValueError('model provenance mismatch')
            dependencies[str(path.relative_to(directory))]=sha(path)
    if stage=='score':
        completion=json.loads((directory/'forecast_completion.json').read_text())
        if not completion['complete'] or set(completion['forecast_files'])!=set(ids):raise ValueError('incomplete forecasts')
        for sid,digest in completion['forecast_files'].items():
            if sha(directory/'forecasts'/f'{sid}.pt')!=digest:raise ValueError('forecast changed')
        if sha(directory/'predictions.jsonl')!=completion['predictions_sha256']:raise ValueError('predictions changed')
        dependencies['forecast_completion.json']=sha(directory/'forecast_completion.json')
        dependencies['predictions.jsonl']=sha(directory/'predictions.jsonl')
    return {'audit_passed':True,'stage_before':stage,'cases':len(ids),'answer_files_read':False,
       'source_sha256':sha(Path(__file__)),'dependencies':dependencies,'teacher_max_abs':max(r['teacher_max_abs'] for r in rows),
       'recent_max_abs':max(r['recent_max_abs'] for r in rows),'all_predictions_equal':True,'cases_parity':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--before',choices=['fit','forecast','score'],required=True)
    args=p.parse_args();torch.set_num_threads(2);result=audit(args.run_dir,args.before)
    save_json(args.run_dir/f'audit_before_{args.before}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['dependencies','cases_parity']},indent=2))
