from copy import deepcopy
import json
from pathlib import Path
import pytest

from scripts import evaluate_idea5_acquisition as evaluator
from scripts import launch_idea5_acquisition as launcher
from scripts.run_idea5_acquisition import atomic_json
from scripts.evaluate_forecast_mcq import file_hash
from ttt_frame.idea5_acquisition import ARMS,resolution_schedules
from scripts.audit_idea5_inputs import compare


def sample(sid):
    schedules=resolution_schedules(sid,[0]*15)
    return {'schema':'idea05_acquisition_scores_v1','id':sid,'status':'ok',
        'protocol_fingerprint':'p','answer_keys_used':False,'future_media_used':False,
        'sampling':{'sampling':{'context_frames':30,'recent_frames':60,'max_side':448},
          'context_requested_seconds':[0,60], 'context_actual_seconds':[0,60],
          'recent_requested_seconds':[1800,1800.5],'recent_actual_seconds':[1800,1800.5]},
        'arms':{a:{'prediction':'A','status':'ok','label_logits':[2,1,0,0,0,0],
                   'label_probabilities':[.5,.1,.1,.1,.1,.1],'resolutions':schedules[a]} for a in ARMS}}


def protocol(ids):
    return {'protocol_fingerprint':'p','payload':{'baseline_protocol':{'payload':{'case_ids':ids}},
        'split':{'splits':{'train':ids[:1],'validation':ids[1:2],'test':ids[2:]}}}}


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


def test_read_scores_rejects_future_and_bad_budget(tmp_path):
    p=tmp_path/'scores.jsonl';row=sample('egolife_001');pr=protocol([row['id']])
    write_rows(p,[row]);assert evaluator.read_scores(p,pr)[row['id']]==row
    bad=deepcopy(row);bad['sampling']['recent_actual_seconds'][1]=1831;write_rows(p,[bad])
    with pytest.raises(ValueError,match='future'):evaluator.read_scores(p,pr)
    bad=deepcopy(row);bad['arms']['motion_high']['resolutions']=[448]*15;write_rows(p,[bad])
    with pytest.raises(ValueError,match='budget'):evaluator.read_scores(p,pr)


def test_read_scores_requires_all_cases_and_consistent_argmax(tmp_path):
    p=tmp_path/'scores.jsonl';row=sample('egolife_001');write_rows(p,[row])
    with pytest.raises(ValueError,match='coverage'):evaluator.read_scores(p,protocol([row['id'],'egolife_002']))
    row['arms']['low112']['prediction']='B';write_rows(p,[row])
    with pytest.raises(ValueError,match='scores'):evaluator.read_scores(p,protocol([row['id']]))


def test_freeze_uses_no_answer_files_and_is_immutable(tmp_path,monkeypatch):
    ids=['egolife_001','egolife_002','egolife_003'];p=tmp_path/'scores.jsonl'
    write_rows(p,[sample(i) for i in ids]);pr=tmp_path/'protocol.json';pr.write_text(json.dumps(protocol(ids)))
    monkeypatch.setattr(evaluator,'validate',lambda _:protocol(ids))
    original=Path.read_text
    reads=[]
    def guarded(path,*args,**kwargs):
        assert path.name not in {'answers.jsonl','metadata.jsonl'}
        reads.append(path.name)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    out=tmp_path/'evaluation'
    assert evaluator.freeze(p,pr,out)==3
    freeze=json.loads((out/'prediction_freeze.json').read_text())
    assert freeze['answer_keys_read'] is False
    assert freeze['predictions_sha256']==file_hash(out/'predictions.jsonl')
    with pytest.raises(FileExistsError):evaluator.freeze(p,pr,out)
    with (out/'predictions.jsonl').open('a') as f:f.write('changed')
    with pytest.raises(ValueError,match='frozen'):evaluator.score(p,out)


def test_paired_counts_treat_both_wrong_disagreement_separately():
    rows=[{'id':'one','correct':{'a':True,'b':False},'predictions':{'a':'A','b':'B'}},
          {'id':'two','correct':{'a':False,'b':True},'predictions':{'a':'C','b':'D'}},
          {'id':'three','correct':{'a':False,'b':False},'predictions':{'a':'E','b':'F'}}]
    result=evaluator.paired(rows,'a','b')
    assert (result['gains'],result['losses'],result['net'],result['prediction_disagreements'])==(1,1,0,3)
    assert result['gain_ids']==['one'] and result['loss_ids']==['two']


def test_merge_verifies_all_shard_hashes(tmp_path,monkeypatch):
    ids=['egolife_001','egolife_002','egolife_003'];pr=tmp_path/'protocol.json'
    monkeypatch.setattr(launcher,'validate',lambda _:protocol(ids))
    for i in range(2):
        p=tmp_path/f'scores_{i}.jsonl';write_rows(p,[sample(sid) for sid in ids[i::2]])
        atomic_json(p.with_suffix('.completion.json'),{'complete':True,'ids':ids[i::2],'scores_sha256':file_hash(p)})
    launcher.merge(pr,tmp_path,2)
    assert [json.loads(x)['id'] for x in (tmp_path/'scores.jsonl').read_text().splitlines()]==ids
    (tmp_path/'scores.jsonl').unlink()
    with (tmp_path/'scores_0.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='shard'):launcher.merge(pr,tmp_path,2)


def test_frame_audit_reports_pixel_confounds_without_hiding_time_mismatch():
    new=sample('egolife_001');new['sampling']['source_media']={'context':'same'}
    old={'id':new['id'],'sampling':deepcopy(new['sampling'])}
    old['sampling']['frame_bundle_sha256']='direct224'
    new['arms']['mid224']['selected_pixel_sha256']='448then224'
    result=compare([new],[old])
    assert result['source_times_equal']==1 and result['mid224_pixels_equal']==0
    old['sampling']['recent_actual_seconds'][1]=1800.4
    with pytest.raises(ValueError,match='timestamp'):compare([new],[old])
