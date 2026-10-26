import json
import math
import numpy as np
from PIL import Image
import pytest

from ttt_frame.idea5_acquisition import (
    ARMS,CELLS,CausalMotionPolicy,chunk_motion,resolution_schedules,thumbnail,
)
from scripts.run_idea5_acquisition import completed


def test_every_budget_policy_has_same_four_high_chunks():
    for seed in range(8):
        for motions in ([0.0]*15,[1.0]*15,np.linspace(0,0.2,15).tolist()):
            schedules=resolution_schedules('egolife_001',motions,seed)
            for arm in ('uniform_high','random_high','motion_high'):
                assert schedules[arm].count(448)==4
                for start,stop in CELLS:
                    assert schedules[arm][start:stop].count(448)==1


def test_motion_decisions_do_not_depend_on_future():
    a=resolution_schedules('x',[.05]*7+[.0]*8)['motion_high']
    b=resolution_schedules('x',[.05]*7+[1.0]*8)['motion_high']
    assert a[:7]==b[:7]


def test_exact_deadline_and_threshold_behavior():
    p=CausalMotionPolicy(.1)
    assert [p.step(i,m) for i,m in enumerate([0,.11,1,1,0,0,0,0])]==[
        False,True,False,False,False,False,False,True]
    with pytest.raises(ValueError):p.step(10,0)


def test_random_schedule_is_independent_of_motion_and_repeatable():
    x=resolution_schedules('egolife_123',[0]*15)
    y=resolution_schedules('egolife_123',[1]*15)
    assert x['random_high']==y['random_high']
    assert x['uniform_high']==y['uniform_high']


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1])
def test_nonfinite_negative_motion_rejected(bad):
    with pytest.raises(ValueError):resolution_schedules('x',[bad]*15)
    with pytest.raises(ValueError):CausalMotionPolicy().step(0,bad)


def test_thumbnail_policy_only_retains_low_evidence():
    white=Image.new('RGB',(448,448),'white');black=Image.new('RGB',(448,448),'black')
    assert chunk_motion([white]*6)==0
    assert chunk_motion([white,black]*3)==1
    assert thumbnail(white,112).size==(112,112)
    assert white.size==(448,448)


def test_completed_rejects_incomplete_and_foreign_rows(tmp_path):
    p=tmp_path/'scores.jsonl'
    row={'id':'egolife_001','protocol_fingerprint':'p','status':'ok',
         'arms':{a:{'prediction':'A','status':'ok'} for a in ARMS}}
    p.write_text(json.dumps(row)+'\n');assert completed(p,{'egolife_001'},'p')=={'egolife_001'}
    with pytest.raises(ValueError):completed(p,{'egolife_002'},'p')
    del row['arms']['high448'];p.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):completed(p,{'egolife_001'},'p')
