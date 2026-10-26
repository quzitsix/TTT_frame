from copy import deepcopy
import pytest
from scripts.run_idea5_deadline import SIDES,validate_rows
from ttt_frame.idea5_acquisition import CELLS


def test_fixed_latest_has_identical_budget_without_motion_or_future():
    assert [i for i,s in enumerate(SIDES) if s==448]==[3,7,11,14]
    assert len(SIDES)==15
    assert all(SIDES[a:b].count(448)==1 for a,b in CELLS)


def test_validate_requires_same_source_pixels_and_full_coverage():
    row={'id':'egolife_001','resolutions':SIDES,'result':{'prediction':'A','status':'ok'},
         'answer_keys_used':False,'future_media_used':False,'source_frame_sha256':'pixels'}
    main={'egolife_001':{'sampling':{'frame_bundle_sha256':'pixels'}}}
    validate_rows([row],['egolife_001'],main)
    with pytest.raises(ValueError,match='IDs'):validate_rows([row,row],['egolife_001'],main)
    bad=deepcopy(row);bad['source_frame_sha256']='changed'
    with pytest.raises(ValueError,match='pixels'):validate_rows([bad],['egolife_001'],main)
