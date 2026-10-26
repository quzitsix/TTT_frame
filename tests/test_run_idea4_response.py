import json
from argparse import Namespace
import pytest
import torch
from scripts.run_idea4_response import save_pt, load, sha, collect


def test_per_case_save_compacts_batch_storage(tmp_path):
    batch=torch.randn(20,12,9)
    path=tmp_path/'case.pt'
    save_pt(path,{'values':batch[3],'nested':[batch[2]]})
    assert path.stat().st_size<10000
    got=load(path)
    torch.testing.assert_close(got['values'],batch[3])
    assert got['values'].untyped_storage().nbytes()==12*9*4


def test_collect_rejects_changed_trajectory_before_loading(tmp_path):
    (tmp_path/'trajectories').mkdir();(tmp_path/'protocol.json').write_text('{}')
    file=tmp_path/'trajectories'/'one.pt';save_pt(file,{'id':'one','protocol_sha256':sha(tmp_path/'protocol.json')})
    (tmp_path/'teacher_completion.json').write_text(json.dumps({'complete':True,'files':{'one':sha(file)}}))
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash changed'):collect(Namespace(output=tmp_path),['one'],'cpu')
