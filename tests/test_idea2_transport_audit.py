"""Independent audit split and real-key preflight invariants."""
import torch
import pytest

from ttt_frame.idea2_transport_audit import (
    aggregate_pairs, audit_matched_pair, fit_pca, matched_indices,
    pair_schedule, project_keys, split_correspondences,
)


def test_schedule_is_fixed_past_only_and_adjacent():
    pairs=pair_schedule()
    assert len(pairs)==12
    for p in pairs:
        assert p["new_seconds"]-p["old_seconds"]==.5
        assert 0<=p["old_seconds"]<p["new_seconds"]<(1800 if p["role"]=="context" else 30)


@pytest.mark.parametrize("n",[32,33,64,112])
def test_fit_gate_audit_have_disjoint_support_and_full_coverage(n):
    splits=split_correspondences(n,"case/0")
    assert len(splits[0])>=16 and min(map(len,splits[1:]))>=8
    assert sorted(torch.cat(splits).tolist())==list(range(n))
    for a,b in zip(splits,split_correspondences(n,"case/0")):assert torch.equal(a,b)


def test_pca_is_orthonormal_and_not_mutating_inputs():
    g=torch.Generator().manual_seed(3)
    train=torch.randn(120,32,generator=g);before=train.clone()
    pca=fit_pca([train[:60],train[60:]])
    torch.testing.assert_close(pca["basis"].T@pca["basis"],torch.eye(16),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(pca["mean"],train.mean(0))
    assert torch.equal(train,before)
    assert pca["token_count"]==120
    assert project_keys(train,pca).shape==(120,16)


def test_mutual_matching_rejects_nonmutual_duplicates():
    old=torch.eye(4);new=torch.stack([old[0],old[0],old[2],old[3]])
    oi,ni,conf=matched_indices(old,new,.7)
    assert oi.tolist()==[0,2,3] and ni.tolist()==[0,2,3]
    assert torch.all(conf==1)


def test_real_pair_audit_identity_and_permuted_control():
    g=torch.Generator().manual_seed(1)
    x=torch.randn(96,32,generator=g)
    pca=fit_pca([torch.randn(200,32,generator=g)])
    identity=audit_matched_pair(x,x,pca,key="known",regularization=.1)
    wrong=audit_matched_pair(x,x,pca,key="known",regularization=.1,permuted=True)
    assert identity["eligible"] and not identity["accepted"]
    assert identity["metrics"]["audit"]["identity_mse"]==0
    assert identity["metrics"]["audit"]["transported_mse"]<1e-12
    assert wrong["metrics"]["audit"]["identity_mse"]>0
    assert identity["audit_count"]==wrong["audit_count"]


def test_insufficient_support_returns_no_fitted_rotation():
    x=torch.eye(16);pca={"mean":torch.zeros(16),"basis":torch.eye(16)}
    r=audit_matched_pair(x,x,pca,key="few",regularization=0)
    assert not r["eligible"] and r["reason"]=="insufficient_matches" and "metrics" not in r


def test_gated_report_uses_identity_for_rejected_rotations():
    metric={"match_count":2,"identity_mse":4.,"transported_mse":1.}
    rows=[{"eligible":True,"accepted":False,"metrics":{s:metric for s in ["fit","gate","audit"]}},
          {"eligible":False,"accepted":False}]
    stats=aggregate_pairs(rows)
    assert stats["eligible_fraction"]==.5
    assert stats["audit"]["candidate_rmse"]==1
    assert stats["audit"]["gated_rmse"]==2


def test_validation_split_can_be_read_without_opening_benchmark_answers():
    # The compute module has no IO or benchmark-answer dependency.
    import inspect
    import ttt_frame.idea2_transport_audit as module
    source=inspect.getsource(module)
    assert "open(" not in source and "answers.json" not in source
