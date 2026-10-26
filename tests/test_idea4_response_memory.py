import torch
import pytest
from ttt_frame.idea4_response_memory import ResponseMemory, teacher_read, teacher_write, initial_teacher, select_probes
from ttt_frame.idea1_mcq_memory import _batched_update


def model(conditioned=False):
    torch.manual_seed(9)
    return ResponseMemory(torch.randn(8, 5), torch.randn(5, 3), conditioned=conditioned)


def test_teacher_parity_and_causal_write():
    wi, wo = initial_teacher(2, 4, 5, 7, "cpu")
    target, key = torch.randn(2, 4, 3, 5), torch.randn(2, 4, 3, 5)
    active = torch.tensor([0, 1, 1, 0]).reshape(1, 4, 1, 1)
    oldi, oldo = wi.clone(), wo.clone()
    expectedi, expectedo = wi.clone(), wo.clone()
    _batched_update(expectedi, expectedo, key, target, active, .05)
    teacher_write(wi, wo, key, target, active)
    torch.testing.assert_close(wi, expectedi)
    torch.testing.assert_close(wo, expectedo)
    assert torch.equal(wo[:, 0], oldo[:, 0])
    assert torch.equal(wi, oldi)  # Zero output initialization first update.
    assert not torch.equal(wo[:, 1], oldo[:, 1])


@pytest.mark.parametrize("conditioned", [False, True])
def test_response_state_update_does_not_modify_input_or_parameters(conditioned):
    m=model(conditioned);s=torch.randn(2, 4, 8, 5);k=torch.randn(2, 4, 3, 5);v=torch.randn_like(k)
    old=s.clone(); params={n:p.detach().clone() for n,p in m.named_parameters()}
    new=m(s,k,v)
    assert torch.equal(s,old)
    assert new.shape==s.shape and torch.isfinite(new).all()
    assert all(torch.equal(params[n],p) for n,p in m.named_parameters())
    assert torch.equal(m(s,k,v,torch.zeros(2,4,1,1)),s)


def test_kernel_psd_and_state_conditioning():
    m=model(True); s=torch.randn(2,4,8,5)
    with torch.no_grad():m.tau_state.fill_(1)
    k=m.kernel(s,m.probes,m.probes)
    assert torch.linalg.eigvalsh(k).min() > -1e-6
    assert not torch.allclose(k,m.kernel(s*2,m.probes,m.probes))
    m.tau_state.data.zero_()
    torch.testing.assert_close(m.kernel(s,m.probes,m.probes),m.kernel(s*2,m.probes,m.probes))


def test_candidate_permutation_and_probe_selection():
    m=model();s=torch.randn(2,4,8,5);q=torch.randn(2,4,6,5);perm=torch.tensor([2,0,4,1,5,3])
    torch.testing.assert_close(m.read(s,q[...,perm,:]),m.read(s,q)[...,perm,:])
    candidates=torch.randn(20,5); p=select_probes(candidates,8,m.projection)
    assert len(torch.unique(p,dim=0))==8
    torch.testing.assert_close(p,select_probes(candidates,8,m.projection))


def test_training_has_finite_gradients():
    m=model(True);s=torch.randn(1,4,8,5)*.1;k=torch.randn(1,4,3,5);v=torch.randn_like(k)
    for _ in range(2):s=m(s,k,v)
    s.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
