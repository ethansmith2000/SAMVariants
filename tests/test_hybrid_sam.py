"""Correctness tests for HybridSAM. Run with: python tests/test_hybrid_sam.py

All tests run on CPU with tiny tensors; the whole suite takes a few seconds.
"""

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hybrid_sam import HybridSAM
from muon import Muon


def make_params(seed=0, sizes=((8, 8), (8, 24), (8,))):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s)) for s in sizes]


def fake_grads(params, seed):
    torch.manual_seed(seed)
    for p in params:
        p.grad = torch.randn_like(p)


def loss_grads(params):
    """Deterministic 'loss' whose gradient depends on current param values,
    emulating real training where grads are computed at the perturbed point."""
    for p in params:
        p.grad = 0.1 * p.data + torch.sin(p.data)


def test_rho_zero_matches_adamw():
    p1 = make_params(seed=1)
    p2 = copy.deepcopy(p1)
    opt1 = HybridSAM(p1, lr=1e-3, rho=0.0, ascent="momentum", descent="adam",
                     beta1=0.9, beta2=0.999, weight_decay=0.0)
    opt2 = torch.optim.AdamW(p2, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for step in range(20):
        fake_grads(p1, seed=100 + step)
        fake_grads(p2, seed=100 + step)
        opt1.step()
        opt2.step()
    for a, b in zip(p1, p2):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_rho_zero_matches_muon():
    p1 = make_params(seed=2)
    p2 = copy.deepcopy(p1)
    kw = dict(lr=1e-3, beta1=0.95, beta2=0.999, weight_decay=0.01, ns_steps=6,
              nesterov=False, muon_max_dim=16384)
    opt1 = HybridSAM(p1, rho=0.0, ascent="muon", descent="muon", **kw)
    opt2 = Muon(p2, **kw)
    for step in range(10):
        fake_grads(p1, seed=200 + step)
        fake_grads(p2, seed=200 + step)
        opt1.step()
        opt2.step()
    for a, b in zip(p1, p2):
        assert torch.equal(a, b), (a - b).abs().max()


def test_perturbation_sign_and_norm():
    """Positive rho must perturb along -exp_avg (MSAM lookahead), per-param unit norm."""
    rho = 0.25
    p_ref = make_params(seed=3)
    p_sam = copy.deepcopy(p_ref)
    opt_ref = HybridSAM(p_ref, lr=1e-3, rho=0.0, ascent="momentum", descent="adam",
                        beta1=0.9, weight_decay=0.0)
    opt_sam = HybridSAM(p_sam, lr=1e-3, rho=rho, ascent="momentum", descent="adam",
                        beta1=0.9, weight_decay=0.0, perturbation_norm="per_param")
    fake_grads(p_ref, seed=300)
    fake_grads(p_sam, seed=300)
    opt_ref.step()
    opt_sam.step()
    for a_ref, a_sam, ref_state in zip(p_ref, p_sam, [opt_ref.state[p] for p in p_ref]):
        m = ref_state["exp_avg"]  # identical buffers in both runs after step 1
        expected = a_ref.data - rho * m / m.norm()
        assert torch.allclose(a_sam.data, expected, atol=1e-6)
        eps_applied = a_sam.data - a_ref.data
        assert abs(eps_applied.norm().item() - rho) < 1e-5


def test_balanced_norm_budget():
    params = make_params(seed=4, sizes=((4, 4), (16, 16), (32,)))
    opt = HybridSAM(params, lr=1e-3, rho=0.5, ascent="momentum", descent="adam",
                    weight_decay=0.0, perturbation_norm="balanced")
    fake_grads(params, seed=400)
    opt.step()
    total_numel = sum(p.numel() for p in params)
    total_sq = 0.0
    for p in params:
        eps_p = opt.state[p]["perturb"]
        expected = 0.5 * (p.numel() / total_numel) ** 0.5
        assert abs(eps_p.norm().item() - expected) < 1e-5
        total_sq += eps_p.norm().item() ** 2
    assert abs(total_sq ** 0.5 - 0.5) < 1e-5


def test_remove_and_unperturbed_roundtrip():
    params = make_params(seed=5)
    opt = HybridSAM(params, lr=1e-2, rho=0.3, ascent="muon", descent="adam",
                    muon_fallback_ascent="momentum", weight_decay=0.01)
    for step in range(5):
        loss_grads(params)
        opt.step()
    snap = [p.data.clone() for p in params]
    with opt.unperturbed():
        for p, s in zip(params, snap):
            assert not torch.equal(p.data, s)  # actually moved
    for p, s in zip(params, snap):
        assert torch.equal(p.data, s)  # bit-exact restore
    # permanent removal: params must equal w = w̃ - eps
    expected = [p.data - opt.state[p]["perturb"] for p in params]
    opt.remove_perturbation()
    for p, e in zip(params, expected):
        assert torch.equal(p.data, e)
        assert opt.state[p].get("perturb") is None
    opt.remove_perturbation()  # idempotent


def test_no_leak_when_grad_goes_none():
    """A param that stops receiving grads must still get its perturbation removed."""
    params = make_params(seed=6)
    opt = HybridSAM(params, lr=1e-2, rho=0.3, ascent="momentum", descent="adam",
                    weight_decay=0.0)
    loss_grads(params)
    opt.step()
    frozen = params[0]
    w_clean = frozen.data - opt.state[frozen]["perturb"]  # its clean iterate
    frozen.grad = None
    for p in params[1:]:
        p.grad = torch.randn_like(p)
    opt.step()
    # no grad -> no descent, no new perturbation; must sit exactly at clean w
    assert torch.equal(frozen.data, w_clean)
    assert opt.state[frozen].get("perturb") is None


def test_perturbation_start_step():
    params = make_params(seed=7)
    opt = HybridSAM(params, lr=1e-3, rho=0.5, ascent="momentum", descent="adam",
                    weight_decay=0.0, perturbation_start_step=3)
    for step in range(1, 6):
        fake_grads(params, seed=700 + step)
        opt.step()
        has_perturb = any(opt.state[p].get("perturb") is not None for p in params)
        assert has_perturb == (step > 3), f"step {step}"


def test_ascent_beta1_separate_buffer():
    params = make_params(seed=8)
    opt = HybridSAM(params, lr=1e-3, rho=0.2, ascent="momentum", descent="adam",
                    beta1=0.9, ascent_beta1=0.5, weight_decay=0.0,
                    perturbation_norm="per_param")
    for step in range(3):
        fake_grads(params, seed=800 + step)
        opt.step()
    for p in params:
        st = opt.state[p]
        assert "exp_avg_ascent" in st
        assert not torch.allclose(st["exp_avg_ascent"], st["exp_avg"])
        expected = -0.2 * st["exp_avg_ascent"] / st["exp_avg_ascent"].norm()
        assert torch.allclose(st["perturb"], expected, atol=1e-6)


def test_state_dict_roundtrip():
    params = make_params(seed=9)
    opt = HybridSAM(params, lr=1e-2, rho=0.3, ascent="muon", descent="muon",
                    muon_fallback_ascent="adam", weight_decay=0.01)
    for _ in range(4):
        loss_grads(params)
        opt.step()
    # checkpoint mid-training: perturbed params + optimizer state
    params2 = [torch.nn.Parameter(p.data.clone()) for p in params]
    opt2 = HybridSAM(params2, lr=1e-2, rho=0.3, ascent="muon", descent="muon",
                     muon_fallback_ascent="adam", weight_decay=0.01)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    for _ in range(3):
        loss_grads(params)
        opt.step()
        loss_grads(params2)
        opt2.step()
    for a, b in zip(params, params2):
        assert torch.equal(a.data, b.data), (a.data - b.data).abs().max()


def test_second_moment_only_when_needed():
    params = make_params(seed=10, sizes=((8, 8), (8,)))
    opt = HybridSAM(params, lr=1e-3, rho=0.1, ascent="muon", descent="muon",
                    muon_fallback_ascent="skip", weight_decay=0.0)
    fake_grads(params, seed=1000)
    opt.step()
    assert "exp_avg_sq" not in opt.state[params[0]]  # pure muon path
    assert "exp_avg_sq" in opt.state[params[1]]      # 1D -> adam descent fallback


def test_stats_tracking():
    params = make_params(seed=11)
    opt = HybridSAM(params, lr=1e-3, rho=0.5, ascent="momentum", descent="adam",
                    weight_decay=0.0, track_stats=True)
    fake_grads(params, seed=1100)
    opt.step()
    stats = opt.last_stats
    assert abs(stats["perturb_norm"] - 0.5) < 1e-4
    # perturbation is along -exp_avg = -grad-ish at step 1 -> negative cosine
    assert stats["cos_perturb_grad"] < -0.9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
