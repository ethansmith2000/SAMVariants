"""How many *actual* steps of lookahead is rho worth?

rho displaces by rho * ||u_t|| along the (unit) perturbation direction, where
u_t is the update we are about to apply. For a MATCHED geometry that direction
IS u_t's direction, so the displacement is exactly rho * u_t. Whether that
equals "where we would be after rho steps" depends only on how straight the
trajectory is:

    straightness S(k) = ||sum_{i<k} u_i|| / sum_{i<k} ||u_i||      (1 = straight line)
    actual k-step displacement magnitude = S(k) * k * ||u||

So rho=4 lands 4 step-lengths out, while 4 real steps only get S(4)*4 out --
overshoot factor 1/S(4). If Muon's trajectory is straighter than Adam's, Muon
tolerates (and wants) larger rho, which would explain the descent-side split.

Also reports cos(adam_dir, muon_dir): for CROSS geometries the perturbation is
not along the path at all, but rho step-lengths sideways.
"""
import json, sys
import torch, datasets
from torch.utils.data import DataLoader
from transformer import Transformer
from hybrid_sam import HybridSAM, _adam_direction, _muon_direction

DATA = "/workspace/data/tokenized/openwebtext_gpt2_bs1024"
STEPS, K = int(sys.argv[1]) if len(sys.argv) > 1 else 400, 8
dev = "cuda"

ds = datasets.load_from_disk(DATA)["train"].select(range(STEPS * 8 + 64))
ds.set_format(type="torch", columns=["input_ids"])
loader = DataLoader(ds, batch_size=8, shuffle=False)

def run(descent):
    torch.manual_seed(0)
    m = Transformer(dim=1024, depth=12, heads=8, ff_mult=4, vocab_size=50257,
                    max_seq_len=1024, gradient_checkpointing=True).to(dev)
    groups = [{"params": [p], "weight_decay": 0.0 if (p.dim() <= 1 or "embed" in n) else 0.01,
               "lr": 4e-4} for n, p in m.named_parameters()]
    opt = HybridSAM(groups, lr=4e-4, muon_lr=6e-3, rho=0.0,
                    perturb_with=descent, update_with=descent,
                    beta1=0.95, beta2=0.95, weight_decay=0.01,
                    muon_fallback_ascent="skip")
    # a few representative square weight matrices
    watch = [(n, p) for n, p in m.named_parameters()
             if p.ndim == 2 and max(p.shape) <= 16384][:4]
    hist = {n: [] for n, _ in watch}
    rel = {n: [] for n, _ in watch}   # ||update|| / ||W||: puts geometries on a common scale
    cross = []
    # track our own second moment for the cross-geometry angle: pure-muon
    # descent never allocates exp_avg_sq (lazy allocation in HybridSAM)
    v_local = torch.zeros_like(watch[0][1])
    it = iter(loader)
    for step in range(STEPS):
        tok = next(it)["input_ids"].to(dev).long()
        loss, _ = m(input_ids=tok[:, :-1], targets=tok[:, 1:])
        loss.backward()
        v_local.mul_(0.95).addcmul_(watch[0][1].grad, watch[0][1].grad, value=0.05)
        before = {n: p.data.clone() for n, p in watch}
        opt.step()
        for n, p in watch:
            hist[n].append((p.data - before[n]).flatten().float())
            rel[n].append(((p.data - before[n]).norm() / p.data.norm()).item())
            if len(hist[n]) > K:
                hist[n].pop(0)
        if step > 20 and step % 25 == 0:      # cross-geometry angle
            n, p = watch[0]
            st = opt.state[p]; g = opt.param_groups[0]
            v = st.get("exp_avg_sq", v_local)
            a = _adam_direction(st["exp_avg"], v, st["step"], 0.95, 0.95, 1e-8)
            mu = _muon_direction(st["exp_avg"], st["exp_avg"], 0.95, 6, False, rescale=False)
            cross.append(torch.nn.functional.cosine_similarity(
                a.flatten().float(), mu.flatten().float(), dim=0).item())
        opt.zero_grad(set_to_none=True)
        del before

    out = {}
    for k in (1, 2, 4, 8):
        vals = []
        for n, _ in watch:
            us = hist[n][-k:]
            num = torch.stack(us).sum(0).norm().item()
            den = sum(u.norm().item() for u in us)
            vals.append(num / max(den, 1e-12))
        out[k] = sum(vals) / len(vals)
    c1 = []
    for n, _ in watch:
        c1.append(torch.nn.functional.cosine_similarity(
            hist[n][-2], hist[n][-1], dim=0).item())
    r = sum(sum(v[-50:]) / len(v[-50:]) for v in rel.values()) / len(rel)
    return out, sum(c1) / len(c1), (sum(cross) / len(cross) if cross else float("nan")), r

print(f"{'descent':<8} {'cos(u_t,u_t+1)':>15} " + " ".join(f"{'S('+str(k)+')':>7}" for k in (1,2,4,8))
      + f" {'rho4 overshoot':>15} {'cos(a,m)':>9} {'|u|/|W|':>9} {'rho*|u|/|W|@4':>14}")
for d in ("muon", "adam"):
    S, c1, cx, r = run(d)
    print(f"{d:<8} {c1:>15.3f} " + " ".join(f"{S[k]:>7.3f}" for k in (1,2,4,8))
          + f" {1.0/max(S[4],1e-9):>14.2f}x {cx:>9.3f} {100*r:>8.3f}% {100*4*r:>13.2f}%")
