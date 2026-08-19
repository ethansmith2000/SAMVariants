"""Directional landscape probe: loss slices along mechanism-relevant directions.

For each checkpoint, computes a fresh minibatch gradient g and builds four
direction families:
  - grad:  g itself (Euclidean steepest ascent)
  - muon:  per-matrix Newton-Schulz orthogonalization of g (spectral geometry;
           what our muon-ascent perturbation moves along)
  - sign:  sign(g) (elementwise-normalized, an Adam-geometry proxy)
  - rand:  Gaussian direction (control)

Each direction is normalized per-param like the optimizer's "balanced" scheme,
then scaled in units of t = multiples of a reference step norm, and we record
loss(w + t*d) for t in ±{1,2,4,8} units. A quadratic fit gives directional
curvature. Winners under the extragradient story should differ along
grad/muon/sign directions but not along rand.

    python eval_directional.py --model_dirs <dirs...> \
        --tokenized_dataset_path <path> [--unit 0.25]

Writes <model_dir>/directional.json and prints a summary table.
"""

import argparse
import json
import os

import datasets
import torch
from torch.utils.data import DataLoader

from transformer import Transformer
from utils import zeropower_via_newtonschulz5


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    config["gradient_checkpointing"] = False
    model = Transformer(**config)
    state = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device)


def make_batches(tok_path, batch_size, num_batches, device):
    lm = datasets.load_from_disk(tok_path)["validation"]
    lm = lm.select(range(min(len(lm), batch_size * num_batches)))
    lm.set_format(type="torch", columns=["input_ids"])
    loader = DataLoader(lm, batch_size=batch_size, shuffle=False)
    return [b["input_ids"].to(device).long() for b in loader]


def batch_loss(model, tokens):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=tokens.is_cuda):
        loss, _ = model(input_ids=tokens[:, :-1], targets=tokens[:, 1:])
    return loss


def mean_grad(model, batches):
    model.zero_grad(set_to_none=True)
    for tokens in batches:
        (batch_loss(model, tokens) / len(batches)).backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return grads


def build_directions(model, grads, seed=0):
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    dirs = {"grad": {}, "muon": {}, "sign": {}, "rand": {}}
    for name, p in model.named_parameters():
        g = grads[name]
        dirs["grad"][name] = g
        if g.ndim == 2 and max(g.shape) <= 16384:
            dirs["muon"][name] = zeropower_via_newtonschulz5(g, steps=6).float()
        else:
            dirs["muon"][name] = g
        dirs["sign"][name] = torch.sign(g)
        dirs["rand"][name] = torch.randn(g.shape, generator=gen, device=g.device)
    return dirs


def normalize_balanced(direction, eps=1e-12):
    """Per-param unit norm scaled by sqrt(numel_p/total): total norm 1."""
    total = sum(d.numel() for d in direction.values())
    return {
        n: d / d.norm().clamp_min(eps) * (d.numel() / total) ** 0.5
        for n, d in direction.items()
    }


@torch.no_grad()
def loss_at_offset(model, batches, direction, t):
    params = dict(model.named_parameters())
    for n, d in direction.items():
        params[n].data.add_(d, alpha=t)
    total = sum(batch_loss(model, tokens).float() for tokens in batches) / len(batches)
    for n, d in direction.items():
        params[n].data.add_(d, alpha=-t)
    return total.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs="+", required=True)
    ap.add_argument("--tokenized_dataset_path", required=True)
    ap.add_argument("--num_batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    # offset unit in absolute (balanced-total-norm) units; 0.25 ~ pilot-scale
    # per-step update norm, so t is roughly "steps of distance"
    ap.add_argument("--unit", type=float, default=0.25)
    ap.add_argument("--ts", nargs="+", type=float,
                    default=[-8, -4, -2, -1, 1, 2, 4, 8])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batches = make_batches(args.tokenized_dataset_path, args.batch_size,
                           args.num_batches, device)

    for model_dir in args.model_dirs:
        model = load_model(model_dir, device)
        model.eval()
        grads = mean_grad(model, batches)
        directions = {k: normalize_balanced(v) for k, v in build_directions(model, grads).items()}
        base = loss_at_offset(model, batches, directions["rand"], 0.0)

        out = {"unit": args.unit, "base_loss": base, "profiles": {}, "curvature": {}}
        for fam, direction in directions.items():
            profile = {}
            for t in args.ts:
                profile[t] = loss_at_offset(model, batches, direction, t * args.unit) - base
            out["profiles"][fam] = profile
            # quadratic fit through the inner points: L(t) ~ a*t + 0.5*c*t^2
            inner = [t for t in args.ts if abs(t) <= 2]
            ys = torch.tensor([profile[t] for t in inner])
            ts_ = torch.tensor([t * args.unit for t in inner])
            A = torch.stack([ts_, 0.5 * ts_ ** 2], dim=1)
            sol = torch.linalg.lstsq(A, ys.unsqueeze(1)).solution.squeeze()
            out["curvature"][fam] = {"slope": sol[0].item(), "curv": sol[1].item()}

        with open(os.path.join(model_dir, "directional.json"), "w") as f:
            json.dump(out, f, indent=2)

        name = os.path.basename(model_dir.rstrip("/"))
        print(f"\n== {name} (base {base:.4f})")
        print("fam    " + "".join(f"{t:>9g}" for t in args.ts) + "     slope      curv")
        for fam in ["grad", "muon", "sign", "rand"]:
            prof = out["profiles"][fam]
            row = f"{fam:<7}" + "".join(f"{prof[t]:>9.4f}" for t in args.ts)
            c = out["curvature"][fam]
            print(row + f"  {c['slope']:>9.3f} {c['curv']:>9.3f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
