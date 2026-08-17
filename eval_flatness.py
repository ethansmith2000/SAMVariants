"""Weight-noise robustness: a direct flatness probe for trained checkpoints.

For each checkpoint, adds Gaussian noise to all weights at several magnitudes
and measures the validation-loss degradation. Flat minima degrade more slowly.

    python eval_flatness.py \
        --model_dirs model-output/run_a model-output/run_b \
        --tokenized_dataset_path /path/to/tokenized \
        --sigmas 0 0.01 0.02 0.05 0.1 --seeds 3

Noise is relative by default: for each param, noise = sigma * std(param) * N(0, I),
so a given sigma perturbs every layer proportionally to its own weight scale
(absolute noise with --noise_mode absolute). Embeddings/norms are included;
exclude 1D params with --skip_1d if norm-scale sensitivity dominates.

Outputs a table and writes <model_dir>/flatness.json for each checkpoint.
"""

import argparse
import json
import os

import datasets
import torch
from torch.utils.data import DataLoader

from transformer import Transformer


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    config["gradient_checkpointing"] = False
    model = Transformer(**config)
    state = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def make_eval_loader(tok_path, batch_size, num_batches):
    lm = datasets.load_from_disk(tok_path)["validation"]
    lm = lm.select(range(min(len(lm), batch_size * num_batches)))
    lm.set_format(type="torch", columns=["input_ids", "labels"])
    return DataLoader(lm, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def mean_loss(model, loader, device):
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)[:, :-1]
        targets = batch["labels"].to(device)[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            loss, _ = model(input_ids=input_ids, targets=targets)
        losses.append(loss.float())
    return torch.stack(losses).mean().item()


@torch.no_grad()
def perturbed_loss(model, loader, device, sigma, seed, noise_mode, skip_1d):
    if sigma == 0:
        return mean_loss(model, loader, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    originals = []
    for p in model.parameters():
        if skip_1d and p.dim() <= 1:
            continue
        originals.append((p, p.data.clone()))
        noise = torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype)
        scale = sigma * (p.data.std() if noise_mode == "relative" else 1.0)
        p.data.add_(noise, alpha=scale)
    loss = mean_loss(model, loader, device)
    for p, orig in originals:
        p.data.copy_(orig)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", nargs="+", required=True)
    ap.add_argument("--tokenized_dataset_path", required=True)
    ap.add_argument("--sigmas", nargs="+", type=float,
                    default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--num_batches", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--noise_mode", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--skip_1d", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = make_eval_loader(args.tokenized_dataset_path, args.batch_size, args.num_batches)

    all_results = {}
    for model_dir in args.model_dirs:
        model = load_model(model_dir, device)
        results = {}
        for sigma in args.sigmas:
            losses = [
                perturbed_loss(model, loader, device, sigma, seed=1000 + s,
                               noise_mode=args.noise_mode, skip_1d=args.skip_1d)
                for s in range(1 if sigma == 0 else args.seeds)
            ]
            t = torch.tensor(losses)
            results[sigma] = {"mean": t.mean().item(), "std": t.std().item() if len(losses) > 1 else 0.0}
        all_results[model_dir] = results
        with open(os.path.join(model_dir, "flatness.json"), "w") as f:
            json.dump({"noise_mode": args.noise_mode, "skip_1d": args.skip_1d,
                       "num_batches": args.num_batches, "results": results}, f, indent=2)
        del model
        torch.cuda.empty_cache()

    print(f"\nnoise_mode={args.noise_mode} skip_1d={args.skip_1d} "
          f"({args.num_batches} batches x bs{args.batch_size}, {args.seeds} seeds)\n")
    header = "sigma".ljust(8) + "".join(os.path.basename(d.rstrip('/'))[:24].ljust(26) for d in args.model_dirs)
    print(header)
    for sigma in args.sigmas:
        row = f"{sigma:<8g}"
        for d in args.model_dirs:
            r = all_results[d][sigma]
            base = all_results[d][args.sigmas[0]]["mean"]
            row += f"{r['mean']:.4f} (+{r['mean'] - base:.4f})".ljust(26)
        print(row)


if __name__ == "__main__":
    main()
