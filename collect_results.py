"""Collect every run's final eval into one grid. Usage: python collect_results.py

Reads slurm_logs/*.log for the last eval_loss and sweep_configs/*.json for the
config, so it works for finished and in-flight runs alike (in-flight runs are
marked and excluded from the grid, since their eval is at a different step).
"""
import glob, json, os, re

STEPS = 25000
rows = {}
for log in sorted(glob.glob("slurm_logs/*.log")):
    name = os.path.basename(log)[:-4]
    cfg_path = f"sweep_configs/{name}.json"
    if not os.path.exists(cfg_path):
        continue
    cfg = json.load(open(cfg_path))
    txt = open(log, errors="ignore").read()
    evals = re.findall(r"eval_loss: ([0-9.]+)", txt)
    if not evals:
        continue
    steps = re.findall(r"(\d+)/%d" % STEPS, txt)
    done = "Saving model to" in txt
    at_end = bool(steps) and int(steps[-1]) >= STEPS
    mode = str(cfg.get("mode", "")).lower()
    if mode == "adamw":
        asc, desc, rho = "-", "adamw", 0.0
    elif mode == "muon":
        asc, desc = "-", "muon-nesterov" if cfg.get("muon_nesterov") else "muon"
        rho = 0.0
    else:
        asc = cfg["hybrid_sam_ascent"][:4]
        desc = cfg["hybrid_sam_descent"]
        rho = float(cfg["hybrid_sam_rho"])
        if rho == 0:
            asc = "-"
    # pilot runs predate perturbation_scale and used absolute-norm rho, which
    # is NOT comparable to the relative (multiples-of-update-norm) sweeps
    scale = cfg.get("hybrid_sam_perturbation_scale", "absolute")
    rows[name] = dict(ascent=asc, descent=desc, rho=rho, scale=scale,
                      loss=float(evals[-1]), complete=at_end, saved=done)

BASE = {}
for r in rows.values():
    if r["rho"] == 0 and r["complete"]:
        BASE[r["descent"]] = min(BASE.get(r["descent"], 9), r["loss"])
BASE.setdefault("muon", BASE.get("muon", 9))
anchor = {"muon": BASE.get("muon"), "adam": BASE.get("adam")}

print(f"{'baselines:':<12}", {k: round(v, 4) for k, v in BASE.items()}, "\n")

for mode, sign in ((" LOOKAHEAD  (rho>0: perturb FORWARD along the descent direction)", 1),
                   (" ASCENT     (rho<0: perturb UPHILL along the gradient direction)", -1)):
    cells_any = any(r["rho"] * sign > 0 and r["scale"] == "relative" for r in rows.values())
    if not cells_any:
        continue
    print("#" * 74)
    print("#" + mode)
    print("#" * 74)
    for desc in ("muon", "adam"):
        rhos = sorted({abs(r["rho"]) for r in rows.values()
                       if r["descent"] == desc and r["rho"] * sign > 0
                       and r["scale"] == "relative"})
        if not rhos:
            continue
        b = anchor.get(desc)
        print(f"\n  perturb along <ascent>, update with {desc}"
              f"   (rho=0 baseline {b:.4f})" if b else f"\n  update with {desc}")
        print(f"  {'along':<9}" + "".join(f"{('|rho|='+str(r)):>13}" for r in rhos))
        for asc, label in (("mome", "momentum"), ("adam", "adam"), ("muon", "muon")):
            cells = []
            for rho in rhos:
                hit = [r for r in rows.values() if r["ascent"] == asc
                       and r["descent"] == desc and r["rho"] == rho * sign
                       and r["scale"] == "relative"]
                if not hit:
                    cells.append(f"{'-':>13}")
                elif not hit[0]["complete"]:
                    cells.append(f"{'(running)':>13}")
                else:
                    d = hit[0]["loss"] - b if b else 0
                    cells.append(f"{hit[0]['loss']:.4f}{d:+.4f}".rjust(13))
            if any(c.strip() != "-" for c in cells):
                print(f"  {label:<9}" + "".join(cells))
    print()

ref = {r["descent"]: r["loss"] for _, r in sorted(rows.items(), key=lambda x: x[1]["loss"], reverse=True)
       if r["rho"] == 0 and r["complete"]}
print("references:", ", ".join(f"{k}={v:.4f}" for k, v in sorted(ref.items(), key=lambda x: x[1])))
best = sorted((r for r in rows.values() if r["complete"] and r["rho"] > 0
               and r["scale"] == "relative"), key=lambda r: r["loss"])[:5]
print("\ntop 5:", ", ".join(f"{r['ascent']}->{r['descent']}@{r['rho']}={r['loss']:.4f}" for r in best))
