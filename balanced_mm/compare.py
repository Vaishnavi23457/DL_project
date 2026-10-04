#!/usr/bin/env python
"""
compare.py - tabulate several runs (history.json / summary.json) side by side.

    python compare.py runs/syn_none runs/syn_opm runs/syn_ogm runs/syn_both
    python compare.py runs/syn_*  --last 3      # average the last 3 epochs instead of best epoch
"""
import argparse
import json
import os
import sys


def load(run_dir):
    with open(os.path.join(run_dir, "history.json")) as f:
        hist = json.load(f)
    cfg = {}
    p = os.path.join(run_dir, "config.json")
    if os.path.exists(p):
        with open(p) as f:
            cfg = json.load(f)
    summ = {}
    p = os.path.join(run_dir, "summary.json")
    if os.path.exists(p):
        with open(p) as f:
            summ = json.load(f)
    return hist, cfg, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--last", type=int, default=0, help="average the last N epochs (0 = report best-val epoch)")
    a = ap.parse_args()

    rows = []
    for r in a.runs:
        try:
            hist, cfg, summ = load(r)
        except FileNotFoundError:
            print(f"skip {r} (no history.json)"); continue
        ev = [h for h in hist if "val_acc" in h]
        if not ev:
            continue
        names = sorted({k[len("val_uni_acc_"):] for k in ev[-1] if k.startswith("val_uni_acc_")})
        if a.last > 0:
            sel = ev[-a.last:]
            agg = lambda k: sum(h.get(k, float("nan")) for h in sel) / len(sel)
            ep = f"last{a.last}"
        else:
            best = max(ev, key=lambda h: h["val_acc"])
            agg = lambda k: best.get(k, float("nan"))
            ep = str(best["epoch"])
        row = {"run": os.path.basename(r.rstrip("/")), "mod": cfg.get("modulation", "?") if cfg.get("only", "none") == "none" else f"only-{cfg['only']}",
               "epoch": ep, "val_acc": agg("val_acc")}
        for n in names:
            row[f"uni_{n}"] = agg(f"val_uni_acc_{n}")
            row[f"rho_{n}"] = agg(f"train_rho_{n}")
        for k, v in (summ.get("probe_best_ckpt") or {}).items():
            row[k.replace("probe_acc_", "probe_")] = v
        rows.append(row)

    if not rows:
        sys.exit("nothing to compare")
    extra = []
    for r in rows:
        for c in r:
            if c not in ("run", "mod", "epoch", "val_acc") and c not in extra:
                extra.append(c)
    cols = ["run", "mod", "epoch", "val_acc"] + sorted(extra)
    widths = {c: max(len(c), *(len(f"{r.get(c, ''):.4f}" if isinstance(r.get(c, ""), float) else str(r.get(c, ""))) for r in rows)) for c in cols}
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    print(line); print("-" * len(line))
    for r in rows:
        print(" | ".join((f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
