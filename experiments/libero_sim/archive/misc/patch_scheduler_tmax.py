#!/usr/bin/env python
"""Patch a saved CosineAnnealingLR T_max in a trainer training_state_*.pt for resume-and-extend.

When a cosine run finishes at step S (T_max=S, last_epoch=S, lr=eta_min) and you want to
RESUME and train to a larger total N, the trainer rebuilds the scheduler with T_max=N but
then load_state_dict() OVERWRITES it with the saved T_max=S. With last_epoch>=T_max the
cosine recurrence diverges (LR explodes). Setting the saved T_max to the NEW total N makes
the cosine span [S, N] correctly (LR decays from ~2e-4 toward eta_min).

Backs up the original to <file>.bak before writing. Idempotent: re-running with the same
--new-t-max is a no-op on an already-patched file (besides refreshing the .bak).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state_files", nargs="+", type=Path, help="training_state_step_*.pt files to patch")
    ap.add_argument("--new-t-max", type=int, required=True, help="new cosine T_max (= new total steps)")
    args = ap.parse_args()

    for p in args.state_files:
        raw = torch.load(p, map_location="cpu", weights_only=False)
        sd = raw.get("lr_scheduler_state_dict")
        if sd is None:
            print(f"[skip] {p}: no lr_scheduler_state_dict")
            continue
        old = sd.get("T_max")
        if old == args.new_t_max:
            print(f"[noop] {p}: T_max already {old}")
            continue
        bak = p.with_suffix(p.suffix + ".bak")
        shutil.copy2(p, bak)
        sd["T_max"] = args.new_t_max
        raw["lr_scheduler_state_dict"] = sd
        torch.save(raw, p)
        print(f"[ok]   {p}: T_max {old} -> {args.new_t_max}  (backup: {bak.name})  "
              f"last_epoch={sd.get('last_epoch')} global_step={raw.get('global_step')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
