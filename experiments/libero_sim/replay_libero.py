#!/usr/bin/env python
"""Replay a LIBERO demo from its recorded MuJoCo states and render 3 camera views.

Loads a LIBERO HDF5 demo file, reconstructs the exact recorded scene from the
per-demo `model_file` XML, steps through the stored sim `states`, and renders
three cameras — two fixed (agentview, frontview) + one wrist (robot0_eye_in_hand)
— writing one tiled side-by-side mp4 per demo (and optionally per-view mp4s).

This does NOT re-simulate the policy; it sets sim state per frame, so the render
exactly reproduces the demonstration. Requires a GPU node (MUJOCO_GL=egl).
"""
import argparse
import json
import os
from pathlib import Path

# Must be set BEFORE robosuite/mujoco import — selects the headless GL backend.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# robosuite hardcodes `logging.FileHandler("/tmp/robosuite.log")` at import time
# (utils/log_utils.py). PACE's /tmp rejects it (FileNotFoundError). Redirect that
# one path to TMPDIR by wrapping FileHandler BEFORE robosuite is imported.
import logging as _logging

_ORIG_FH = _logging.FileHandler


class _RedirectFileHandler(_ORIG_FH):
    def __init__(self, filename, *a, **k):
        if str(filename) == "/tmp/robosuite.log":
            filename = os.path.join(os.environ.get("TMPDIR", os.path.expanduser("~")), "robosuite.log")
        super().__init__(filename, *a, **k)


_logging.FileHandler = _RedirectFileHandler

import h5py
import imageio
import numpy as np

# Default 3-camera set: 2 fixed + 1 wrist.
DEFAULT_CAMS = ["agentview", "frontview", "robot0_eye_in_hand"]


def resolve_bddl(hdf5_attrs) -> str:
    """Map the HDF5's recorded bddl path to the local LIBERO package copy."""
    from libero.libero import get_libero_path

    bddl_root = get_libero_path("bddl_files")
    recorded = hdf5_attrs["bddl_file_name"]  # e.g. libero/.../libero_90/<name>.bddl
    suite = Path(recorded).parent.name       # libero_90
    name = Path(recorded).name
    path = os.path.join(bddl_root, suite, name)
    assert os.path.exists(path), f"bddl not found: {path}"
    return path


def localize_model_xml(xml_str: str) -> str:
    """Rewrite the dataset author's absolute asset paths to the local installs.

    Recorded LIBERO demos embed the collector's machine paths in `model_file`,
    e.g. `/home/yifengz/.../robosuite/models/assets/...` and
    `/home/yifengz/.../chiliocosm/assets/...`. MuJoCo can't open those here, so
    we re-root them onto the local robosuite + LIBERO installs. The regex anchors
    on the `/robosuite/models/assets` and `/chiliocosm/assets` markers, so it's
    robust to whatever absolute prefix a given demo was recorded with.
    """
    import re

    import robosuite
    from libero import libero as _libero_pkg

    rs_assets = os.path.join(os.path.dirname(robosuite.__file__), "models", "assets")
    libero_assets = os.path.join(os.path.dirname(_libero_pkg.__file__), "assets")
    xml_str = re.sub(r'[^"\s]*/robosuite/models/assets', rs_assets, xml_str)
    xml_str = re.sub(r'[^"\s]*/chiliocosm/assets', libero_assets, xml_str)
    return xml_str


def replay_demo(env, demo_grp, cams):
    """Set the recorded scene + step through states, returning {cam: [frames]}."""
    env.reset()
    env.reset_from_xml_string(localize_model_xml(demo_grp.attrs["model_file"]))
    states = demo_grp["states"][:]
    frames = {c: [] for c in cams}
    for t in range(states.shape[0]):
        obs = env.regenerate_obs_from_state(states[t])
        for c in cams:
            # robosuite renders bottom-up (OpenGL); flip to image-natural top-down.
            frames[c].append(obs[f"{c}_image"][::-1])
    return frames, states.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="LIBERO demo .hdf5 file")
    ap.add_argument("--demos", default="0", help="comma-sep demo indices, or 'all'")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cams", default=",".join(DEFAULT_CAMS))
    ap.add_argument("--res", type=int, default=256, help="per-view square resolution")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all")
    ap.add_argument("--save_per_view", action="store_true")
    args = ap.parse_args()

    cams = args.cams.split(",")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero.envs import OffScreenRenderEnv

    with h5py.File(args.hdf5, "r") as h:
        data = h["data"]
        lang = json.loads(data.attrs["problem_info"])["language_instruction"]
        bddl = resolve_bddl(data.attrs)
        all_demos = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        if args.demos == "all":
            sel = all_demos
        else:
            idxs = [int(x) for x in args.demos.split(",")]
            sel = [f"demo_{i}" for i in idxs]

        print(f"[task] {lang}")
        print(f"[bddl] {bddl}")
        print(f"[cams] {cams} @ {args.res}px | demos: {sel}")

        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_names=cams,
            camera_heights=args.res,
            camera_widths=args.res,
        )
        try:
            stem = Path(args.hdf5).stem
            for dk in sel:
                if dk not in data:
                    print(f"[skip] {dk} absent")
                    continue
                frames, n = replay_demo(env, data[dk], cams)
                if args.max_frames:
                    frames = {c: v[: args.max_frames] for c, v in frames.items()}
                    n = min(n, args.max_frames)
                # Tiled: hstack the 3 views per frame -> (H, 3W, 3).
                tiled = [np.hstack([frames[c][t] for c in cams]) for t in range(n)]
                tpath = out_dir / f"{stem}_{dk}_tiled.mp4"
                imageio.mimwrite(tpath, tiled, fps=args.fps, macro_block_size=8)
                print(f"[ok] {dk}: {n} frames -> {tpath}  (tile {tiled[0].shape})")
                if args.save_per_view:
                    for c in cams:
                        vpath = out_dir / f"{stem}_{dk}_{c}.mp4"
                        imageio.mimwrite(vpath, frames[c], fps=args.fps, macro_block_size=8)
                        print(f"     view {c} -> {vpath}")
        finally:
            env.close()
    print("[done]")


if __name__ == "__main__":
    main()
