#!/usr/bin/env python
"""Generate + render the FULL interaction family on the clean tomato_sauce-only scene.

Thesis (user): "understanding the physics of the object" = covering the space of
possible interactions with it — the UNION of
  - gt      : the real dataset trajectory (on-manifold pick-place),
  - random  : clamped uniform-random exploration (broad, off-manifold),
  - reach   : approach + grasp (kinematic once grasped),
  - poke    : gripper-open lateral push/topple (object physics NOT set by gripper).

All run in the minimal `_ONLY.bddl` scene (table + tomato_sauce + basket, the 3
distractor cans removed). Because the clean model has a different state dimension
than the recorded full-scene demo, we can't set_state_from_flattened the demo
vector; instead we port ONLY the shared joints (robot + tomato_sauce + basket)
from the demo's start, so every behavior begins from the real demo configuration
and the recorded GT actions remain valid.

Run on a GPU node (MUJOCO_GL=egl). Writes per-(behavior,episode) mp4s + a tiled
side-by-side comparison mp4.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_libero_play import (  # noqa: E402
    ReachPlayPolicy, RandomPolicy, PokePlayPolicy, PerturbPolicy,
    ee_pos, object_positions, shell_from_objects, localize_model_xml,
)

DEMO_DEFAULT = ("/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/"
                "libero_90/LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo.hdf5")


def check_success(env) -> bool:
    for attr in ("_check_success", "check_success"):
        fn = getattr(env, attr, None) or getattr(getattr(env, "env", None), attr, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
    return False


def port_joints(dst_env, src_qpos: dict):
    """Copy shared joint qpos (robot + tomato_sauce + basket) into the clean env;
    distractor joints simply aren't present in dst and are skipped."""
    sim = dst_env.env.sim
    n = 0
    for jn in sim.model.joint_names:
        if jn in src_qpos:
            try:
                sim.data.set_joint_qpos(jn, src_qpos[jn])
                n += 1
            except Exception:
                pass
    sim.forward()
    return n


def make_policy(name, rng, env):
    sim = env.env.sim
    shell = shell_from_objects(object_positions(env), ee_pos(sim))
    if name == "reach":
        return ReachPlayPolicy(rng, shell)
    if name == "poke":
        return PokePlayPolicy(rng, shell)
    if name == "random":
        return RandomPolicy(rng, shell)
    raise ValueError(name)


def run_episode(env, behavior, src_qpos, gt_actions, steps, rng, cam):
    """Return list of agentview frames + success flag for one episode."""
    env.reset()
    port_joints(env, src_qpos)
    sim = env.env.sim
    frames = []
    if behavior == "gt":
        for a in gt_actions:
            obs, *_ = env.step(a)
            frames.append(obs[f"{cam}_image"][::-1])
    else:
        pol = make_policy(behavior, rng, env)
        for _ in range(steps):
            a = pol.step(ee_pos(sim), object_positions(env))
            obs, *_ = env.step(a)
            frames.append(obs[f"{cam}_image"][::-1])
    return frames, check_success(env)


def tile_row(frame_lists, labels, pad_color=(30, 30, 30)):
    """Horizontally tile one frame per behavior into a comparison video (pad to
    the longest clip with its last frame)."""
    T = max(len(f) for f in frame_lists)
    H, W = frame_lists[0][0].shape[:2]
    out = []
    for t in range(T):
        row = []
        for f in frame_lists:
            row.append(f[t] if t < len(f) else f[-1])
        out.append(np.concatenate(row, axis=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default=DEMO_DEFAULT)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--clean-bddl", default=None)
    ap.add_argument("--behaviors", default="gt,reach,poke,random")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--cam", default="agentview")
    ap.add_argument("--out-dir", default="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT/"
                    "experiments/libero_sim/outputs/clean_interact")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_root = get_libero_path("bddl_files")
    clean_bddl = args.clean_bddl or os.path.join(
        bddl_root, "libero_90",
        "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_ONLY.bddl")
    assert os.path.exists(clean_bddl), f"clean bddl missing: {clean_bddl}"
    os.makedirs(args.out_dir, exist_ok=True)
    behaviors = [b.strip() for b in args.behaviors.split(",") if b.strip()]
    cam_kw = dict(camera_names=[args.cam], camera_heights=args.res, camera_widths=args.res)

    # --- read demo actions + start state ---
    with h5py.File(args.hdf5, "r") as f:
        demo = f["data"][args.demo]
        gt_actions = demo["actions"][:]
        state0 = demo["states"][0]
        model_xml = demo.attrs["model_file"]
    print(f"[demo] {args.demo}: {len(gt_actions)} GT actions", flush=True)

    # --- source env on recorded scene -> read demo-start joint poses ---
    src = OffScreenRenderEnv(bddl_file_name=clean_bddl, **cam_kw)
    src.reset()
    src.reset_from_xml_string(localize_model_xml(model_xml))
    src.sim.set_state_from_flattened(state0)
    src.sim.forward()
    src_qpos = {jn: np.array(src.sim.data.get_joint_qpos(jn)) for jn in src.sim.model.joint_names}
    src.close()

    # --- clean env, verify distractors gone ---
    env = OffScreenRenderEnv(bddl_file_name=clean_bddl, **cam_kw)
    env.reset()
    bodies = list(env.sim.model.body_names)
    for bad in ("alphabet_soup", "cream_cheese", "ketchup"):
        assert not any(bad in b for b in bodies), f"distractor {bad} still present!"
    print(f"[clean] objects: {[b for b in bodies if any(k in b for k in ('tomato','basket'))]}", flush=True)

    rep = {}   # behavior -> representative (episode-0) frames for the tiled comparison
    try:
        for b in behaviors:
            for e in range(args.episodes):
                rng = np.random.default_rng(args.seed + 7919 * hash(b) % 100000 + e)
                frames, ok = run_episode(env, b, src_qpos, gt_actions, args.steps, rng, args.cam)
                path = os.path.join(args.out_dir, f"{b}_ep{e}.mp4")
                imageio.mimwrite(path, frames, fps=20)
                if e == 0:
                    rep[b] = frames
                print(f"[ok] {b} ep{e}: {len(frames)} frames, success={ok} -> {os.path.basename(path)}", flush=True)
    finally:
        env.close()

    # --- side-by-side comparison (episode 0 of each behavior) ---
    if len(rep) > 1:
        order = [b for b in behaviors if b in rep]
        tiled = tile_row([rep[b] for b in order], order)
        cpath = os.path.join(args.out_dir, "compare_" + "_".join(order) + ".mp4")
        imageio.mimwrite(cpath, tiled, fps=20)
        print(f"[compare] panels L->R: {order} -> {cpath}", flush=True)
    print("CLEAN_INTERACT_DONE", flush=True)


if __name__ == "__main__":
    main()
