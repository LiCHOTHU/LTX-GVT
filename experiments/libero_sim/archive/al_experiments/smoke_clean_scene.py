#!/usr/bin/env python
"""Smoke test: strip LIVING_ROOM_SCENE1 to ONLY tomato_sauce (+basket) and prove
the recorded ground-truth pick-place trajectory still executes in the clean scene.

Why this is non-trivial: both replay_libero.py and gen_libero_play.py rebuild the
sim from each demo's *recorded* model_file XML (all 4 cans + basket), so editing
the bddl alone does nothing. Here we instead build the env from the minimal
`_ONLY.bddl`, then port ONLY the shared joints (robot + tomato_sauce + basket)
from the demo's initial state — the 3 distractor cans simply don't exist in the
clean model, so their joints are skipped. Then we step the demo's recorded 7-D
OSC_POSE actions (which reference the arm, not the distractors) and check success.

Run on a GPU node (MUJOCO_GL=egl). Prints object inventory + task success and
saves first/last agentview frames + an mp4.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# robosuite hardcodes logging.FileHandler("/tmp/robosuite.log"); redirect to TMPDIR.
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


def localize_model_xml(xml_str: str) -> str:
    import re
    import robosuite
    from libero import libero as _libero_pkg

    rs_assets = os.path.join(os.path.dirname(robosuite.__file__), "models", "assets")
    libero_assets = os.path.join(os.path.dirname(_libero_pkg.__file__), "assets")
    xml_str = re.sub(r'[^"\s]*/robosuite/models/assets', rs_assets, xml_str)
    xml_str = re.sub(r'[^"\s]*/chiliocosm/assets', libero_assets, xml_str)
    return xml_str


def body_names(env):
    return list(env.sim.model.body_names)


def check_success(env) -> bool:
    for attr in ("_check_success", "check_success"):
        fn = getattr(env, attr, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/"
                    "LIBERO-datasets/libero_90/"
                    "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo.hdf5")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--clean-bddl", default=None,
                    help="minimal bddl; default = the _ONLY.bddl in the libero_90 suite")
    ap.add_argument("--out-dir", default="/storage/home/hcoda1/8/lwang831/workspace/LTX-GVT/"
                    "experiments/libero_sim/outputs/smoke_clean_scene")
    ap.add_argument("--cam", default="agentview")
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_root = get_libero_path("bddl_files")
    clean_bddl = args.clean_bddl or os.path.join(
        bddl_root, "libero_90",
        "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_ONLY.bddl")
    assert os.path.exists(clean_bddl), f"clean bddl missing: {clean_bddl}"
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. read the recorded demo (actions + initial full-scene state) ---
    with h5py.File(args.hdf5, "r") as f:
        demo = f["data"][args.demo]
        actions = demo["actions"][:]
        states = demo["states"][:]
        model_xml = demo.attrs["model_file"]
    print(f"[demo] {args.demo}: {actions.shape[0]} actions, state_dim={states.shape[1]}")

    cam_kw = dict(camera_names=[args.cam], camera_heights=args.res, camera_widths=args.res)

    # --- 2. SOURCE env on the recorded full scene -> read demo-initial joint poses ---
    src = OffScreenRenderEnv(bddl_file_name=clean_bddl, **cam_kw)  # bddl irrelevant; XML overrides
    src.reset()
    src.reset_from_xml_string(localize_model_xml(model_xml))
    src.sim.set_state_from_flattened(states[0])
    src.sim.forward()
    src_joint_qpos = {jn: np.array(src.sim.data.get_joint_qpos(jn)) for jn in src.sim.model.joint_names}
    print(f"[source] bodies incl. distractors: "
          f"{[b for b in body_names(src) if any(k in b for k in ('soup','cheese','ketchup','tomato','basket'))]}")
    src.close()

    # --- 3. CLEAN env from the minimal bddl ---
    env = OffScreenRenderEnv(bddl_file_name=clean_bddl, **cam_kw)
    env.reset()
    clean_bodies = body_names(env)
    obj_bodies = [b for b in clean_bodies if any(k in b for k in ("soup", "cheese", "ketchup", "tomato", "basket"))]
    print(f"[clean] object bodies present: {obj_bodies}")
    for bad in ("alphabet_soup", "cream_cheese", "ketchup"):
        assert not any(bad in b for b in clean_bodies), f"distractor {bad} still in clean scene!"
    assert any("tomato_sauce" in b for b in clean_bodies), "tomato_sauce missing from clean scene!"

    # --- 4. port shared joints (robot + tomato_sauce + basket) from the demo start ---
    common = [jn for jn in env.sim.model.joint_names if jn in src_joint_qpos]
    for jn in common:
        try:
            env.sim.data.set_joint_qpos(jn, src_joint_qpos[jn])
        except Exception as e:
            print(f"  (skip joint {jn}: {e})")
    env.sim.forward()
    print(f"[port] copied {len(common)} shared joints; "
          f"skipped {len(src_joint_qpos) - len(common)} distractor/absent joints")

    # --- 5. execute the recorded GT actions in the clean scene ---
    frames = []
    for a in actions:
        obs, *_ = env.step(a)
        frames.append(obs[f"{args.cam}_image"][::-1])
    success = check_success(env)
    ts_body = [b for b in clean_bodies if "tomato_sauce" in b][0]
    ts_pos = env.sim.data.get_body_xpos(ts_body).copy()
    print(f"[result] GT executed {len(frames)} steps | task success = {success} | "
          f"tomato_sauce final xyz = {np.round(ts_pos, 3).tolist()}")

    # --- 6. save evidence ---
    imageio.imwrite(os.path.join(args.out_dir, "first.png"), frames[0])
    imageio.imwrite(os.path.join(args.out_dir, "last.png"), frames[-1])
    imageio.mimwrite(os.path.join(args.out_dir, "gt_clean_scene.mp4"), frames, fps=20)
    print(f"[saved] {args.out_dir}/first.png last.png gt_clean_scene.mp4")
    print("SMOKE_OK" if success else "SMOKE_RAN_BUT_TASK_NOT_SUCCESS")


if __name__ == "__main__":
    main()
