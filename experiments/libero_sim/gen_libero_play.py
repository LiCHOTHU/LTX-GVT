#!/usr/bin/env python
"""Generate LIBERO *play* episodes and write them as LIBERO-format HDF5 files.

Motivation (Interactive World Simulator, Wang et al.): an action-conditioned
world model must predict the consequence of *arbitrary* actions, including the
off-task ones a deployed/suboptimal policy actually takes. Training on success
demos alone lets the model ignore the action input (the action is near-
deterministic given context). The paper fixes this with *play data* — but pure
uniform-random actions waste almost all their frames waving in free space. Their
trick (Sec. IV-A) is to **constrain the action space to the interaction
submanifold** ("limit motion of robot effector to bimanual plane motion") so
every random action is a potential contact.

This generator implements the generalization of that trick for LIBERO-90,
recipe **#3 + #4**:
  #3 demo-state initialization — reset the sim to a state sampled along a real
     demo, so play branches off the *task-relevant* manifold, not free space;
  #4 clamped, object-relative play — drive the end-effector toward randomized
     goals biased onto object bounding boxes (smooth OU path, not per-step
     jitter), with the action space clamped to an interaction shell (z-band over
     the objects, small orientation wobble) and the gripper gated on object
     proximity. Every episode is contact-rich and physically plausible.

Output is byte-compatible with the real LIBERO demo HDF5 (data.attrs
problem_info / bddl_file_name; per-demo attrs model_file; states (T, state_dim)
= sim.get_state().flatten(); actions (T, 7) OSC_POSE in [-1, 1]). So the
existing build_all_libero.py -> build_libero_context.py -> encode pipeline runs
UNCHANGED — just point --dataset at the play directory:

    python build_all_libero.py --dataset <play_dir> --out_root <play_context> ...

Two extra policies are provided for ablation: `perturb` (#1: replay demo actions
+ smoothed noise, near-manifold) and `random` (#2: demo-state init + clamped
uniform random). The recommended default is `reach` (#3+#4).

This is a CPU-light MuJoCo stepping job (no context rendering here); one GPU is
still wanted because LIBERO's OffScreenRenderEnv builds an EGL context, but the
per-step cost is physics, not the (tiny) image render.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

# Headless GL backend — must be set before robosuite/mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# robosuite hardcodes logging.FileHandler("/tmp/robosuite.log") at import; PACE's
# /tmp rejects it. Redirect to TMPDIR (same shim as replay_libero.py).
import logging as _logging

_ORIG_FH = _logging.FileHandler


class _RedirectFileHandler(_ORIG_FH):
    def __init__(self, filename, *a, **k):
        if str(filename) == "/tmp/robosuite.log":
            filename = os.path.join(
                os.environ.get("TMPDIR", os.path.expanduser("~")), "robosuite.log")
        super().__init__(filename, *a, **k)


_logging.FileHandler = _RedirectFileHandler

import h5py
import numpy as np

# Same-dir import (script dir is on sys.path[0] when run directly). The replay-buffer
# helpers define the perturb-source contract; keeping them there avoids duplication.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al_replay_buffer import bases_for_task, load_manifest, task_key_of

_HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(_HERE))  # localize_model_xml
from replay_libero import localize_model_xml  # noqa: E402

# OSC_POSE: action = [dx, dy, dz, drx, dry, drz, gripper], each in [-1, 1].
# The position channels are scaled by the controller's output_max (default 0.05 m
# at full deflection), so a normalized action ~= clip(world_delta / POS_SCALE).
POS_SCALE = 0.05
EE_SITE = "gripper0_grip_site"  # robosuite Panda end-effector site
ROBOT_KEYS = ("robot", "gripper", "panda", "mount", "world", "table",
              "wall", "floor", "bin", "arena")


# --------------------------------------------------------------------------- sim geometry helpers
def ee_pos(sim) -> np.ndarray:
    """End-effector world position (grip site), with a body fallback."""
    try:
        return sim.data.get_site_xpos(EE_SITE).copy()
    except Exception:
        return sim.data.get_body_xpos("robot0_right_hand").copy()


def object_positions(env) -> dict[str, np.ndarray]:
    """World positions of the task's manipulable objects (goal candidates).

    Prefers the env's `obj_of_interest` (the BDDL goal objects); falls back to
    every non-robot/non-fixture body so the shell still has something to aim at.
    """
    sim = env.env.sim
    out: dict[str, np.ndarray] = {}
    for n in getattr(env.env, "obj_of_interest", None) or []:
        for cand in (f"{n}_main", n, f"{n}_1_main", f"{n}_g0"):
            try:
                out[n] = sim.data.get_body_xpos(cand).copy()
                break
            except Exception:
                continue
    if not out:
        for i in range(sim.model.nbody):
            bn = sim.model.body_id2name(i)
            if bn and not any(k in bn.lower() for k in ROBOT_KEYS):
                p = sim.data.body_xpos[i].copy()
                if np.isfinite(p).all():
                    out[bn] = p
    return out


def bowl_radius(env, default: float = 0.04) -> float:
    """XY radius (m) of the task's bowl, for the scripted-circle policy.

    Picks the obj_of_interest whose name mentions 'bowl' (else the first object),
    resolves its body, and takes the max XY geom half-size over that body's geoms
    (mujoco geom_size: box -> half-extent, sphere/cylinder -> radius). Falls back
    to `default` if nothing matches.
    """
    sim = env.env.sim
    names = list(getattr(env.env, "obj_of_interest", None) or [])
    pick = next((n for n in names if "bowl" in n.lower()), names[0] if names else None)
    if pick is None:
        return default
    bid = None
    for cand in (f"{pick}_main", pick, f"{pick}_1_main", f"{pick}_g0"):
        try:
            bid = sim.model.body_name2id(cand)
            break
        except Exception:
            continue
    if bid is None:
        return default
    r = 0.0
    for gi in range(sim.model.ngeom):
        if sim.model.geom_bodyid[gi] == bid:
            sz = sim.model.geom_size[gi]
            r = max(r, float(max(sz[0], sz[1])))  # XY half-size / radius
    return r if r > 1e-4 else default


# --------------------------------------------------------------------------- play policies
class ReachPlayPolicy:
    """#3 demo-state init (handled by caller) + #4 clamped object-relative play.

    Picks a random object goal (+ small offset), P-controls the EE toward it with
    OU-smoothed noise, clamps the EE into an interaction shell derived from the
    object heights, and toggles the gripper only when near an object.
    """

    def __init__(self, rng: np.random.Generator, shell: tuple[float, float]):
        self.rng = rng
        self.z_lo, self.z_hi = shell
        self.noise = np.zeros(6)          # OU state over the 6 pose channels
        self.theta, self.sigma = 0.15, np.array([0.45, 0.45, 0.30, 0.12, 0.12, 0.12])
        self.goal: np.ndarray | None = None
        self.goal_ttl = 0
        self.grip = -1.0                  # -1 open, +1 closed
        self.grip_hold = 0

    def _pick_goal(self, objs: dict[str, np.ndarray]):
        keys = list(objs)
        tgt = objs[keys[self.rng.integers(len(keys))]] if keys else np.array([0.0, 0.0, self.z_lo])
        off = self.rng.uniform(-0.06, 0.06, 3)
        # ~40% of goals dip to the object surface (approach/contact); else hover.
        off[2] = self.rng.uniform(-0.02, 0.04) if self.rng.random() < 0.4 else self.rng.uniform(0.05, 0.15)
        self.goal = tgt + off
        self.goal[2] = float(np.clip(self.goal[2], self.z_lo, self.z_hi))
        self.goal_ttl = int(self.rng.integers(18, 40))

    def step(self, ee: np.ndarray, objs: dict[str, np.ndarray]) -> np.ndarray:
        if self.goal is None or self.goal_ttl <= 0 or np.linalg.norm(self.goal - ee) < 0.02:
            self._pick_goal(objs)
        self.goal_ttl -= 1

        a = np.zeros(6)
        a[:3] = np.clip((self.goal - ee) / POS_SCALE, -1.0, 1.0)
        # OU smoothing -> correlated, plausible motion instead of per-step jitter.
        self.noise += -self.theta * self.noise + self.sigma * self.rng.standard_normal(6)
        a = np.clip(a + self.noise, -1.0, 1.0)

        # Interaction-shell clamp (the paper's "constrain the action space" trick):
        # never command further out of the z-band the objects live in.
        if ee[2] >= self.z_hi:
            a[2] = min(a[2], 0.0)
        elif ee[2] <= self.z_lo:
            a[2] = max(a[2], 0.0)

        # Proximity-gated gripper: only grasp/release near an object.
        d = min((np.linalg.norm(ee - p) for p in objs.values()), default=1e9)
        if self.grip_hold > 0:
            self.grip_hold -= 1
        elif d < 0.05:
            if self.grip < 0 and self.rng.random() < 0.4:
                self.grip, self.grip_hold = 1.0, int(self.rng.integers(8, 20))   # close & hold
            elif self.grip > 0 and self.rng.random() < 0.2:
                self.grip, self.grip_hold = -1.0, int(self.rng.integers(4, 10))  # release
        elif self.grip > 0 and self.rng.random() < 0.05:
            self.grip = -1.0   # don't carry a closed gripper around forever
        return np.concatenate([a, [self.grip]])


class PerturbPolicy:
    """#1 near-manifold: replay a real demo's actions + OU-smoothed noise."""

    def __init__(self, rng: np.random.Generator, demo_actions: np.ndarray, sigma: float = 0.25):
        self.rng = rng
        self.acts = demo_actions
        self.sigma = sigma
        self.noise = np.zeros(demo_actions.shape[1])
        self.theta = 0.2

    def step_idx(self, t: int) -> np.ndarray:
        base = self.acts[min(t, len(self.acts) - 1)].copy()
        self.noise += -self.theta * self.noise + self.sigma * self.rng.standard_normal(self.noise.shape[0])
        a = base + self.noise
        a[-1] = base[-1]  # leave the recorded gripper command intact
        return np.clip(a, -1.0, 1.0)


class RandomPolicy:
    """#2 clamped uniform random from a demo state (OU for smoothness)."""

    def __init__(self, rng: np.random.Generator, shell: tuple[float, float]):
        self.rng = rng
        self.z_lo, self.z_hi = shell
        self.noise = np.zeros(6)
        self.theta, self.sigma = 0.1, np.array([0.6, 0.6, 0.4, 0.15, 0.15, 0.15])
        self.grip = -1.0

    def step(self, ee: np.ndarray, objs: dict[str, np.ndarray]) -> np.ndarray:
        self.noise += -self.theta * self.noise + self.sigma * self.rng.standard_normal(6)
        a = np.clip(self.noise, -1.0, 1.0)
        if ee[2] >= self.z_hi:
            a[2] = min(a[2], 0.0)
        elif ee[2] <= self.z_lo:
            a[2] = max(a[2], 0.0)
        if self.rng.random() < 0.04:
            self.grip *= -1.0
        return np.concatenate([a, [self.grip]])


class PokePlayPolicy:
    """Gripper-OPEN lateral push/poke: drive the EE horizontally *into* the target
    object so it slides / tips / rolls. Unlike ReachPlayPolicy (which grasps, after
    which the object rigidly follows the gripper = kinematics the model already gets
    from the action tokens), a poke excites object dynamics NOT determined by the
    gripper pose — the actual physics we want the world model to learn. Re-picks a
    random push azimuth periodically so successive pokes hit the object from many
    angles; the gripper is held OPEN the whole time.
    """

    def __init__(self, rng: np.random.Generator, shell: tuple[float, float],
                 target_hint: tuple[str, ...] = ("tomato", "sauce")):
        self.rng = rng
        self.z_lo, self.z_hi = shell
        self.hint = target_hint
        self.noise = np.zeros(6)
        self.theta = 0.15
        self.sigma = np.array([0.40, 0.40, 0.15, 0.10, 0.10, 0.10])
        self.dir: np.ndarray | None = None
        self.ttl = 0
        self.z_push: float | None = None

    def _target(self, objs: dict[str, np.ndarray]) -> np.ndarray:
        for k in objs:
            if any(h in k.lower() for h in self.hint):
                return objs[k]
        return next(iter(objs.values())) if objs else np.array([0.0, 0.0, self.z_lo])

    def _new_push(self, obj: np.ndarray):
        ang = self.rng.uniform(0.0, 2.0 * np.pi)
        self.dir = np.array([np.cos(ang), np.sin(ang)])
        self.ttl = int(self.rng.integers(25, 45))
        # push near the object's lower body so it topples/slides, not skims the top.
        self.z_push = float(np.clip(obj[2] - 0.01, self.z_lo, self.z_hi))

    def step(self, ee: np.ndarray, objs: dict[str, np.ndarray]) -> np.ndarray:
        obj = self._target(objs)
        if self.dir is None or self.ttl <= 0:
            self._new_push(obj)
        self.ttl -= 1
        # Aim ~8 cm beyond the object along the push direction so the EE drives
        # *through* it (contact + follow-through), rather than stopping at the surface.
        goal = np.array([obj[0] + self.dir[0] * 0.08,
                         obj[1] + self.dir[1] * 0.08,
                         self.z_push])
        a = np.zeros(6)
        a[:3] = np.clip((goal - ee) / POS_SCALE, -1.0, 1.0)
        self.noise += -self.theta * self.noise + self.sigma * self.rng.standard_normal(6)
        a = np.clip(a + self.noise, -1.0, 1.0)
        if ee[2] >= self.z_hi:
            a[2] = min(a[2], 0.0)
        elif ee[2] <= self.z_lo:
            a[2] = max(a[2], 0.0)
        return np.concatenate([a, [-1.0]])  # gripper OPEN throughout


class ScriptedCirclePolicy:
    """Deterministic hand-authored OOD motion (no noise): the end-effector goes
    straight down until it contacts the table, the gripper closes, then it traces
    a horizontal circle of radius `radius` centered at the contact point, held at
    the contact height.

    Phases (driven purely by the measured EE position, so it adapts per scene):
      descend     -> command dz=-1, gripper OPEN, until EE z stalls (contact)
      grip_close  -> hold pose, gripper CLOSED, for a few steps to finish closing
      circle      -> P-control xy around the circle, hold z at contact, grip CLOSED
    """

    def __init__(self, radius: float, period: int = 70, close_hold: int = 5,
                 contact_eps: float = 0.003, min_descend: int = 3):
        self.R = float(radius)
        self.dtheta = 2.0 * np.pi / max(8, int(period))
        self.close_hold = int(close_hold)
        self.contact_eps = float(contact_eps)
        self.min_descend = int(min_descend)
        self.phase = "descend"
        self.prev_z = None
        self.n_down = 0
        self.hold = 0
        self.center = None      # (x, y) of contact
        self.z_contact = None
        self.theta = 0.0

    def step(self, ee: np.ndarray, objs: dict[str, np.ndarray]) -> np.ndarray:
        a = np.zeros(6)
        if self.phase == "descend":
            stalled = (self.prev_z is not None
                       and abs(ee[2] - self.prev_z) < self.contact_eps)
            self.prev_z = float(ee[2])
            self.n_down += 1
            if self.n_down >= self.min_descend and stalled:
                self.center = ee[:2].copy()
                self.z_contact = float(ee[2])
                self.phase = "grip_close"
                self.hold = self.close_hold
                a[2] = -0.2                       # keep light downward pressure
                return np.concatenate([a, [1.0]])  # start closing
            a[2] = -1.0
            return np.concatenate([a, [-1.0]])     # full down, gripper open

        if self.phase == "grip_close":
            a[2] = np.clip((self.z_contact - ee[2]) / POS_SCALE, -1.0, 1.0)
            self.hold -= 1
            if self.hold <= 0:
                self.phase = "circle"
            return np.concatenate([a, [1.0]])

        # circle: spiral the radius out from 0 to R over the first lap so the
        # trace is a clean circle (no instant radial dash from the contact point).
        self.theta += self.dtheta
        r_eff = self.R * min(1.0, self.theta / (2.0 * np.pi))
        target_xy = self.center + r_eff * np.array([np.cos(self.theta),
                                                    np.sin(self.theta)])
        a[:2] = np.clip((target_xy - ee[:2]) / POS_SCALE, -1.0, 1.0)
        a[2] = np.clip((self.z_contact - ee[2]) / POS_SCALE, -1.0, 1.0)
        return np.concatenate([a, [1.0]])


# --------------------------------------------------------------------------- rollout
def shell_from_objects(objs: dict[str, np.ndarray], ee: np.ndarray) -> tuple[float, float]:
    zs = [p[2] for p in objs.values()]
    if zs:
        return min(zs) - 0.03, max(zs) + 0.18
    return ee[2] - 0.15, ee[2] + 0.15


def rollout_episode(env, demo_grp, policy_name: str, steps: int, rng: np.random.Generator):
    """Init to a sampled demo state, then step the chosen play policy.

    Returns (states (T, sdim), actions (T, 7)) with states[t] the state from
    which actions[t] was applied — the same pairing real LIBERO demos use.
    """
    sim = env.env.sim
    demo_states = demo_grp["states"][:]
    demo_acts = demo_grp["actions"][:]

    # #3: start from a state sampled along the demo (bias toward the early/mid
    # phase so there's room to interact, not the post-completion tail). The
    # scripted-circle and GT-replay policies are deterministic, so they start from
    # the demo's first state for a clean, reproducible full-demo rollout.
    if policy_name in ("circle", "gt"):
        start = 0
    else:
        start = int(rng.integers(0, max(1, int(0.6 * len(demo_states)))))
    sim.set_state_from_flattened(demo_states[start])
    sim.forward()

    objs0 = object_positions(env)
    shell = shell_from_objects(objs0, ee_pos(sim))
    if policy_name == "reach":
        pol = ReachPlayPolicy(rng, shell)
    elif policy_name == "poke":
        pol = PokePlayPolicy(rng, shell)
    elif policy_name == "random":
        pol = RandomPolicy(rng, shell)
    elif policy_name == "perturb":
        pol = PerturbPolicy(rng, demo_acts[start:])
    elif policy_name == "gt":
        # On-manifold reference: exact zero-noise replay of the real demo actions
        # (PerturbPolicy with sigma=0). Deterministic full-demo rollout from t=0.
        pol = PerturbPolicy(rng, demo_acts[start:], sigma=0.0)
    elif policy_name == "circle":
        # User chose a fixed 14cm "max clean" circle (2x bowl-diameter = ~22cm is
        # beyond the reachable/trackable workspace from the table contact point and
        # traces messily). Bowl radius is still logged for reference.
        r_bowl = bowl_radius(env)
        radius = float(os.environ.get("CIRCLE_RADIUS_M", "0.14"))
        print(f"[circle] bowl radius={r_bowl*100:.1f}cm -> circle R={radius*100:.1f}cm (fixed)", flush=True)
        pol = ScriptedCirclePolicy(radius)
    else:
        raise ValueError(policy_name)

    states, actions = [], []
    for t in range(steps):
        states.append(sim.get_state().flatten().copy())
        if policy_name in ("perturb", "gt"):
            a = pol.step_idx(t)
        else:
            a = pol.step(ee_pos(sim), object_positions(env))
        env.step(a)
        actions.append(np.asarray(a, np.float64))
    return np.stack(states, 0), np.stack(actions, 0)


# --------------------------------------------------------------------------- per-task driver
def _write_play_hdf5(out_path: Path, top_attrs: dict, policy: str,
                     episodes: list[tuple[np.ndarray, np.ndarray, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".part")
    with h5py.File(tmp, "w") as h:
        d = h.create_group("data")
        for k, v in top_attrs.items():
            d.attrs[k] = v
        d.attrs["num_demos"] = str(len(episodes))
        d.attrs["tag"] = "libero-play-v1"
        d.attrs["play_policy"] = policy
        for i, (states, actions, model_xml) in enumerate(episodes):
            g = d.create_group(f"demo_{i}")
            g.attrs["model_file"] = model_xml
            g.attrs["num_samples"] = states.shape[0]
            g.attrs["init_state"] = states[0]
            g.create_dataset("states", data=states)
            g.create_dataset("actions", data=actions)
    tmp.rename(out_path)


def gen_task(hdf5: str, out_dir: Path, jobs: list[tuple[str, int]], steps: int,
             seed: int, perturb_entries: list[dict] | None = None) -> list[str]:
    """Generate a mix of play policies for one LIBERO task.

    `jobs` is a list of (policy, n_episodes). The env is created ONCE and reused
    across every policy/episode (the expensive part is env build, not stepping),
    and each policy is written to its OWN HDF5 `<stem>_play_<policy>.hdf5` so all
    policies coexist in one flat dataset dir that build_all_libero.py enumerates,
    yielding unique ep_ids (`<stem>_play_<policy>_demo_N`).
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    stem = Path(hdf5).stem
    pending = [(pol, n, out_dir / f"{stem}_play_{pol}.hdf5") for pol, n in jobs if n > 0]
    pending = [(pol, n, p) for pol, n, p in pending if not p.exists()]
    if not pending:
        print(f"[skip] {stem}: all play policies already exist", flush=True)
        return []

    with h5py.File(hdf5, "r") as h:
        data = h["data"]
        top_attrs = {k: data.attrs[k] for k in data.attrs}
        recorded = data.attrs["bddl_file_name"]
        bddl = os.path.join(get_libero_path("bddl_files"),
                            Path(recorded).parent.name, Path(recorded).name)
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        # Cache each source demo's states/actions/model_file (env reset closes h).
        src = [(data[dk]["states"][:], data[dk]["actions"][:], data[dk].attrs["model_file"])
               for dk in demo_keys]

    # Buffer-sourced perturb bases (SAME task only — an action sequence is valid only
    # in the scene it was recorded in). Empty -> perturb falls back to this task's GT
    # demos (identical to the no-buffer behaviour), so round 1 / uncovered tasks are safe.
    perturb_bases: list[tuple[np.ndarray, np.ndarray, str]] = []
    if perturb_entries:
        for ent in perturb_entries:
            try:
                with h5py.File(ent["hdf5"], "r") as bh:
                    g = bh["data"][ent["demo"]]
                    perturb_bases.append((g["states"][:], g["actions"][:], g.attrs["model_file"]))
            except Exception as ex:
                print(f"[warn] perturb base unreadable {ent['hdf5']}:{ent['demo']}: {ex}", flush=True)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=["agentview"],
                             camera_heights=84, camera_widths=84)
    written: list[str] = []
    try:
        for pi, (policy, n, out_path) in enumerate(pending):
            episodes = []
            for e in range(n):
                # Distinct seed per (policy, episode) so the three splits never coincide.
                rng = np.random.default_rng(seed + 1_000_000 * pi + e)
                if policy == "perturb" and perturb_bases:
                    s_states, s_acts, model_xml = perturb_bases[int(rng.integers(len(perturb_bases)))]
                else:
                    s_states, s_acts, model_xml = src[e % len(src)]
                model_xml = localize_model_xml(model_xml)
                env.reset()
                env.reset_from_xml_string(model_xml)
                states, actions = rollout_episode(
                    env, _DemoShim({"states": s_states, "actions": s_acts}),
                    policy, steps, rng)
                episodes.append((states, actions, model_xml))
            _write_play_hdf5(out_path, top_attrs, policy, episodes)
            written.append(str(out_path))
            print(f"[ok] {out_path.name}: {len(episodes)} {policy} demos", flush=True)
    finally:
        env.close()
    return written


class _DemoShim:
    """Minimal h5py-group-like accessor so rollout_episode reads cached arrays."""

    def __init__(self, d):
        self._d = d

    def __getitem__(self, k):
        return self._d[k]


# --------------------------------------------------------------------------- main
def parse_mix(spec: str) -> list[tuple[str, int]]:
    """'reach:36,perturb:18,random:6' -> [('reach',36),('perturb',18),('random',6)]."""
    jobs = []
    for part in spec.split(","):
        pol, _, n = part.strip().partition(":")
        pol = pol.strip()
        if pol not in ("reach", "poke", "perturb", "random", "circle", "gt"):
            raise ValueError(f"bad policy in --mix: {pol!r}")
        jobs.append((pol, int(n)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90")
    ap.add_argument("--out_dir", default="/storage/scratch1/8/lwang831/libero90_play")
    # Play-dominant mix (user decision 2026-06-08): reach 36 / perturb 18 / random 6
    # per task -> 60/task x 90 = 5,400 eps ~= 38k chunks (~1.7x the 22,482 demos).
    ap.add_argument("--mix", default="reach:36,perturb:18,random:6",
                    help="comma list policy:count, e.g. 'reach:36,perturb:18,random:6'")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max_tasks", type=int, default=0, help="cap tasks (smoke test)")
    ap.add_argument("--perturb-source", default=None,
                    help="replay-buffer manifest.json; perturb draws same-task bases "
                         "from it (+ noise). Omit -> perturb off the task's GT demos.")
    args = ap.parse_args()

    jobs = parse_mix(args.mix)
    buffer_entries = load_manifest(Path(args.perturb_source)) if args.perturb_source else []
    if args.perturb_source:
        print(f"[driver] perturb-source: {len(buffer_entries)} buffer entries "
              f"from {args.perturb_source}", flush=True)
    out_dir = Path(args.out_dir)
    # Per-task markers (mirrors build_all_libero.py) so the gen phase is resumable
    # and the watchdog can detect completion: _gen_done on success, _gen_failed on
    # exception, _total_tasks = the full enumerated task count (NOT the shard slice).
    done_dir = out_dir / "_gen_done"
    fail_dir = out_dir / "_gen_failed"
    done_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.dataset, "*.hdf5")))
    (out_dir / "_total_tasks").write_text(str(len(all_files)))
    files = all_files
    if args.num_shards > 1:
        files = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    if args.max_tasks:
        files = files[: args.max_tasks]
    per_task = sum(n for _, n in jobs)
    print(f"[driver] shard {args.shard}/{args.num_shards}: {len(files)}/{len(all_files)} tasks "
          f"x {per_task} eps ({args.mix}) x {args.steps} steps -> {out_dir}", flush=True)

    for fi, f in enumerate(files):
        stem = Path(f).stem
        if (done_dir / stem).exists() or (fail_dir / stem).exists():
            continue
        try:
            pe = bases_for_task(buffer_entries, task_key_of(stem)) if buffer_entries else None
            gen_task(f, out_dir, jobs, args.steps,
                     seed=args.seed + 1000 * (args.shard + 1) + 100 * fi,
                     perturb_entries=pe)
            (done_dir / stem).write_text("ok")
        except Exception as ex:  # one bad task must not kill the shard
            (fail_dir / stem).write_text(f"{type(ex).__name__}: {ex}")
            print(f"[FAIL] {stem}: {type(ex).__name__}: {ex}", flush=True)
    nd = len(list(done_dir.glob("*"))); nf = len(list(fail_dir.glob("*")))
    print(f"[driver] done. gen markers: done={nd} failed={nf} total={len(all_files)}", flush=True)


if __name__ == "__main__":
    main()
