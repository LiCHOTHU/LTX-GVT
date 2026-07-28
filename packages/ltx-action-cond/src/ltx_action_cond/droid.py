"""DROID RLDS loading + matching to the April 2025 calibration JSONs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REL_RE = re.compile(r"/r2d2-data-full/(.+?)(?:/recordings(?:/MP4)?)?$")

# Map April 2025 calibration camera roles -> RLDS image keys.
ROLE_TO_IMG_KEY: dict[str, str] = {
    "ext1_cam_serial": "exterior_image_1_left",
    "ext2_cam_serial": "exterior_image_2_left",
}


@dataclass
class CalibrationIndex:
    cam2base: dict
    intrinsics: dict
    episode_id_to_path: dict
    camera_serials: dict
    cam2base_superset: dict | None = None  # both cams calibrated per episode (subset)
    cam2cam: dict | None = None             # rig geometry per episode (subset, ~91k)

    @classmethod
    def load(cls, calib_dir: str | Path, *, with_extras: bool = True) -> "CalibrationIndex":
        calib_dir = Path(calib_dir)
        with (calib_dir / "cam2base_extrinsics.json").open() as f:
            cam2base = json.load(f)
        with (calib_dir / "intrinsics.json").open() as f:
            intrinsics = json.load(f)
        with (calib_dir / "episode_id_to_path.json").open() as f:
            ep_id_to_path = json.load(f)
        with (calib_dir / "camera_serials.json").open() as f:
            camera_serials = json.load(f)
        superset = c2c = None
        if with_extras:
            sup_path = calib_dir / "cam2base_extrinsic_superset.json"
            if sup_path.exists():
                with sup_path.open() as f:
                    superset = json.load(f)
            c2c_path = calib_dir / "cam2cam_extrinsics.json"
            if c2c_path.exists():
                with c2c_path.open() as f:
                    c2c = json.load(f)
        return cls(cam2base, intrinsics, ep_id_to_path, camera_serials,
                   cam2base_superset=superset, cam2cam=c2c)

    @property
    def path_to_ep_id(self) -> dict[str, str]:
        return {v: k for k, v in self.episode_id_to_path.items()}


def recording_folderpath_to_relpath(rec: str) -> str:
    """Convert RLDS `recording_folderpath` to the form used by `episode_id_to_path.json`.

    The RLDS field looks like `/nfs/kun2/datasets/r2d2/r2d2-data-full/INST/.../foo/recordings/MP4`.
    The calibration JSON uses just `INST/.../foo`.
    """
    m = REL_RE.search(rec.rstrip("/"))
    return m.group(1) if m else rec.rstrip("/")


@dataclass
class CandidateEpisode:
    ep_index: int
    episode_id: str
    relpath: str
    serial: str
    quality: float
    n_steps: int


def find_iou_calibrated_episodes(
    dataset_split,
    calib: CalibrationIndex,
    *,
    min_quality: float = 0.0,
    count_steps: bool = True,
) -> list[CandidateEpisode]:
    """Scan an RLDS dataset split and return episodes that have an IoU-calibrated
    exterior view in the April 2025 release.
    """
    path_to_id = calib.path_to_ep_id
    out: list[CandidateEpisode] = []
    for i, ep in enumerate(dataset_split):
        rec = ep["episode_metadata"]["recording_folderpath"].numpy().decode()
        relpath = recording_folderpath_to_relpath(rec)
        ep_id = path_to_id.get(relpath)
        if ep_id is None or ep_id not in calib.cam2base:
            continue
        entry = calib.cam2base[ep_id]
        if entry.get("metric_type") != "IoU":
            continue
        q = float(entry.get("quality_metric", -1.0))
        if q < min_quality:
            continue
        digit_keys = [k for k in entry if k.isdigit()]
        if not digit_keys or ep_id not in calib.intrinsics:
            continue
        serial = next((s for s in digit_keys if s in calib.intrinsics[ep_id]), None)
        if serial is None:
            continue
        # Iterating ep["steps"] forces TFDS to materialize all 3-camera frames
        # (~30 MB per episode) for every one of the 95K DROID episodes, which
        # OOM-kills a 48 GB worker. Skip when caller doesn't need n_steps
        # (extract / augment paths use the universe.json count instead).
        n_steps = sum(1 for _ in ep["steps"]) if count_steps else 0
        out.append(CandidateEpisode(
            ep_index=i, episode_id=ep_id, relpath=relpath,
            serial=serial, quality=q, n_steps=n_steps,
        ))
    return out


def serial_to_rls_image_key(camera_serials: dict, ep_id: str, serial: str) -> str:
    """Look up which RLDS image stream corresponds to a given camera serial."""
    role = {v: k for k, v in camera_serials[ep_id].items()}[serial]
    if role not in ROLE_TO_IMG_KEY:
        raise ValueError(f"calibrated camera {serial} has non-exterior role {role!r}")
    return ROLE_TO_IMG_KEY[role]


_EP_DATE_RE = re.compile(r"\+(\d{4}-\d{2}-\d{2}-\d{2}h-\d{2}m-\d{2}s)$")


def parse_episode_datetime(ep_id: str):
    """Return a datetime for an episode_id like 'LAB+HASH+YYYY-MM-DD-HHh-MMm-SSs'.

    Returns None if the suffix can't be parsed.
    """
    from datetime import datetime
    m = _EP_DATE_RE.search(ep_id)
    if m is None:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d-%Hh-%Mm-%Ss")


def estimate_rig_geometry(
    calib: CalibrationIndex,
    target_ep_id: str,
    src_serial: str,
    dst_serial: str,
    *,
    max_days_apart: float | None = None,
) -> dict | None:
    """Estimate T_{src_to_dst}: the relative pose of camera `src` in camera `dst`'s frame.

    The April 2025 DROID calibration release only ships ONE camera's extrinsic per
    episode entry. To recover the second exterior camera's pose in a target episode
    where it isn't directly calibrated, we estimate the rig geometry between the
    two cameras using sibling episodes from the same lab + same camera pair.

    Idea:
      Sibling A has `src` calibrated  -> T_src_to_base_A
      Sibling B has `dst` calibrated  -> T_dst_to_base_B
      Assuming A and B share the same base position (i.e. the robot setup is
      static within the sibling group):
          T_src_to_dst = inv(T_dst_to_base_B) @ T_src_to_base_A

    We collect T_src_to_dst from every (A, B) pair we can form, then return the
    median translation + the highest-quality pair's rotation. The spread across
    pairs is reported as a confidence indicator.

    To apply in the target episode (which has only `dst` calibrated):
        T_src_to_base_target = T_dst_to_base_target @ T_src_to_dst

    If `max_days_apart` is given, only sibling episodes whose recorded date
    is within that many days of the target episode are considered. This
    matters because in DROID the camera rig actually moves between recording
    sessions — restricting to same-day or same-week siblings gives a much
    tighter rig estimate than averaging over all-time.

    Returns None if no valid sibling pairs exist. Otherwise:
        {
            "T_src_to_dst": 4x4 ndarray,
            "n_pairs": int,
            "translation_spread_mm": float,  # max-min per axis, in mm
            "best_pair": (src_ep_id, dst_ep_id, combined_iou),
            "all_pairs": list of (src_ep_id, dst_ep_id, qa, qb),
            "time_window_days": float or None,
        }
    """
    from datetime import timedelta
    from .calibration import extrinsic_6dof_to_matrix

    target_sers = calib.camera_serials[target_ep_id]
    target_pair = frozenset({target_sers.get("ext1_cam_serial"), target_sers.get("ext2_cam_serial")})
    lab = target_ep_id.split("+")[0]
    target_date = parse_episode_datetime(target_ep_id)
    window = timedelta(days=max_days_apart) if max_days_apart is not None else None

    src_eps, dst_eps = [], []
    for eid, entry in calib.cam2base.items():
        if entry.get("metric_type") != "IoU":
            continue
        if eid.split("+")[0] != lab:
            continue
        sib_sers = calib.camera_serials.get(eid, {})
        sib_pair = frozenset({sib_sers.get("ext1_cam_serial"), sib_sers.get("ext2_cam_serial")})
        if sib_pair != target_pair:
            continue
        if window is not None and target_date is not None:
            sib_date = parse_episode_datetime(eid)
            if sib_date is None or abs(sib_date - target_date) > window:
                continue
        iou = float(entry.get("quality_metric", -1.0))
        if src_serial in entry:
            src_eps.append((eid, entry[src_serial], iou))
        if dst_serial in entry:
            dst_eps.append((eid, entry[dst_serial], iou))

    if not src_eps or not dst_eps:
        return None

    pairs = []
    Ts = []
    for ea, va, qa in src_eps:
        for eb, vb, qb in dst_eps:
            T_src_b = extrinsic_6dof_to_matrix(va)
            T_dst_b = extrinsic_6dof_to_matrix(vb)
            T_src_to_dst = np.linalg.inv(T_dst_b) @ T_src_b
            pairs.append((ea, eb, qa, qb))
            Ts.append(T_src_to_dst)

    Ts = np.stack(Ts, axis=0)
    quals = np.array([qa * qb for (_, _, qa, qb) in pairs])
    t_med = np.median(Ts[:, :3, 3], axis=0)
    best = int(np.argmax(quals))
    R_best = Ts[best, :3, :3]

    rig = np.eye(4)
    rig[:3, :3] = R_best
    rig[:3, 3] = t_med

    spread_mm = float(1000.0 * (Ts[:, :3, 3].max(axis=0) - Ts[:, :3, 3].min(axis=0)).max())
    return {
        "T_src_to_dst": rig,
        "n_pairs": len(pairs),
        "translation_spread_mm": spread_mm,
        "best_pair": (pairs[best][0], pairs[best][1], float(quals[best])),
        "all_pairs": pairs,
        "time_window_days": float(max_days_apart) if max_days_apart is not None else None,
    }


def _match_cam2cam_serial(entry: dict, focal: float, pp: list[float], candidate_serials: list[str]) -> str | None:
    """Identify which RLDS camera serial corresponds to a cam2cam {left,right}_cam entry,
    by matching focal length + principal-point to the intrinsics.json values."""
    for s in candidate_serials:
        cm = entry.get(s, {}).get("cameraMatrix")
        if cm is None:
            continue
        # cameraMatrix in intrinsics.json is [fx, cx, fy, cy]
        fx, cx = cm[0], cm[1]
        if abs(fx - focal) < 1.0 and abs(cx - pp[0]) < 1.0:
            return s
    return None


def rig_geometry_from_cam2cam(
    calib: CalibrationIndex,
    cam2cam_entry: dict,
    intrinsics_entry: dict,
    src_serial: str,
    dst_serial: str,
) -> np.ndarray | None:
    """Compute T_src_to_dst from a single cam2cam_extrinsics entry.

    The cam2cam JSON labels its two cams as 'left_cam' / 'right_cam'; we recover
    which is which by matching their reported focal + principal-point to the
    per-serial intrinsics. Returns None if the matching is ambiguous.
    """
    left = cam2cam_entry["left_cam"]
    right = cam2cam_entry["right_cam"]
    left_s = _match_cam2cam_serial(intrinsics_entry, left["focal"], left["principal_point"],
                                    [src_serial, dst_serial])
    right_s = _match_cam2cam_serial(intrinsics_entry, right["focal"], right["principal_point"],
                                     [src_serial, dst_serial])
    if left_s is None or right_s is None or {left_s, right_s} != {src_serial, dst_serial}:
        return None
    T_left_target = np.asarray(left["pose"], dtype=np.float64)
    T_right_target = np.asarray(right["pose"], dtype=np.float64)
    T_right_to_left = np.linalg.inv(T_left_target) @ T_right_target
    if left_s == src_serial:  # src=left, dst=right -> we want T_src_to_dst = T_left_to_right
        return np.linalg.inv(T_right_to_left)
    return T_right_to_left  # src=right, dst=left -> T_right_to_left


def resolve_both_cams_cam2base(
    calib: CalibrationIndex,
    target_ep_id: str,
) -> dict | None:
    """Return cam2base 4x4 matrices for BOTH exterior cameras of target_ep_id,
    using the most reliable available source. Tries in order:

      1. `cam2base_extrinsic_superset.json` if it has both cams of this episode.
      2. `cam2cam_extrinsics.json` for THIS episode + the target's own cam2base
         from `cam2base_extrinsics.json` (the one camera that IS shipped).
      3. `cam2cam_extrinsics.json` for a near-in-time sibling (same lab, same
         camera pair) + target's own cam2base. Tries 1d/7d/30d/90d/365d windows.

    Returns:
        {
            "ext1_cam2base": 4x4,
            "ext2_cam2base": 4x4,
            "ext1_source": "superset" | "self_calibrated" | "rig_from_target_cam2cam" | "rig_from_sibling_cam2cam",
            "ext2_source": same domain,
            "rig_source_ep": str | None,  # which episode supplied the rig
            "rig_time_window_days": float | None,
        }
        Or None if even sibling fallback can't produce both cameras.
    """
    from .calibration import extrinsic_6dof_to_matrix

    sers = calib.camera_serials.get(target_ep_id, {})
    ext1 = sers.get("ext1_cam_serial")
    ext2 = sers.get("ext2_cam_serial")
    if not (ext1 and ext2):
        return None

    # --- (1) superset ---
    if calib.cam2base_superset is not None:
        sup = calib.cam2base_superset.get(target_ep_id)
        if sup is not None and ext1 in sup and ext2 in sup:
            return {
                "ext1_cam2base": extrinsic_6dof_to_matrix(sup[ext1]),
                "ext2_cam2base": extrinsic_6dof_to_matrix(sup[ext2]),
                "ext1_source": "superset",
                "ext2_source": "superset",
                "rig_source_ep": target_ep_id,
                "rig_time_window_days": 0.0,
            }

    # We need the calibrated cam2base for one of the two from cam2base.json.
    own_entry = calib.cam2base.get(target_ep_id, {})
    own_serial = next((k for k in own_entry if k.isdigit() and k in (ext1, ext2)), None)
    if own_serial is None:
        return None
    own_cam2base = extrinsic_6dof_to_matrix(own_entry[own_serial])
    other_serial = ext2 if own_serial == ext1 else ext1
    own_role = "ext1" if own_serial == ext1 else "ext2"
    other_role = "ext2" if own_role == "ext1" else "ext1"

    intr_target = calib.intrinsics.get(target_ep_id, {})

    # --- (2) target's own cam2cam ---
    if calib.cam2cam is not None and target_ep_id in calib.cam2cam:
        T_other_to_own = rig_geometry_from_cam2cam(
            calib, calib.cam2cam[target_ep_id], intr_target,
            src_serial=other_serial, dst_serial=own_serial,
        )
        if T_other_to_own is not None:
            other_cam2base = own_cam2base @ T_other_to_own
            return {
                f"{own_role}_cam2base": own_cam2base,
                f"{other_role}_cam2base": other_cam2base,
                f"{own_role}_source": "self_calibrated",
                f"{other_role}_source": "rig_from_target_cam2cam",
                "rig_source_ep": target_ep_id,
                "rig_time_window_days": 0.0,
            }

    # --- (3) sibling cam2cam, progressively widening time window ---
    if calib.cam2cam is not None:
        target_date = parse_episode_datetime(target_ep_id)
        target_pair = frozenset({ext1, ext2})
        lab = target_ep_id.split("+")[0]
        from datetime import timedelta
        for window_days in (1.0, 7.0, 30.0, 90.0, 365.0):
            best = None  # (abs_time_delta_days, sib_eid)
            window = timedelta(days=window_days)
            for sib_eid, sib_c2c in calib.cam2cam.items():
                if sib_eid == target_ep_id:
                    continue
                if sib_eid.split("+")[0] != lab:
                    continue
                sib_sers = calib.camera_serials.get(sib_eid, {})
                sib_pair = frozenset({sib_sers.get("ext1_cam_serial"), sib_sers.get("ext2_cam_serial")})
                if sib_pair != target_pair:
                    continue
                if target_date is not None:
                    sib_date = parse_episode_datetime(sib_eid)
                    if sib_date is None or abs(sib_date - target_date) > window:
                        continue
                # Use the sibling's intrinsics (they should match target's for the same pair).
                sib_intr = calib.intrinsics.get(sib_eid, intr_target)
                rig = rig_geometry_from_cam2cam(
                    calib, sib_c2c, sib_intr,
                    src_serial=other_serial, dst_serial=own_serial,
                )
                if rig is None:
                    continue
                delta = 0.0 if target_date is None else abs((parse_episode_datetime(sib_eid) - target_date).total_seconds() / 86400.0)
                if best is None or delta < best[0]:
                    best = (delta, sib_eid, rig)
            if best is not None:
                _, sib_eid, T_other_to_own = best
                other_cam2base = own_cam2base @ T_other_to_own
                return {
                    f"{own_role}_cam2base": own_cam2base,
                    f"{other_role}_cam2base": other_cam2base,
                    f"{own_role}_source": "self_calibrated",
                    f"{other_role}_source": "rig_from_sibling_cam2cam",
                    "rig_source_ep": sib_eid,
                    "rig_time_window_days": window_days,
                }

    return None


def estimate_rig_geometry_progressive(
    calib: CalibrationIndex,
    target_ep_id: str,
    src_serial: str,
    dst_serial: str,
    *,
    windows_days: tuple[float, ...] = (1.0, 7.0, 30.0, 90.0, 365.0),
) -> dict | None:
    """Try progressively wider time windows until at least one sibling pair is found.

    Returns the rig estimate from the narrowest window that yielded pairs, so the
    rig is anchored to the temporally-closest siblings (where the rig is most
    likely to be in the same physical configuration as the target).
    """
    for w in windows_days:
        rig = estimate_rig_geometry(
            calib, target_ep_id, src_serial, dst_serial, max_days_apart=w,
        )
        if rig is not None:
            return rig
    return estimate_rig_geometry(calib, target_ep_id, src_serial, dst_serial)


def resolve_gripper_signal(
    cmd_gripper: np.ndarray,
    obs_gripper: np.ndarray,
    *,
    min_useful_range: float = 0.10,
) -> tuple[np.ndarray, str]:
    """Pick the better of cmd vs obs gripper-position signal for visualization.

    Convention assumed: 0 = open, 1 = closed (DROID `gripper_position`).

    Strategy:
      - If `obs` has range >= `min_useful_range`, use it (it tracks the actual
        physical width, so it matches what the GT video shows). Rescale to fill
        [0, 1] using its observed min/max -- some episodes have obs that only
        reaches 0.7 or 0.9 because that's what the gripper can mechanically
        achieve when holding an object.
      - Else if `cmd` has range >= `min_useful_range`, use it, rescaled to fill
        [0, 1] using its observed min/max. This handles episodes (like WEIRD)
        where the command never reaches 1.0 because the object is wide.
      - Else fall back to cmd unscaled.

    Returns (rescaled signal in [0,1], source label).
    """
    obs = np.asarray(obs_gripper, dtype=np.float64).reshape(-1)
    cmd = np.asarray(cmd_gripper, dtype=np.float64).reshape(-1)
    def rescale(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    obs_range = obs.max() - obs.min()
    cmd_range = cmd.max() - cmd.min()
    if obs_range >= min_useful_range:
        return rescale(obs), f"obs_rescaled (raw range [{obs.min():.2f},{obs.max():.2f}])"
    if cmd_range >= min_useful_range:
        return rescale(cmd), f"cmd_rescaled (raw range [{cmd.min():.2f},{cmd.max():.2f}])"
    return cmd, f"cmd_raw (range [{cmd.min():.2f},{cmd.max():.2f}] -- no useful signal)"


def extract_episode_arrays(ep, img_key: str) -> dict[str, np.ndarray]:
    """Pull joint trajectories + frames out of an RLDS episode.

    Returns dict with keys:
      joint_position      (T, 7)  observed joint angles
      cmd_joint_position  (T, 7)  commanded joint targets (action_dict.joint_position)
      action              (T, 7)  top-level RLDS action (cartesian EE + gripper)
      frames              (T, H, W, 3) uint8
    """
    obs_j, cmd_j, act, obs_g, cmd_g, frames = [], [], [], [], [], []
    for step in ep["steps"]:
        obs_j.append(step["observation"]["joint_position"].numpy())
        cmd_j.append(step["action_dict"]["joint_position"].numpy())
        act.append(step["action"].numpy())
        obs_g.append(step["observation"]["gripper_position"].numpy())
        cmd_g.append(step["action_dict"]["gripper_position"].numpy())
        frames.append(step["observation"][img_key].numpy())
    return {
        "joint_position": np.stack(obs_j, axis=0),
        "cmd_joint_position": np.stack(cmd_j, axis=0),
        "action": np.stack(act, axis=0),
        "gripper_position": np.stack(obs_g, axis=0).reshape(-1),
        "cmd_gripper_position": np.stack(cmd_g, axis=0).reshape(-1),
        "frames": np.stack(frames, axis=0),
    }
