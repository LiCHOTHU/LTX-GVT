"""Franka Panda forward kinematics via Modified DH (Craig convention).

We use Modified DH so a, alpha belong to row i-1 in the table and d, theta belong to row i.
The result is the 3D origin of every link (base + 7 joints + flange) expressed in the
robot base frame. For Stage 1 (skeleton prior) the origins are all we need; for Stage 2
(mesh render) the per-link 4x4 frames are also computed.

Source: Franka official documentation (frankaemika.github.io).
"""

from __future__ import annotations

import numpy as np

# (a_{i-1}, alpha_{i-1}, d_i, theta_offset_i)
DH_PARAMS: list[tuple[float, float, float, float]] = [
    (0.0,     0.0,         0.333, 0.0),   # joint 1
    (0.0,    -np.pi / 2,   0.0,   0.0),   # joint 2
    (0.0,     np.pi / 2,   0.316, 0.0),   # joint 3
    (0.0825,  np.pi / 2,   0.0,   0.0),   # joint 4
    (-0.0825, -np.pi / 2,  0.384, 0.0),   # joint 5
    (0.0,     np.pi / 2,   0.0,   0.0),   # joint 6
    (0.088,   np.pi / 2,   0.0,   0.0),   # joint 7
    (0.0,     0.0,         0.107, 0.0),   # flange (fixed)
]


def modified_dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """One Modified-DH link transform as a 4x4 homogeneous matrix."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,        -st,      0.0,   a],
        [st * ca,   ct * ca,  -sa,   -sa * d],
        [st * sa,   ct * sa,  ca,    ca * d],
        [0.0,       0.0,      0.0,   1.0],
    ])


def fk_frames(q: np.ndarray) -> np.ndarray:
    """Return 9 link 4x4 frames in base coordinates: [base, J1..J7, flange]. Shape (9, 4, 4)."""
    if q.shape != (7,):
        raise ValueError(f"q must be (7,), got {q.shape}")
    frames = [np.eye(4)]
    T = np.eye(4)
    for i, (a, alpha, d, theta_off) in enumerate(DH_PARAMS):
        theta = q[i] + theta_off if i < 7 else theta_off
        T = T @ modified_dh_transform(a, alpha, d, theta)
        frames.append(T.copy())
    return np.stack(frames, axis=0)


def fk(q: np.ndarray) -> np.ndarray:
    """Return 9 link origins (base + 7 joints + flange) in base coordinates. Shape (9, 3)."""
    return fk_frames(q)[:, :3, 3]


# URDF joint origins from franka_description/robots/fer/kinematics.yaml.
# Each entry: (parent_link_index, xyz, rpy, is_revolute).  Joint i is at index i.
URDF_JOINTS: list[tuple[int, tuple[float, float, float], tuple[float, float, float], bool]] = [
    # joint1: link0 -> link1
    (0, (0.0, 0.0, 0.333),  (0.0, 0.0, 0.0),  True),
    # joint2: link1 -> link2
    (1, (0.0, 0.0, 0.0),    (-np.pi / 2, 0.0, 0.0), True),
    # joint3: link2 -> link3
    (2, (0.0, -0.316, 0.0), (np.pi / 2, 0.0, 0.0),  True),
    # joint4: link3 -> link4
    (3, (0.0825, 0.0, 0.0), (np.pi / 2, 0.0, 0.0),  True),
    # joint5: link4 -> link5
    (4, (-0.0825, 0.384, 0.0), (-np.pi / 2, 0.0, 0.0), True),
    # joint6: link5 -> link6
    (5, (0.0, 0.0, 0.0),    (np.pi / 2, 0.0, 0.0),  True),
    # joint7: link6 -> link7
    (6, (0.088, 0.0, 0.0),  (np.pi / 2, 0.0, 0.0),  True),
    # joint8 (fixed): link7 -> flange
    (7, (0.0, 0.0, 0.107),  (0.0, 0.0, 0.0),        False),
]

# Hand attachment: flange -> hand, fixed at yaw=-pi/4 (Franka Panda gripper convention).
HAND_ORIGIN: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (0.0, 0.0, 0.0), (0.0, 0.0, -np.pi / 4),
)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _joint_origin_matrix(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rpy_to_matrix(*rpy)
    m[:3, 3] = xyz
    return m


def fk_urdf(q: np.ndarray) -> dict[str, np.ndarray]:
    """URDF-convention forward kinematics matching the franka_description meshes.

    The result is a dict of 4x4 transforms in the robot-base frame, keyed by
    the visual mesh file stem:
        link0, link1, ..., link7, flange (joint8 tip), hand

    Each link mesh `linkN.dae` is positioned at the returned `linkN` transform.
    `hand.dae` is positioned at the returned `hand` transform.
    """
    if q.shape != (7,):
        raise ValueError(f"q must be (7,), got {q.shape}")
    link_frames: list[np.ndarray] = [np.eye(4)]  # link0 = base
    T = np.eye(4)
    for i, (_parent_idx, xyz, rpy, is_rev) in enumerate(URDF_JOINTS[:-1]):  # joints 1..7 (revolute)
        T = T @ _joint_origin_matrix(xyz, rpy)
        if is_rev:
            cz, sz = np.cos(q[i]), np.sin(q[i])
            Rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            T = T @ Rz
        link_frames.append(T.copy())
    # joint8 (fixed): link7 -> flange
    _, xyz8, rpy8, _ = URDF_JOINTS[7]
    T_flange = link_frames[7] @ _joint_origin_matrix(xyz8, rpy8)
    T_hand = T_flange @ _joint_origin_matrix(*HAND_ORIGIN)
    return {
        **{f"link{i}": link_frames[i] for i in range(8)},
        "flange": T_flange,
        "hand": T_hand,
    }
