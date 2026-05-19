"""Kinematics for the Robotiq 2F-85 (a-price/robotiq_arg85_description URDF).

Tree (anchored at `robotiq_85_base_link`):

    robotiq_85_base_link
    +-- left_outer_knuckle    (finger_joint, revolute -Y, drive)
    |   `-- left_outer_finger (fixed)
    +-- left_inner_knuckle    (mimic x +1,  revolute -Y)
    |   `-- left_inner_finger (mimic x -1,  revolute -Y)
    +-- right_outer_knuckle   (mimic x -1,  revolute +Y, yaw=pi)
    +-- right_inner_knuckle   (mimic x -1,  revolute +Y, yaw=pi)
        `-- right_inner_finger(mimic x +1,  revolute +Y)

Right-side outer-finger geometry is attached via right_outer_knuckle as in URDF.

Drive parameter `theta` is the `finger_joint` angle in radians, range [0, 0.725].
Map DROID's normalized `gripper_position` (0 open -> 1 closed) with
`theta = gripper_position * 0.725`.
"""

from __future__ import annotations

import numpy as np

ROBOTIQ_LINK_NAMES: tuple[str, ...] = (
    "base",
    "left_outer_knuckle",
    "left_outer_finger",
    "left_inner_knuckle",
    "left_inner_finger",
    "right_outer_knuckle",
    "right_outer_finger",
    "right_inner_knuckle",
    "right_inner_finger",
)

# Visual mesh stem (under meshes/) for each link in ROBOTIQ_LINK_NAMES.
ROBOTIQ_MESH_STEM: dict[str, str] = {
    "base":                "robotiq_85_base_link_fine",
    "left_outer_knuckle":  "outer_knuckle_fine",
    "left_outer_finger":   "outer_finger_fine",
    "left_inner_knuckle":  "inner_knuckle_fine",
    "left_inner_finger":   "inner_finger_fine",
    "right_outer_knuckle": "outer_knuckle_fine",
    "right_outer_finger":  "outer_finger_fine",
    "right_inner_knuckle": "inner_knuckle_fine",
    "right_inner_finger":  "inner_finger_fine",
}

# Stroke limit on the drive joint (URDF upper bound).
FINGER_JOINT_MAX: float = 0.725


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rpy_to_R(*rpy)
    m[:3, 3] = xyz
    return m


def _axis_rot(axis: tuple[float, float, float], theta: float) -> np.ndarray:
    """Revolute joint transform: rotation by theta around `axis`."""
    ax = np.asarray(axis, dtype=float)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    x, y, z = ax
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    R = np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


# Static joint origins (parent -> child) from robotiq_arg85_description.URDF.
# Each entry: (parent, child, xyz, rpy, axis, mimic_multiplier).
# mimic_multiplier == 0.0 means "fixed" joint (no rotation applied).
_JOINTS: list[tuple[str, str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]] = [
    ("base", "left_outer_knuckle",
     (0.0306011444260539, 0.0,  0.0627920162695395), (0.0, 0.0, 0.0),
     (0.0, -1.0, 0.0), +1.0),
    ("left_outer_knuckle", "left_outer_finger",
     (0.0316910442266543, 0.0, -0.00193396375724605), (0.0, 0.0, 0.0),
     (0.0, 0.0, 0.0), 0.0),
    ("base", "left_inner_knuckle",
     (0.0127000000001501, 0.0,  0.0693074999999639), (0.0, 0.0, 0.0),
     (0.0, -1.0, 0.0), +1.0),
    ("left_inner_knuckle", "left_inner_finger",
     (0.034585310861294, 0.0,  0.0454970193817975), (0.0, 0.0, 0.0),
     (0.0, -1.0, 0.0), -1.0),
    ("base", "right_outer_knuckle",
     (-0.0306011444258893, 0.0,  0.0627920162695395), (0.0, 0.0, np.pi),
     (0.0,  1.0, 0.0), -1.0),
    ("right_outer_knuckle", "right_outer_finger",
     (0.0317095909367246, 0.0, -0.0016013564954687), (0.0, 0.0, 0.0),
     (0.0, 0.0, 0.0), 0.0),
    ("base", "right_inner_knuckle",
     (-0.0126999999998499, 0.0,  0.0693075000000361), (0.0, 0.0, np.pi),
     (0.0,  1.0, 0.0), -1.0),
    ("right_inner_knuckle", "right_inner_finger",
     (0.0341060475457406, 0.0,  0.0458573878541688), (0.0, 0.0, 0.0),
     (0.0,  1.0, 0.0), +1.0),
]


def robotiq_link_T(theta: float) -> dict[str, np.ndarray]:
    """Return per-link transforms in the Robotiq base frame.

    `theta` is the `finger_joint` angle in radians. All other joints follow via
    their URDF `mimic` relationship (multiplier x theta, with the axis sign
    already encoded in the static axis vector).
    """
    theta = float(np.clip(theta, 0.0, FINGER_JOINT_MAX))
    link_T: dict[str, np.ndarray] = {"base": np.eye(4)}
    for parent, child, xyz, rpy, axis, mimic in _JOINTS:
        T_origin = _origin(xyz, rpy)
        if mimic == 0.0:
            T_joint = np.eye(4)
        else:
            T_joint = _axis_rot(axis, mimic * theta)
        link_T[child] = link_T[parent] @ T_origin @ T_joint
    return link_T


def gripper_position_to_theta(gripper_position: float) -> float:
    """Map DROID `gripper_position` in [0, 1] (0 open, 1 closed) to drive angle."""
    g = float(np.clip(gripper_position, 0.0, 1.0))
    return g * FINGER_JOINT_MAX
