"""Headless rendering of the Franka arm via pyrender + EGL.

Two modes:

  * `mode="mask"`  -- per-link flat color (articulation-aware segmentation).
    Black background. Each pixel covered by `linkN` gets a unique flat color.
    Renders with `RenderFlags.FLAT` (no lighting variation) so the output is
    a clean segmentation that the model can read pose identity from.

  * `mode="photo"` -- photoreal-ish render preserving each mesh's original
    PBR material. The Franka DAE files are tagged with white/light-grey/dark-grey
    PBR colors; with a directional light + ambient the result looks like the
    real robot in a black void. Use this when you want the model to see the
    actual robot appearance as conditioning.

Camera convention: DROID's `cam2base` is OpenCV (camera looks down +Z, +Y down).
pyrender uses OpenGL (camera looks down -Z, +Y up). We compose with
`diag(1, -1, -1, 1)` to convert.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender  # noqa: E402
import trimesh  # noqa: E402

from ltx_action_cond.kinematics import fk_urdf
from ltx_action_cond.robotiq_kinematics import (
    ROBOTIQ_LINK_NAMES,
    ROBOTIQ_MESH_STEM,
    gripper_position_to_theta,
    robotiq_link_T,
)

# Arm-only meshes (Franka link0..link7). The end-effector is mounted on `flange`.
ARM_LINK_NAMES: tuple[str, ...] = (
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
)

# Default Franka-Hand gripper meshes (loaded when `gripper="franka_hand"`).
FRANKA_HAND_LINK_NAMES: tuple[str, ...] = ("hand",)

# Per-link colors used only in `mode="mask"`. Robotiq links share grays.
LINK_COLORS: dict[str, tuple[float, float, float]] = {
    "link0": (0.20, 0.20, 0.20),
    "link1": (0.90, 0.30, 0.30),
    "link2": (0.95, 0.60, 0.20),
    "link3": (0.95, 0.90, 0.20),
    "link4": (0.30, 0.85, 0.30),
    "link5": (0.20, 0.70, 0.95),
    "link6": (0.30, 0.40, 0.95),
    "link7": (0.65, 0.30, 0.90),
    "hand":  (0.95, 0.20, 0.70),
    "finger": (1.00, 1.00, 1.00),
    "base":                (0.01, 0.01, 0.01),
    "left_outer_knuckle":  (0.90, 0.30, 0.30),
    "left_outer_finger":   (0.95, 0.60, 0.20),
    "left_inner_knuckle":  (0.30, 0.85, 0.30),
    "left_inner_finger":   (0.20, 0.70, 0.95),
    "right_outer_knuckle": (0.30, 0.40, 0.95),
    "right_outer_finger":  (0.65, 0.30, 0.90),
    "right_inner_knuckle": (0.95, 0.20, 0.70),
    "right_inner_finger":  (1.00, 1.00, 1.00),
}

# Franka Hand finger geometry from franka_hand.xacro:
# both fingers attach at hand-local xyz=(0, 0, 0.0584); right finger has rpy=(0,0,pi)
# so its local +Y is hand's -Y; both joints prismatic, mimic'd, q in [0, 0.04].
_FINGER_ATTACH_Z = 0.0584
_FINGER_MAX = 0.04

DEFAULT_MESH_DIR = Path(os.environ.get(
    "FRANKA_MESH_DIR",
    "/storage/project/r-agarg35-0/lwang831/droid/franka_meshes",
))

DEFAULT_ROBOTIQ_MESH_DIR = Path(os.environ.get(
    "ROBOTIQ_MESH_DIR",
    "/storage/project/r-agarg35-0/lwang831/droid/robotiq_arg85_description/meshes",
))

# Robotiq color in `mode="photo"`. Default matches the URDF spec (black plastic).
# Override with env ROBOTIQ_PHOTO_RGB="r,g,b" (0..1).
_ROBOTIQ_PHOTO_COLOR: tuple[float, float, float] = tuple(  # type: ignore[assignment]
    float(c) for c in os.environ.get("ROBOTIQ_PHOTO_RGB", "0.01,0.01,0.01").split(",")
)

_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

Mode = Literal["mask", "photo"]
Gripper = Literal["franka_hand", "robotiq_2f85"]


def _rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return T


def _tz(dz: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = dz
    return T


def _load_photo_mesh(path: Path) -> pyrender.Mesh:
    """Load a DAE preserving its original PBR materials. Sub-geometries become primitives.

    The Franka FER DAE files store sub-meshes inside a scene graph with per-node
    transforms (link1 has z=-0.192 m, link3 z=-0.121, link5 z=-0.259, link6
    z=-0.0148, link7 z=+0.052, link4 y=+0.003). We must apply each node's
    transform before handing the geometry to pyrender or the link looks
    "disconnected" -- chunks of the mesh appear at the link origin instead of
    at their authored offset within the link.
    """
    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Trimesh):
        return pyrender.Mesh.from_trimesh(loaded, smooth=True)
    primitives = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geom_name = loaded.graph[node_name]
        geom = loaded.geometry[geom_name].copy()
        if transform is not None and not np.allclose(transform, np.eye(4)):
            geom.apply_transform(transform)
        pm = pyrender.Mesh.from_trimesh(geom, smooth=True)
        primitives.extend(pm.primitives)
    return pyrender.Mesh(primitives=primitives)


def _load_mask_mesh(path: Path, color: tuple[float, float, float]) -> pyrender.Mesh:
    """Load a DAE flattened to a single mesh, overriding material with a flat link color."""
    tri = trimesh.load(str(path), force="mesh")
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*color, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return pyrender.Mesh.from_trimesh(tri, material=mat, smooth=False)


def _load_stl_photo_mesh(path: Path, color: tuple[float, float, float]) -> pyrender.Mesh:
    """Load an STL with a fixed PBR color (Robotiq URDF has no per-link textures)."""
    tri = trimesh.load(str(path), force="mesh")
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*color, 1.0],
        metallicFactor=0.2,
        roughnessFactor=0.7,
    )
    return pyrender.Mesh.from_trimesh(tri, material=mat, smooth=True)


def _load_stl_mask_mesh(path: Path, color: tuple[float, float, float]) -> pyrender.Mesh:
    """Load an STL flattened with a flat per-link color for segmentation mode."""
    tri = trimesh.load(str(path), force="mesh")
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*color, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return pyrender.Mesh.from_trimesh(tri, material=mat, smooth=False)


class FrankaMeshRenderer:
    """Headless renderer for the Franka arm in either segmentation or photoreal mode.

    Usage:
        r = FrankaMeshRenderer(size=(320, 180), mode="photo")
        rgb, depth = r.render(q=joints_t, K=K_rls, cam2base=cam2base_4x4,
                              gripper_position=g_t)
        r.close()
    """

    def __init__(
        self,
        size: tuple[int, int],
        mesh_dir: str | Path = DEFAULT_MESH_DIR,
        *,
        mode: Mode = "mask",
        gripper: Gripper = "franka_hand",
        robotiq_mesh_dir: str | Path = DEFAULT_ROBOTIQ_MESH_DIR,
        robotiq_mount_yaw: float = 1.5 * np.pi,
        robotiq_mount_z_offset: float = 0.0,
        znear: float = 0.01,
        zfar: float = 10.0,
        ambient_intensity: float = 0.4,
        directional_intensity: float = 3.0,
    ) -> None:
        if mode not in ("mask", "photo"):
            raise ValueError(f"mode must be 'mask' or 'photo', got {mode!r}")
        if gripper not in ("franka_hand", "robotiq_2f85"):
            raise ValueError(f"gripper must be 'franka_hand' or 'robotiq_2f85', got {gripper!r}")
        self.size = size
        self.mode = mode
        self.gripper = gripper
        self.znear = znear
        self.zfar = zfar
        self.ambient_intensity = ambient_intensity
        self.directional_intensity = directional_intensity
        # T_mount: flange -> Robotiq base. Default yaw=pi per the 2F-85 assembly
        # convention; z=0 (adapter-plate thickness is unknown, tune visually).
        self._robotiq_T_mount = _rz(robotiq_mount_yaw) @ _tz(robotiq_mount_z_offset)
        self._meshes: dict[str, pyrender.Mesh] = {}

        # Arm meshes (link0..link7) are always loaded from the Franka mesh dir.
        mesh_dir = Path(mesh_dir)
        for name in ARM_LINK_NAMES:
            path = mesh_dir / f"{name}.dae"
            if not path.exists():
                raise FileNotFoundError(f"missing mesh: {path}")
            self._meshes[name] = (
                _load_photo_mesh(path) if mode == "photo"
                else _load_mask_mesh(path, LINK_COLORS[name])
            )

        if gripper == "franka_hand":
            for name in (*FRANKA_HAND_LINK_NAMES, "finger"):
                path = mesh_dir / f"{name}.dae"
                if not path.exists():
                    raise FileNotFoundError(f"missing mesh: {path}")
                self._meshes[name] = (
                    _load_photo_mesh(path) if mode == "photo"
                    else _load_mask_mesh(path, LINK_COLORS[name])
                )
        else:  # robotiq_2f85
            robotiq_dir = Path(robotiq_mesh_dir)
            # Cache per-stem so reused meshes (left/right knuckle, finger) are loaded once.
            stem_cache: dict[str, pyrender.Mesh] = {}
            for link_name in ROBOTIQ_LINK_NAMES:
                stem = ROBOTIQ_MESH_STEM[link_name]
                if stem not in stem_cache:
                    path = robotiq_dir / f"{stem}.STL"
                    if not path.exists():
                        raise FileNotFoundError(f"missing Robotiq mesh: {path}")
                    if mode == "photo":
                        stem_cache[stem] = _load_stl_photo_mesh(path, _ROBOTIQ_PHOTO_COLOR)
                    else:
                        stem_cache[stem] = _load_stl_mask_mesh(path, LINK_COLORS[link_name])
                self._meshes[link_name] = stem_cache[stem]
        self._renderer = pyrender.OffscreenRenderer(viewport_width=size[0], viewport_height=size[1])

    @staticmethod
    def _finger_poses(T_hand: np.ndarray, gripper_position: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (T_leftfinger, T_rightfinger) in base frame.

        `gripper_position` follows DROID's convention: 0 = fully open, 1 = fully closed.
        Per-finger displacement q = (1 - gripper_position) * 0.04 m, mirrored.

        Both fingers attach at the hand-local point (0, 0, 0.0584). The right finger's
        joint frame is yawed 180° around Z relative to the hand, so a +q slide along
        the joint's local +Y axis becomes a -q slide in the hand's Y axis — i.e. the
        two fingers separate symmetrically. The slide therefore has to be applied
        AFTER the yaw (T_attach @ T_yaw @ T_slide), not packed into the yaw matrix.
        """
        q = float(np.clip((1.0 - gripper_position) * _FINGER_MAX, 0.0, _FINGER_MAX))

        T_attach = np.eye(4)
        T_attach[2, 3] = _FINGER_ATTACH_Z

        # Left finger: no yaw, slides +q along the joint axis (= +Y in hand frame).
        T_slide = np.eye(4); T_slide[1, 3] = +q
        T_left = T_hand @ T_attach @ T_slide

        # Right finger: yaw 180° around Z first, then slide +q along the yawed +Y
        # axis (which is -Y in hand frame) -- so the right finger lands at (0, -q, 0.0584)
        # in the hand frame, mirroring the left finger across the Y=0 plane.
        c, s = np.cos(np.pi), np.sin(np.pi)
        T_right_yaw = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T_right = T_hand @ T_attach @ T_right_yaw @ T_slide

        return T_left, T_right

    def render(
        self,
        q: np.ndarray,
        K: np.ndarray,
        cam2base: np.ndarray,
        *,
        gripper_position: float = 0.0,
        background: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        gripper_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render one frame. Returns (color uint8 HxWx3, depth float32 HxW).

        Pixels with no robot are at the background color and depth=0.

        `gripper_only=True` skips the arm links and draws only the end-effector
        (Franka hand or Robotiq). Used by the wrist-camera context renderer.
        """
        link_T = fk_urdf(q)

        if self.mode == "photo":
            ambient = [self.ambient_intensity] * 3
            scene = pyrender.Scene(bg_color=list(background), ambient_light=ambient)
        else:
            scene = pyrender.Scene(bg_color=list(background), ambient_light=[0.0, 0.0, 0.0])

        if not gripper_only:
            for name in ARM_LINK_NAMES:
                scene.add(self._meshes[name], pose=link_T[name])

        if self.gripper == "franka_hand":
            scene.add(self._meshes["hand"], pose=link_T["hand"])
            T_left, T_right = self._finger_poses(link_T["hand"], gripper_position)
            scene.add(self._meshes["finger"], pose=T_left)
            scene.add(self._meshes["finger"], pose=T_right)
        else:  # robotiq_2f85
            T_robotiq_base = link_T["flange"] @ self._robotiq_T_mount
            theta = gripper_position_to_theta(gripper_position)
            for link_name, T_local in robotiq_link_T(theta).items():
                scene.add(self._meshes[link_name], pose=T_robotiq_base @ T_local)

        cam_pose = cam2base @ _CV_TO_GL
        cam = pyrender.IntrinsicsCamera(
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            znear=self.znear, zfar=self.zfar,
        )
        scene.add(cam, pose=cam_pose)

        if self.mode == "photo":
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.directional_intensity)
            scene.add(light, pose=cam_pose)  # key light, from camera direction
            # Fill light from above (+Z in base) so the upper-arm shaft isn't pitch-black
            # against a black background.
            fill = pyrender.DirectionalLight(color=np.ones(3), intensity=self.directional_intensity * 0.5)
            fill_pose = np.eye(4)
            fill_pose[2, 3] = 2.0
            scene.add(fill, pose=fill_pose)
            color, depth = self._renderer.render(scene)
        else:
            color, depth = self._renderer.render(scene, flags=pyrender.RenderFlags.FLAT)

        return color, depth

    def close(self) -> None:
        self._renderer.delete()
