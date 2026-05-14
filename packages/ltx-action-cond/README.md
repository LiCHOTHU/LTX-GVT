# ltx-action-cond

Action-conditioned video model on DROID for LTX-2 fine-tuning.

Builds a per-frame visual prior by running Franka Panda forward kinematics on the
commanded joint trajectory and projecting the link skeleton (or, later, the link
meshes) into the calibrated camera frame. The prior is fed to LTX-2 as an extra
conditioning channel alongside the action stream, the initial image, and the text
instruction (TIA2V).

## Modules

- `kinematics`  Franka Panda forward kinematics (Modified DH, no URDF needed for keypoints).
- `calibration` Parse the April 2025 DROID calibration JSONs (6-DoF extrinsic, 4-tuple intrinsic).
- `projection`  3D base-frame points -> image-frame pixels.
- `rendering`   Skeleton drawing on a frame (Stage 1). Mesh rendering goes here later (Stage 2).
- `droid`       RLDS episode loading + matching to the calibration map.

## Entry-point scripts

End-to-end runners live in `experiments/droid_action_cond/` at the repo root.
