"""
Post-migration compatibility with Gymnasium (>= 1.0).

The old wrapper based on `mujoco_py` is no longer needed: we re-exported
simply the `MujocoEnv` class from Gymnasium, then instructions like

    from env.mujoco_env import MujocoEnv

continue to work without modification elsewhere.
"""

from gymnasium.envs.mujoco import MujocoEnv  # noqa: F401

__all__ = ["MujocoEnv"]