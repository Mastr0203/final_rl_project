"""
Custom Hopper environment compatible with Gymnasium (>= 1.0) and MuJoCo (>= 2.3).

The API now follows the Gymnasium spec:
    - reset()  →  obs, info
    - step()   →  obs, reward, terminated, truncated, info

See: https://gymnasium.farama.org/api/env/#gymnasium.Env
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
from gymnasium.spaces import Box

import numpy as np
import gymnasium as gym
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register


class CustomHopper(MujocoEnv, utils.EzPickle):
    """One-legged hopper with optional domain randomization."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 125}

    def __init__(
        self,
        domain: Optional[str] = None,
        xml_file: str = "hopper.xml",
        frameskip: int = 4,
        **kwargs,
    ):
        # --- EzPickle -------------------------------------------------- #
        utils.EzPickle.__init__(
            self,
            domain=domain,
            xml_file=xml_file,
            frameskip=frameskip,
            **kwargs,
        )
        self.observation_space = None
        MujocoEnv.__init__(self, xml_file, frameskip, observation_space=None, **kwargs)

        if getattr(self, "observation_space", None) is None:
            obs_sample = self._get_obs()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=obs_sample.dtype
            )

        # --------------------------------------------------------------- #
        self.original_masses = np.copy(self.model.body_mass[1:])
        self.udr_enabled = False  # By default desabled

        if domain == "source":
            self.model.body_mass[1] *= 0.7  # 30 % lighter torso

    # ------------------------------------------------------------------ #
    # Uniform-Domain‑Randomization implementation                        #
    # ------------------------------------------------------------------ #

    def set_random_parameters(self) -> None:
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self) -> np.ndarray:  # UDR Implementation
        masses = self.original_masses.copy()
        # Indexes: 0 = torso (fixed), 1 = thigh, 2 = leg, 3 = foot
        masses[1] = masses[1] * np.random.uniform(0.8, 1.2)  # thigh
        masses[2] = masses[2] * np.random.uniform(0.8, 1.2)  # leg
        masses[3] = masses[3] * np.random.uniform(0.8, 1.2)  # foot
        return masses

    def get_parameters(self) -> np.ndarray:
        return np.asarray(self.model.body_mass[1:])

    def set_parameters(self, masses: np.ndarray) -> None:
        self.model.body_mass[1:] = masses  # Masses retrieved form sample_parameters

    # ------------------------------------------------------------------ #
    # Core API                                                           #
    # ------------------------------------------------------------------ #

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        pos_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        pos_after, height, ang = self.data.qpos[0:3]

        alive_bonus = 1.0
        reward = (pos_after - pos_before) / self.dt + alive_bonus
        reward -= 1e-3 * np.square(action).sum()

        s = self.state_vector()
        terminated = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def reset_model(self) -> np.ndarray:
        """Randomize qpos/qvel at every episode start and return observation."""
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        if getattr(self, "udr_enabled", False):         #If udr is enabled, set random parameters and return this
            self.set_random_parameters()                #Otherwise returns default. False is the default.

        return self._get_obs()

    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.flat[1:], self.data.qvel.flat])

    def viewer_setup(self) -> None:
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    # Helpers to manipulate MuJoCo state -------------------------------- #

    def set_mujoco_state(self, state: np.ndarray) -> None:
        mjstate = deepcopy(self.get_mujoco_state())
        mjstate.qpos[0] = 0.0
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]
        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):  # type: ignore[override]
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        return self.sim.get_state()


# ---------------------------------------------------------------------- #
# Register environments                                                  #
# ---------------------------------------------------------------------- #

register(
    id="CustomHopper-v0",
    entry_point=f"{__name__}:CustomHopper",
    max_episode_steps=500,
)

register(
    id="CustomHopper-source-v0",
    entry_point=f"{__name__}:CustomHopper",
    max_episode_steps=500,
    kwargs={"domain": "source"},
)

register(
    id="CustomHopper-target-v0",
    entry_point=f"{__name__}:CustomHopper",
    max_episode_steps=500,
    kwargs={"domain": "target"},
)