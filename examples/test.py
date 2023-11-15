from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["reach-v2-goal-observable"]
env = cls(seed=1)
env.observation_space.shape
env.reset()
print(len(env.get_env_state()[0]))

from gymnasium.envs.mujoco import MujocoEnv as mjenv_gym