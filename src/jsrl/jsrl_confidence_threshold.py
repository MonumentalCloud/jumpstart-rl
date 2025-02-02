import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import TD3
from stable_baselines3.common.type_aliases import GymEnv
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm, Logger
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from .RND import RNDEstimator
import torch
import wandb

class JSRLAfterEvalCallback(BaseCallback):
    def __init__(self, policy, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.logger = logger
        self.best_moving_mean_reward = -np.inf
        self.tolerated_moving_mean_reward = -np.inf
        self.mean_rewards = np.full(policy.window_size, -np.inf, dtype=np.float32)
        self.mean_uncertainty = np.full(policy.window_size, -np.inf, dtype=np.float32)
        self.last_uncertainty = 1.0

    def _on_step(self) -> bool:
        self.policy.jsrl_evaluation = False
        # self.logger.record("jsrl/horizon", self.policy.horizon)

        if self.policy.strategy == "random":
            return True

        self.mean_rewards = np.roll(self.mean_rewards, 1)
        self.mean_uncertainty = np.roll(self.mean_uncertainty, 1)

        self.mean_rewards[0] = self.parent.last_mean_reward
        self.mean_uncertainty[0] = self.last_uncertainty

        moving_mean_reward = np.mean(self.mean_rewards)
        moving_mean_uncertainty = np.mean(self.mean_uncertainty)
        
        guide_inference = self.policy.guide_inference/self.policy.total_inference

        self.logger.record("jsrl/moving_mean_reward", moving_mean_reward)
        self.logger.record("jsrl/moving_mean_uncertainty", moving_mean_uncertainty)
        self.logger.record("jsrl/best_moving_mean_reward", self.best_moving_mean_reward)
        self.logger.record("jsrl/tolerated_moving_mean_reward", self.tolerated_moving_mean_reward)
        self.logger.record("jsrl/guide_inference", guide_inference)
        self.logger.record("jsrl/absolute_guide_inference", self.policy.guide_inference)
        self.logger.record("jsrl/cumulative_guide_inference", self.policy.cumulative_guide_inference)
        self.logger.record("jsrl/total_inference", self.policy.total_inference)
        self.logger.dump(self.num_timesteps)
        # Basically replace the above commented out logger.record commands into wandb logging
        # wandb.log({"jsrl/moving_mean_reward": moving_mean_reward})
        # wandb.log({"jsrl/moving_mean_uncertainty": moving_mean_uncertainty})
        # wandb.log({"jsrl/best_moving_mean_reward": self.best_moving_mean_reward})
        # wandb.log({"jsrl/tolerated_moving_mean_reward": self.tolerated_moving_mean_reward})
        # wandb.log({"jsrl/guide_inference": guide_inference})
        # wandb.log({"jsrl/absolute_guide_inference": self.policy.guide_inference})
        # wandb.log({"jsrl/total_inference": self.policy.total_inference})
        

        if self.mean_rewards[-1] == -np.inf or self.policy.horizon <= 0:
            return True
        elif self.best_moving_mean_reward == -np.inf:
            self.best_moving_mean_reward = moving_mean_reward

        if moving_mean_reward >= self.best_moving_mean_reward:
            self.tolerated_moving_mean_reward = moving_mean_reward - self.policy.tolerance * np.abs(moving_mean_reward)
            self.best_moving_mean_reward = max(self.best_moving_mean_reward, moving_mean_reward)

        self.policy.guide_inference = 0
        self.policy.total_inference = 0
        self.policy.jsrl_evaluation = False
        return True


class JSRLEvalCallback(EvalCallback):
    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.logger = JSRLLogger(self.logger)

    def _on_step(self) -> bool:
        self.model.policy.jsrl_evaluation = True
        super()._on_step()
        self.model.policy.jsrl_evaluation = False


class JSRLLogger():
    def __init__(self, logger: Logger):
        self._logger = logger

    def record(self, key: str, value: Any, exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        key = key.replace("eval/", "jsrl/")
        self._logger.record(key, value, exclude)

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        self._logger.dump(step)

def get_jsrl_threshold_policy(ExplorationPolicy: BasePolicy, uncertainty_estimator: RNDEstimator, threshold: float):
    def get_jsrl_policy(ExplorationPolicy: BasePolicy):

        class JSRLPolicy(ExplorationPolicy):
            def __init__(
                self,
                *args,
                guide_policy: BasePolicy = None,
                max_horizon: int = 0,
                horizons: List[int] = [0],
                tolerance: float = 0.0,
                strategy: str = "curriculum",
                window_size: int = 1,
                eval_freq: int = 1000,
                n_eval_episodes: int = 20,
                **kwargs,
            ) -> None:
                super().__init__(*args, **kwargs)
                self.guide_policy = guide_policy
                self.tolerance = tolerance
                assert strategy in ["curriculum", "random", "threshold"], f"strategy: '{strategy}' must be 'curriculum' or 'random' or 'threshold'"
                self.strategy = strategy
                self.horizon = max_horizon
                self.max_horizon = max_horizon
                assert window_size > 0, f"window_size: {window_size} must be greater than 0"
                self.window_size = window_size
                self.eval_freq = eval_freq
                if self.strategy == "curriculum" or self.strategy == "threshold":
                    self.n_eval_episodes = n_eval_episodes
                else:
                    self.n_eval_episodes = 0
                self.jsrl_evaluation = False


            # def predict(
            #     self,
            #     observation: Union[np.ndarray, Dict[str, np.ndarray]],
            #     timesteps: np.ndarray,
            #     state: Optional[Tuple[np.ndarray, ...]] = None,
            #     episode_start: Optional[np.ndarray] = None,
            #     deterministic: bool = False,
            # ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            #     """
            #     Get the policy action from an observation (and optional hidden state).
            #     Includes sugar-coating to handle different observations (e.g. normalizing images).

            #     :param observation: the input observation
            #     :param timesteps: the number of timesteps since the beginning of the episode
            #     :param state: The last hidden states (can be None, used in recurrent policies)
            #     :param episode_start: The last masks (can be None, used in recurrent policies)
            #         this correspond to beginning of episodes,
            #         where the hidden states of the RNN must be reset.
            #     :param deterministic: Whether or not to return deterministic actions.
            #     :return: the model's action and the next hidden state
            #         (used in recurrent policies)
            #     """
            #     horizon = self.horizon
            #     if not self.training and not self.jsrl_evaluation:
            #         horizon = 0
            #     timesteps_lte_horizon = timesteps <= horizon
            #     timesteps_gt_horizon = timesteps > horizon
            #     if isinstance(observation, dict):
            #         observation_lte_horizon = {k: v[timesteps_lte_horizon] for k, v in observation.items()}
            #         observation_gt_horizon = {k: v[timesteps_gt_horizon] for k, v in observation.items()}
            #     elif isinstance(observation, np.ndarray):
            #         observation_lte_horizon = observation[timesteps_lte_horizon]
            #         observation_gt_horizon = observation[timesteps_gt_horizon]
            #     if state is not None:
            #         state_lte_horizon = state[timesteps_lte_horizon]
            #         state_gt_horizon = state[timesteps_gt_horizon]
            #     else:
            #         state_lte_horizon = None
            #         state_gt_horizon = None
            #     if episode_start is not None:
            #         episode_start_lte_horizon = episode_start[timesteps_lte_horizon]
            #         episode_start_gt_horizon = episode_start[timesteps_gt_horizon]
            #     else:
            #         episode_start_lte_horizon = None
            #         episode_start_gt_horizon = None

            #     action = np.zeros((len(timesteps), *self.action_space.shape), dtype=self.action_space.dtype)
            #     if state is not None:
            #         state = np.zeros((len(timesteps), *state.shape[1:]), dtype=state_lte_horizon.dtype)

            #     if timesteps_lte_horizon.any():
            #         action_lte_horizon, state_lte_horizon = self.guide_policy.predict(
            #             observation_lte_horizon, state_lte_horizon, episode_start_lte_horizon, deterministic
            #         )
            #         action[timesteps_lte_horizon] = action_lte_horizon
            #         if state is not None:
            #             state[timesteps_lte_horizon] = state_lte_horizon

            #     if timesteps_gt_horizon.any():
            #         action_gt_horizon, state_gt_horizon = super().predict(
            #             observation_gt_horizon, state_gt_horizon, episode_start_gt_horizon, deterministic
            #         )
            #         action[timesteps_gt_horizon] = action_gt_horizon
            #         if state is not None:
            #             state[timesteps_gt_horizon] = state_gt_horizon

            #     return action, state

            def update_horizon(self, decrease_quantity = 0) -> None:
                """
                Update the horizon based on the current strategy.
                """
                if self.strategy == "threshold":
                    self.horizon -= decrease_quantity
            
        return JSRLPolicy

    JSRLPolicy = get_jsrl_policy(ExplorationPolicy)
    class JSRLThresholdPolicy(JSRLPolicy):
        def __init__(
            self,
            *args,
            guide_policy: BasePolicy = None,
            max_horizon: int = 0,
            horizons: List[int] = [0],
            tolerance: float = 0.0,
            strategy: str = "curriculum",
            window_size: int = 1,
            eval_freq: int = 1000,
            n_eval_episodes: int = 20,
            **kwargs,
        ) -> None:
            super().__init__(
                *args,
                guide_policy=guide_policy,
                max_horizon=max_horizon,
                horizons=horizons,
                tolerance=tolerance,
                strategy=strategy,
                window_size=window_size,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                **kwargs,
            )
            self.threshold = threshold
            self.guide_inference = 0
            self.cumulative_guide_inference = 0
            self.total_inference = 0
            self.uncertainty_estimator = uncertainty_estimator

        def get_uncertainty(self, observation):
            """
            Get the confidence of the action taken by the exploration policy.
            """
            rnd_reward = self.uncertainty_estimator.get_reward(observation)
            uncertainty = rnd_reward.mean()
            wandb.log({"uncertainty": uncertainty})
            return uncertainty
        
        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            timesteps: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            """
            Get the policy action from an observation (and optional hidden state).
            Includes sugar-coating to handle different observations (e.g. normalizing images).

            :param observation: the input observation
            :param timesteps: the number of timesteps since the beginning of the episode
            :param state: The last hidden states (can be None, used in recurrent policies)
            :param episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            :param deterministic: Whether or not to return deterministic actions.
            :return: the model's action and the next hidden state
                (used in recurrent policies)
            """
            self.total_inference += 1
            obs_tensor = torch.FloatTensor(observation['observation'])
            if self.jsrl_evaluation:
                action, state = super().predict(
                        observation, state, episode_start, deterministic
                    )
                return action, state
            else:    
                uncertainty = self.get_uncertainty(obs_tensor)
                if uncertainty < self.threshold:
                    action, state = super().predict(
                            observation, state, episode_start, deterministic
                        )
                else:
                    self.guide_inference += 1
                    self.cumulative_guide_inference += 1
                    action, state = self.guide_policy.predict(
                            observation, state, episode_start, deterministic
                        )
                
                #convert observation into tensors
                self.uncertainty_estimator.update_obs(obs_tensor)
                self.uncertainty_estimator.update_model()
                return action, state
        
    return JSRLThresholdPolicy

def get_jsrl_algorithm(Algorithm: BaseAlgorithm, uncertainty_estimator: RNDEstimator, threshold: float):
    class JSRLAlgorithm(Algorithm):
        def __init__(self, policy, *args, **kwargs):
            if isinstance(policy, str):
                policy = self._get_policy_from_name(policy)
            else:
                policy = policy
            policy = get_jsrl_threshold_policy(policy, uncertainty_estimator, threshold)
            kwargs["learning_starts"] = 0
            super().__init__(policy, *args, **kwargs)
            self._timesteps = np.zeros((self.env.num_envs), dtype=np.int32)

        def _init_callback(
            self,
            callback: MaybeCallback,
            progress_bar: bool = False,
        ) -> BaseCallback:
            """
            :param callback: Callback(s) called at every step with state of the algorithm.
            :param progress_bar: Display a progress bar using tqdm and rich.
            :return: A hybrid callback calling `callback` and performing evaluation.
            """
            callback = super()._init_callback(callback, progress_bar)
            eval_callback = JSRLEvalCallback(
                self.env,
                callback_after_eval=JSRLAfterEvalCallback(
                    self.policy,
                    self.logger,
                    verbose=self.verbose,
                ),
                eval_freq=self.policy.eval_freq,
                n_eval_episodes=self.policy.n_eval_episodes,
                verbose=self.verbose,
            )
            callback = CallbackList(
                [
                    callback,
                    eval_callback,
                ]
            )
            callback.init_callback(self)
            return callback

        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            """
            Get the policy action from an observation (and optional hidden state).
            Includes sugar-coating to handle different observations (e.g. normalizing images).

            :param observation: the input observation
            :param state: The last hidden states (can be None, used in recurrent policies)
            :param episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            :param deterministic: Whether or not to return deterministic actions.
            :return: the model's action and the next hidden state
                (used in recurrent policies)
            """
            action, state = self.policy.predict(observation, self._timesteps, state, episode_start, deterministic)

            self._timesteps += 1
            self._timesteps[self.env.buf_dones] = 0
            if self.policy.strategy == "random" and self.env.buf_dones.any():
                self.policy.update_horizon()
            return action, state

    return JSRLAlgorithm