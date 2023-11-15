from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm, Logger
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn,TrainFreq, TrainFrequencyUnit
from stable_baselines3 import SAC, PPO, DQN, DDPG, TD3, A2C

import torch


from stable_baselines3.common.utils import should_collect_more_steps, TrainFreq, get_schedule_fn
import wandb

from gymnasium import spaces

# Class of buffer just storing the states, observation, and its corresponding action
class StatesActionsBuffer():
    def __init__(self, buffer_size, action_dim, observation_dim, *args, **kwargs):
        self.buffer_size = buffer_size
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.states = dict()
        self.observations = np.zeros((buffer_size, observation_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.index = 0
        
    def add(self, states, observations, actions):
        # states, actions, and observations are lists of states, actions, and observations
        # We need to iterate through each of the states, actions, and observations and add them to the buffer
        # We also need to make sure that the index is not greater than the buffer size
        if self.index + states.shape[0] >= self.buffer_size:
            return
        # Iterate through the states, actions, and observations and add them to the numpy buffer
        for i in range(states.shape[0]):
            self.states[len(self.states)] = states[i]
            self.observations[-1] = observations[i]
            self.actions[-1] = actions[i]
            # Increment the index
        self.index += states.shape[0]

            
            
    def sample(self, batch_size):
        indices = np.random.randint(0, self.index, batch_size)
        return [self.states[i] for i in indices.tolist()], self.observations[indices], self.actions[indices]
    
    def __len__(self):
        return self.index


class HelperEvalCallback(EvalCallback):
    def __init__(self, log_true_q, gamma=0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_true_q = log_true_q
        self.gamma = gamma
        
    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.logger = HelperLogger(self.logger)
        self.model = model


    def _on_step(self) -> bool:
        self.model.policy.jsrl_evaluation = True
        self.model.jsrl_evaluation = True
        if self.n_calls % 10 == 0 and self.log_true_q:
            self.monte_carlo_evaluation()
        super()._on_step()
        self.model.policy.jsrl_evaluation = False
        self.model.jsrl_evaluation = False
    
    def monte_carlo_evaluation(self):
        # Compute true q values using the current model monte carlo style by initializing the environment from the 10000 starting states and running the model for 1000 steps from the replay buffer
        # Then, compute the mean of the true q values and log it
        # Also, log the Q values of the current model on the same 10000 starting states
        
        # Sample the starting states from the replay buffer
        starting_states, observations, actions = self.model.state_action_buffer.sample(1000)      
        observations = torch.from_numpy(observations).float().to(self.model.device)
        actions = torch.from_numpy(actions).float().to(self.model.device)
        pred_q_values = self.model.critic(observations, actions)
        
        # Get the mean of the pred_q_values
        mean_pred_q_values = np.mean(pred_q_values[0].cpu().detach().numpy()[:,0])
        
        # Get the true q values by running the model for 1000 steps from the starting states
        true_q_values = []
        for i in range(len(starting_states)):
            env = self.model.env.envs[0]
            env.reset()
            # Reset the environment to the starting state
            env.set_env_state(starting_states[i])
            # Get the observation
            observation = env.get_obs()
            # Get the action
            action, _ = self.model.policy.predict(observation, deterministic=True)
            # Get the reward
            reward = 0
            # Get the info
            info = {}
            # Iterate through the environment for 1000 steps
            for j in range(1000):
                # Get the next observation
                next_observation, next_reward, terminated, truncate, next_info = env.step(action)
                # Get the next action
                next_action, _ = self.model.policy.predict(next_observation, deterministic=True)
                # Add the reward to the total reward
                reward += (self.gamma**j) * next_reward
                # Set the observation to the next observation
                observation = next_observation
                # Set the action to the next action
                action = next_action

                # Set the info to the next info
                info = next_info
                if truncate:
                    env.reset()
                    break
               
            # Append the reward to the list of true q values
            true_q_values.append(reward)
            
        mean_true_q_values = np.mean(true_q_values)
        
        #log the mean of the true q values and the mean of the pred q values
        self.logger.record("jsrl/mean_true_q_values", mean_true_q_values)
        self.logger.record("jsrl/mean_pred_q_values", mean_pred_q_values)
        
        
          



class HelperLogger():
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
        # key = key.replace("eval/", "jsrl/")
        self._logger.record(key, value, exclude)

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        self._logger.dump(step)


def get_algorithm(Algorithm: BaseAlgorithm):
    class HelpedAlgorithm(Algorithm):
        def __init__(self, policy, eval_env, learning_starts, log_true_q, eval_freq, n_eval_episodes, gamma=0.99, *args, **kwargs):
            if isinstance(policy, str):
                policy = self._get_policy_from_name(policy)
            else:
                policy = policy
            kwargs["learning_starts"] = learning_starts
            super().__init__(policy, *args, **kwargs)
            self._timesteps = np.zeros((self.env.num_envs), dtype=np.int32)
            self.eval_env = eval_env
            self.log_true_q = log_true_q
            self.state_action_buffer = StatesActionsBuffer(100000, self.policy.action_space.shape[0], self.policy.observation_space.shape[0])
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.gamma = gamma


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
            eval_callback = HelperEvalCallback(
                log_true_q=self.log_true_q,
                gamma=self.gamma,
                eval_env=self.eval_env,
                eval_freq=self.eval_freq,
                n_eval_episodes=self.n_eval_episodes,
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
                
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
            self._timesteps += 1
            self._timesteps[self.env.buf_dones] = 0
            return action, state
        
        # Write a collect_rollouts function that is the same as the one in stable_baselines3.common.on_policy_algorithm but with the following changes:
        # 1. When the np.random.random() < self.epsilon, the action is random
        
        def collect_rollouts(
                self,
                env: VecEnv,
                callback: BaseCallback,
                train_freq: TrainFreq,
                replay_buffer: ReplayBuffer,
                action_noise: Optional[ActionNoise] = None,
                learning_starts: int = 0,
                log_interval: Optional[int] = None,
            ) -> RolloutReturn:
                """
                Collect experiences and store them into a ``ReplayBuffer``.

                :param env: The training environment
                :param callback: Callback that will be called at each step
                    (and at the beginning and end of the rollout)
                :param train_freq: How much experience to collect
                    by doing rollouts of current policy.
                    Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
                    or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
                    with ``<n>`` being an integer greater than 0.
                :param action_noise: Action noise that will be used for exploration
                    Required for deterministic policy (e.g. TD3). This can also be used
                    in addition to the stochastic policy for SAC.
                :param learning_starts: Number of steps before learning for the warm-up phase.
                :param replay_buffer:
                :param log_interval: Log data every ``log_interval`` episodes
                :return:
                """
                # Switch to eval mode (this affects batch norm / dropout)
                self.policy.set_training_mode(False)

                num_collected_steps, num_collected_episodes = 0, 0

                assert isinstance(env, VecEnv), "You must pass a VecEnv"
                assert train_freq.frequency > 0, "Should at least collect one step or episode."

                if env.num_envs > 1:
                    assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

                # Vectorize action noise if needed
                if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
                    action_noise = VectorizedActionNoise(action_noise, env.num_envs)

                if self.use_sde:
                    self.actor.reset_noise(env.num_envs)

                callback.on_rollout_start()
                continue_training = True
                while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                        # Sample a new noise matrix
                        self.actor.reset_noise(env.num_envs)

                    actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

                    #iterate through the envs and get all of its states
                    states = np.empty(env.num_envs, dtype=object)
                    for i in range(env.num_envs):
                        states[i] = np.array(env.envs[i].get_env_state(), dtype=object)

                    # Rescale and perform action
                    new_obs, rewards, dones, infos = env.step(actions)

                    self.num_timesteps += env.num_envs
                    num_collected_steps += 1

                    # Give access to local variables
                    callback.update_locals(locals())
                    # Only stop training if return value is False, not when it is None.
                    if callback.on_step() is False:
                        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

                    # Retrieve reward and episode length if using Monitor wrapper
                    self._update_info_buffer(infos, dones)

                    # Store data in replay buffer (normalized action and unnormalized observation)
                    self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
                    self.state_action_buffer.add(states, new_obs, actions)

                    self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                    # For DQN, check if the target network should be updated
                    # and update the exploration schedule
                    # For SAC/TD3, the update is dones as the same time as the gradient update
                    # see https://github.com/hill-a/stable-baselines/issues/900
                    self._on_step()

                    for idx, done in enumerate(dones):
                        if done:
                            # Update stats
                            num_collected_episodes += 1
                            self._episode_num += 1

                            if action_noise is not None:
                                kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                                action_noise.reset(**kwargs)

                            # Log training infos
                            if log_interval is not None and self._episode_num % log_interval == 0:
                                self._dump_logs()
                callback.on_rollout_end()

                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


    return HelpedAlgorithm