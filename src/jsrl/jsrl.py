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


class JSRLAfterEvalCallback(BaseCallback):
    def __init__(self, policy, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.logger = logger
        self.best_moving_mean_reward = -np.inf
        self.tolerated_moving_mean_reward = -np.inf
        self.mean_rewards = np.full(policy.window_size, -np.inf, dtype=np.float32)

    def _on_step(self) -> bool:
        self.policy.jsrl_evaluation = False
        self.logger.record("jsrl/horizon", self.policy.horizon)

        if self.policy.strategy == "random":
            return True

        self.mean_rewards = np.roll(self.mean_rewards, 1)
        self.mean_rewards[0] = self.parent.last_mean_reward
        moving_mean_reward = np.mean(self.mean_rewards)
        
        # guide_inference = self.policy.guide_inference/self.policy.total_inference

        self.logger.record("jsrl/moving_mean_reward", moving_mean_reward)
        self.logger.record("jsrl/best_moving_mean_reward", self.best_moving_mean_reward)
        self.logger.record("jsrl/tolerated_moving_mean_reward", self.tolerated_moving_mean_reward)
        # self.logger.record("jsrl/guide_inference", guide_inference)
        # self.logger.record("jsrl/absolute_guide_inference", self.policy.guide_inference)
        # self.logger.record("jsrl/cumulative_guide_inference", self.policy.cumulative_guide_inference)
        # self.logger.record("jsrl/total_inference", self.policy.total_inference)
        self.logger.dump(self.num_timesteps)

        if self.mean_rewards[-1] == -np.inf or self.policy.horizon <= 0:
            return True
        elif self.best_moving_mean_reward == -np.inf:
            self.best_moving_mean_reward = moving_mean_reward
        elif moving_mean_reward > self.tolerated_moving_mean_reward:
            self.policy.update_horizon()

        if moving_mean_reward >= self.best_moving_mean_reward:
            self.tolerated_moving_mean_reward = moving_mean_reward - self.policy.tolerance * np.abs(moving_mean_reward)
            self.best_moving_mean_reward = max(self.best_moving_mean_reward, moving_mean_reward)

        self.policy.guide_inference = 0
        self.policy.total_inference = 0

        return True
    
# Class of buffer just storing the states, observation, and its corresponding action
class JSRLStatesActionsBuffer():
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


class JSRLEvalCallback(EvalCallback):
    def init_callback(self, model: BaseAlgorithm, gamma=0.99) -> None:
        super().init_callback(model)
        self.logger = JSRLLogger(self.logger)
        self.model = model
        self.log_true_q = self.model.log_true_q
        self.gamma = gamma

    def _on_step(self) -> bool:
        self.model.policy.jsrl_evaluation = True
        self.model.jsrl_evaluation = True
        if self.n_calls % 100000 == 0 and self.log_true_q:
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
        # key = key.replace("eval/", "jsrl/")
        self._logger.record(key, value, exclude)

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        self._logger.dump(step)
        
# class of JSRLGuides that takes in pretrained models and returns the action of the guide policy
class JSRLGuides():
    def __init__(self, guides_directory, priority=False, cuda=1, *args, **kwargs):
        self.guides = []
        for guide in guides_directory:
            self.guides.append(self.load(guide, device=f"cuda:{cuda}"))
        
        if priority:
            probability = np.array([1/(i+1) for i in range(len(self.guides))])
            self.p = np.exp(probability)/np.sum(np.exp(probability))
        else:
            # Uniform sampling distribution
            self.p = np.array([1/len(self.guides) for i in range(len(self.guides))])
    
    # def load takes in a path to a pretrained model and returns the loaded model
    # This method should be applicable to all pretrained model types in stable baselines 3
    def load(self, path, device="cuda:1"):
        """
        Load a pretrained model from a path.
        The path will be in the following format: /ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_guide_{model_name}_{'sparse' if sparse else 'dense'}

        :param path: The path to the pretrained model.
        :return: The loaded model.
        """
        switch = {
            "sac": SAC,
            "ppo": PPO,
            "dqn": DQN,
            "ddpg": DDPG,
            "td3": TD3,
            "a2c": A2C,
        }
        model_name = path.split("/")[-2].split("_")[2]
        model_class = switch[model_name]
        model = model_class.load(path, device=device)
        
        return model.policy
    
    def predict(self, observation, state, episode_start, deterministic):
        # With the sampling distribution chance, we can choose which guide policy to use and return the action of that guide policy
        # If the sampling distribution uniform, all guide policy gets an equal chance of being chosen as the guide policy at this timestep
        # If the sampling distribution is not uniform, the guide policy will be sampled according to the sampling distribution
        # The sampling distribution is a list of probabilities that sum to 1
        
        # If the sampling distribution is None, we will use a uniform sampling distribution
        
        index = np.random.choice(len(self.guides), p=self.p)
            
        guide_policy = self.guides[index]
        action, state = guide_policy.predict(observation, state, episode_start, deterministic)          
        
        return action, state
              
    


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
            eval_freq: int = 100,
            n_eval_episodes: int = 0,
            epsilon: float = 0.1,
            delay_steps: int = 0,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.guide_policy = guide_policy
            self.tolerance = tolerance
            assert strategy in ["curriculum", "random"], f"strategy: '{strategy}' must be 'curriculum' or 'random'"
            self.strategy = strategy
            self.horizon_step = 0
            self.max_horizon = max_horizon
            self.horizons = horizons
            assert window_size > 0, f"window_size: {window_size} must be greater than 0"
            self.window_size = window_size
            self.eval_freq = eval_freq
            if self.strategy == "curriculum":
                self.n_eval_episodes = n_eval_episodes
            else:
                self.n_eval_episodes = 0
            self.jsrl_evaluation = False
            self.guide_inference = 0
            self.cumulative_guide_inference = 0
            self.total_inference = 0
            self.epsilon = epsilon
            self.delay_steps = delay_steps

        @property
        def horizon(self):
            return self.horizons[self.horizon_step]
        
        def save(self, save_path):
            """
            Save the model to a file.

            :param save_path: the path to the file
            """
            super().save(save_path)

        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            timesteps: Optional[np.ndarray] = None,
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
            if self.jsrl_evaluation:
                # if state == None:
                #     action, state = super().predict(observation, observation, timesteps == 0, deterministic)
                # else:
                action, state = super().predict(observation, state, timesteps == 0, deterministic)
                return action, state
            else:                    
                self.total_inference += 1
                horizon = self.horizon
                # if not self.training and not self.jsrl_evaluation:
                #     horizon = 0
                timesteps_lte_horizon = timesteps <= horizon
                timesteps_gt_horizon = timesteps > horizon
                # if isinstance(observation, dict):
                #     observation_lte_horizon = {k: v[timesteps_lte_horizon] for k, v in observation.items()}
                #     observation_gt_horizon = {k: v[timesteps_gt_horizon] for k, v in observation.items()}
                # elif isinstance(observation, np.ndarray):
                #     # repeat the timesteps so that it matches the dimension of observation
                #     # make a mask called timesteps_lte_horizon that is the same shape as observation and is True for timesteps <= horizon
                #     timesteps_lte_horizon = [timesteps_lte_horizon[0]] * observation.shape[1]
                #     observation = observation.squeeze()
                #     print(observation.shape)
                #     observation_lte_horizon = observation[timesteps_lte_horizon]
                #     observation_gt_horizon = observation[timesteps_gt_horizon]
                # else:
                #     observation_lte_horizon = observation
                #     observation_gt_horizon = observation
                if state is not None:
                    state_lte_horizon = state[timesteps_lte_horizon]
                    state_gt_horizon = state[timesteps_gt_horizon]
                else:
                    state_lte_horizon = None
                    state_gt_horizon = None
                if episode_start is not None:
                    episode_start_lte_horizon = episode_start[timesteps_lte_horizon]
                    episode_start_gt_horizon = episode_start[timesteps_gt_horizon]
                else:
                    episode_start_lte_horizon = None
                    episode_start_gt_horizon = None

                action = np.zeros((len(timesteps), *self.action_space.shape), dtype=self.action_space.dtype)
                if state is not None:
                    state = np.zeros((len(timesteps), *state.shape[1:]), dtype=state_lte_horizon.dtype)
                    
                

                if len(timesteps_lte_horizon) > 1:
                    raise ValueError("timesteps_lte_horizon is more than one")
                if timesteps_lte_horizon[0]:
                # if timesteps_lte_horizon.any():
                    action_lte_horizon, state_lte_horizon = self.guide_policy.predict(
                        observation, state_lte_horizon, episode_start_lte_horizon, deterministic
                    )
                    num_true = np.sum(timesteps_lte_horizon)
                    self.guide_inference += 1
                    self.cumulative_guide_inference += 1
                    action[timesteps_lte_horizon] = action_lte_horizon
                    if state is not None:
                        state[timesteps_lte_horizon] = state_lte_horizon

                elif timesteps_gt_horizon[0]:
                    action_gt_horizon, state_gt_horizon = super().predict(
                        observation, state_gt_horizon, episode_start_gt_horizon, deterministic
                    )
                    action[timesteps_gt_horizon] = action_gt_horizon
                    if state is not None:
                        state[timesteps_gt_horizon] = state_gt_horizon
                        
                return action, state

        def update_horizon(self) -> None:
            """
            Update the horizon based on the current strategy.
            """
            if self.strategy == "curriculum":
                self.horizon_step += 1
                self.horizon_step = min(self.horizon_step, len(self.horizons) - 1)
            elif self.strategy == "random":
                self.horizon_step = np.random.randint(len(self.horizons))
    return JSRLPolicy


def get_jsrl_algorithm(Algorithm: BaseAlgorithm):
    class JSRLAlgorithm(Algorithm):
        def __init__(self, policy, eval_env, learning_starts, epsilon, data_collection_strategy, log_true_q, *args, **kwargs):
            if isinstance(policy, str):
                policy = self._get_policy_from_name(policy)
            else:
                policy = policy
            policy = get_jsrl_policy(policy)
            kwargs["learning_starts"] = learning_starts
            super().__init__(policy, *args, **kwargs)
            self._timesteps = np.zeros((self.env.num_envs), dtype=np.int32)
            self.eval_env = eval_env
            self.epsilon = epsilon
            assert data_collection_strategy in ["noisy", "normal", "delay", "multi"], f"data_collection_strategy: '{data_collection_strategy}' must be 'noisy' or 'normal' or 'delay' or 'multi'"
            self.data_collection_strategy = data_collection_strategy
            self.log_true_q = log_true_q
            self.state_action_buffer = JSRLStatesActionsBuffer(100000, self.policy.action_space.shape[0], self.policy.observation_space.shape[0])


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
                self.eval_env,
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
                    
                    noise_chance = np.random.random()

                    # Select action randomly or according to policy
                    if noise_chance < self.epsilon and self.data_collection_strategy == "noisy":
                        # print("random")
                        actions = np.array([self.action_space.sample() for _ in range(env.num_envs)])
                        buffer_actions = actions
                    else:
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


    return JSRLAlgorithm