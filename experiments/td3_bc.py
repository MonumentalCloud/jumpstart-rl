class TD3_BC(TD3):
    def __init__(
        self,
        policy: str,
        env: GymEnv,
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        bc_coef: float = 0.5,  # BC coefficient for the policy objective
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            optimize_memory_usage,
            policy_kwargs,
        )
        self.bc_coef = bc_coef

    def train(self) -> None:
        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.observation_space, self.action_space)

        # Set up the learning rate schedule
        self.lr_schedule = self._get_schedule(self.learning_rate, self.lr_schedule)

        # Create the optimizer
        self.optimizer = self._create_optimizer()

        # Initialize the target networks
        self._create_aliases()
        self._create_target_network()

        # Initialize the variables
        self._create_buffer()

        # Set up the exploration strategy
        self.exploration_noise = self._create_exploration_noise()

        # Initialize the episode reward and step count
        episode_reward = 0.0
        episode_steps = 0

        # Initialize the total timesteps
        self._total_timesteps = self._episode_num * self.env.max_episode_steps

        # Start the training loop
        while self._total_timesteps < self.total_timesteps:
            # Perform a rollout
            episode_reward, episode_steps = self._rollout(episode_reward, episode_steps)

            # Update the target networks
            if self._total_timesteps % self.target_update_interval == 0:
                polyak_update(self.q_net_target.parameters(), self.q_net.parameters(), self.tau)

            # Train the policy and Q networks
            self._train()

            # Log the training progress
            self._log_training_progress()

    def _train(self) -> None:
        for gradient_step in range(self.gradient_steps):
            # Sample a batch of transitions from the replay buffer
            batch = self.replay_buffer.sample(self.batch_size)

            # Unpack the batch
            obs, actions, rewards, next_obs, dones = batch

            # Compute the target Q values
            with torch.no_grad():
                next_actions = self.policy_target(next_obs)
                next_q_values = self.q_net_target(next_obs, next_actions)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute the current Q values
            current_q_values = self.q_net(obs, actions)

            # Compute the BC loss
            bc_loss = self.policy.compute_bc_loss(obs, actions)

            # Compute the Q loss
            q_loss = F.mse_loss(current_q_values, target_q_values)

            # Compute the total loss
            actor_loss = q_loss - self.bc_coef * bc_loss

            # Optimize the Q network
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()

            # Update the target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.q_net_target.parameters(), self.q_net.parameters(), self.tau)
    
    def compute_bc_loss(self, obs, actions):
        # Behavior cloning loss
        return F.mse_loss(self.policy(obs), actions)