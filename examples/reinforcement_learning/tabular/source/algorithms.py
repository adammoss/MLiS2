import numpy as np

class episodic_algorithm(object):
	"""A wrapper for episodic policy gradient RL algorithms."""

	def __init__(self, parameters):
		self.environment = parameters['environment']
		self.average_return = 0
		self.average_returns = []
		self.returns = []
		self.return_learning_rate = parameters['return_learning_rate']
		self.policy = parameters['policy']
		self.episode = 0
		self.past_state = self.environment.state.copy()
		self.action = 0
		self.current_state = self.environment.state.copy()
		self.reward = 0
		self.current_return = 0
	
	def _transition(self):
		self.past_state = self.current_state.copy()
		self.action, self.eligibility = self.policy.action(self.current_state)
		self.current_state, self.reward = self.environment.transition(self.action)
		self.current_return += self.reward

	def _per_step(self):
		self._transition()

	def _per_episode(self):
		self.environment.reset()
		self.past_state = self.environment.state.copy()
		self.current_state = self.environment.state.copy()

	def _episode(self):
		self.current_return = 0
		while not self.environment.terminal_state:
			self._per_step()
		self._per_episode()
		self.average_return += self.return_learning_rate * (self.current_return 
															- self.average_return)
		self.episode += 1

	def train(self, episodes):
		self.episode = 0
		while self.episode < episodes:
			self._episode()
			self.average_returns.append(self.average_return)
			self.returns.append(self.current_return)

	def _sample(self):
		trajectory = []
		while not self.environment.terminal_state:
			self._transition()
			trajectory.append(self.current_state)
		self.environment.reset()
		self.past_state = self.environment.state.copy()
		self.current_state = self.environment.state.copy()
		return trajectory

	def samples(self, sample_count):
		trajectories = []
		sample = 0
		while sample < sample_count:
			trajectory = self._sample()
			trajectories.append(trajectory)
			sample += 1
		return trajectories


class monte_carlo_returns(episodic_algorithm):
	"""A purely return based policy gradient algorithm."""

	def __init__(self, parameters):
		super().__init__(parameters)
		self.states = []
		self.rewards = []
		self.eligibilities = []

	def _per_step(self):
		self._transition()
		self.states.append(self.past_state)
		self.rewards.append(self.reward)
		self.eligibilities.append(self.eligibility)

	def _update(self):
		self.rewards = np.array(self.rewards)
		for index in range(len(self.states)):
			state_return = np.sum(self.rewards[index:])
			self.policy.step(self.states[index], state_return, self.eligibilities[index])

	def _per_episode(self):
		self._update()
		super()._per_episode()
		self.states = []
		self.rewards = []
		self.eligibilities = []


class monte_carlo_value_baseline(monte_carlo_returns):
	"""Contrasts returns with estimated values for policy updates."""

	def __init__(self, parameters):
		super().__init__(parameters)
		self.values = parameters['values']
		self.state_values = []

	def _per_step(self):
		super()._per_step()
		self.state_values.append(self.values.forward(self.past_state))

	def _update(self):
		self.rewards = np.array(self.rewards)
		for index in range(len(self.states)):
			error = np.sum(self.rewards[index:]) - self.state_values[index]
			self.policy.step(self.states[index], error, self.eligibilities[index])
			self.values.step(self.states[index], error)

	def _per_episode(self):
		super()._per_episode()
		self.state_values = []


class actor_critic(episodic_algorithm):
	"""Uses the value as a baseline and an estimate of future returns."""

	def __init__(self, parameters):
		super().__init__(parameters)
		self.values = parameters['values']

	def _update(self):
		past_value = self.values.forward(self.past_state)
		current_value = self.values.forward(self.current_state)
		td_error = current_value + self.reward - past_value
		self.values.step(self.past_state, td_error)
		self.policy.step(self.past_state, td_error, self.eligibility)

	def _per_step(self):
		self._transition()
		self._update()