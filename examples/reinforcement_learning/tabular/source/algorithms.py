import numpy as np

class episodic_algorithm(object):
	"""A wrapper for episodic RL algorithms."""

	def __init__(self, parameters):
		self.environment = parameters['environment']
		self.average_return = 0
		self.average_returns = []
		self.returns = []
		self.return_learning_rate = parameters['return_learning_rate']
		self.policy = parameters['policy']
		self.episode = 0
		self.current_state = self.environment.reset()
		self.past_state = self.current_state.copy()
		self.action = 0
		self.reward = 0
		self.current_return = 0
		self.end = False
		self.info = None
	
	def _transition(self):
		"""Requests an action from the policy and sends it to the environment."""
		self.past_state = self.current_state.copy()
		self.action, self.eligibility = self.policy.action(self.current_state)
		self.current_state, self.reward, self.end, self.info = self.environment.step(
			self.action)
		self.current_return += self.reward

	def _per_step(self):
		"""A placeholder for a learning algorithms computations per transition."""
		self._transition()

	def _per_episode(self):
		"""A placeholder for a learning algorithms computations after episodes."""
		self.current_state = self.environment.reset()
		self.past_state = self.current_state.copy()
		self.end = False

	def _episode(self):
		"""Uses _per_step and _per_episode to run a generic episodes computations."""
		self.current_return = 0
		while not self.end:
			self._per_step()
		self._per_episode()
		self.average_return += self.return_learning_rate * (self.current_return 
															- self.average_return)
		self.episode += 1

	def train(self, episodes):
		"""Trains the policy by repeatedly running episodes, storing return info."""
		self.episode = 0
		while self.episode < episodes:
			self._episode()
			self.average_returns.append(self.average_return)
			self.returns.append(self.current_return)

	def _sample(self):
		"""Generates a sample trajectory using the current policy."""
		self.current_return = 0
		trajectory = [self.current_state.copy()]
		while not self.end:
			self._transition()
			trajectory.append(self.current_state.copy())
		self.current_state = self.environment.reset()
		self.past_state = self.current_state.copy()
		self.end = False
		return trajectory

	def samples(self, sample_count):
		"""Generates a set of trajectory samples."""
		trajectories = []
		sample = 0
		while sample < sample_count:
			trajectory = self._sample()
			trajectories.append(trajectory)
			sample += 1
		return trajectories

	def _return_sample(self):
		"""Runs an episode to sample a return for evaulation."""
		self.current_return = 0
		while not self.end:
			self._transition()
		self.current_state = self.environment.reset()
		self.past_state = self.current_state.copy()
		self.end = False

	def evaluate(self, sample_count, set_average = True):
		"""Evaluates the policy by estimating the average return."""
		sample = 1
		average_return = 0
		while sample <= sample_count:
			self._return_sample()
			average_return += (self.current_return - average_return)/sample
			sample += 1
		if set_average:
			self.average_return = average_return
		return average_return


class monte_carlo_returns(episodic_algorithm):
	"""A purely return based policy gradient algorithm."""

	def __init__(self, parameters):
		super().__init__(parameters)
		self.states = []
		self.rewards = []
		self.eligibilities = []

	def _per_step(self):
		"""Adds required data storage for learning post-episode."""
		self._transition()
		self.states.append(self.past_state)
		self.rewards.append(self.reward)
		self.eligibilities.append(self.eligibility)

	def _update(self):
		"""Loops over the episode in reverse, updating the policy in each state."""
		self.rewards = np.array(self.rewards)
		state_return = 0
		for index in range(len(self.states) - 1, -1, -1):
			state_return += self.rewards[index]
			self.policy.step(self.states[index], state_return, self.eligibilities[index])

	def _per_episode(self):
		"""Adds additional resets relevant to learning algorithm."""
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
		"""Adds required data storage for learning post-episode."""
		super()._per_step()
		self.state_values.append(self.values.forward(self.past_state))

	def _update(self):
		"""Loops over the episode in reverse, updating state policies and values."""
		self.rewards = np.array(self.rewards)
		state_return = 0
		for index in range(len(self.states) - 1, -1, -1):
			state_return += self.rewards[index]
			error = state_return - self.state_values[index]
			self.policy.step(self.states[index], error, self.eligibilities[index])
			self.values.step(self.states[index], error)

	def _per_episode(self):
		"""Adds additional resets relevant to learning algorithm."""
		super()._per_episode()
		self.state_values = []


class actor_critic(episodic_algorithm):
	"""Uses the value as a baseline and an estimate of future returns."""

	def __init__(self, parameters):
		super().__init__(parameters)
		self.values = parameters['values']

	def _update(self):
		"""Updates the policy and value for the previous state."""
		past_value = self.values.forward(self.past_state)
		if not self.end:
			current_value = self.values.forward(self.current_state)
		else:
			current_value = 0
		td_error = current_value + self.reward - past_value
		self.values.step(self.past_state, td_error)
		self.policy.step(self.past_state, td_error, self.eligibility)

	def _per_step(self):
		"""Overrides the _per_step method, to transition and update each step."""
		self._transition()
		self._update()