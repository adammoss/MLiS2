class excursions(object):
	"""A simple environment which provides rewards based on excursions."""

	def __init__(self, parameters):
		self.trajectory_length = parameters['trajectory_length']
		self.positivity_bias = parameters['positivity_bias']
		self.target_bias = parameters['target_bias']
		self.action = 0
		self.state = [0, 0]
		self.terminal_state = False

	def _reward(self):
		"""Calculates the reward for the last transition to occur."""
		if self.state[0] < 0:
			reward = -self.positivity_bias * abs(self.state[0])
		else:
			reward = 0
		if self.state[1] == self.trajectory_length:
			reward -= self.target_bias * abs(self.state[0])
		return reward

	def step(self, action):
		"""Updates the environment state based on the input action."""
		self.action = action
		self.state[0] += 2*action - 1
		self.state[1] += 1
		if self.state[1] == self.trajectory_length:
			self.terminal_state = True
		return self.state, self._reward(), self.terminal_state, None

	def reset(self):
		"""Resets the environment state and terminal boolean."""
		self.action = 0
		self.state = [0, 0]
		self.terminal_state = False
		return self.state