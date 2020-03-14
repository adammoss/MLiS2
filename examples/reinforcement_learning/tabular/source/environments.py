class excursions(object):

	def __init__(self, parameters):
		self.trajectory_length = parameters['trajectory_length']
		self.positivity_bias = parameters['positivity_bias']
		self.target_bias = parameters['target_bias']
		self.action = 0
		self.state = [0, 0]
		self.terminal_state = False

	def _reward(self):
		if self.state[0] < 0:
			reward = -self.positivity_bias * abs(self.state[0])
		else:
			reward = 0
		if self.state[1] == self.trajectory_length:
			reward -= self.target_bias * abs(self.state[0])
		return reward

	def transition(self, action):
		self.action = action
		self.state[0] += 2*action - 1
		self.state[1] += 1
		if self.state[1] == self.trajectory_length:
			self.terminal_state = True
		return self.state, self._reward()

	def reset(self):
		self.action = 0
		self.state = [0, 0]
		self.terminal_state = False