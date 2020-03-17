import math
import numpy as np
import numpy.random


class identity_preprocessor(object):
	def process(self, state):
		return tuple(state)


class value_table(object):
	"""A generic tabular state value function."""

	def __init__(self, dimensions, learning_rate, preprocessor = None):
		self.table = np.zeros(dimensions)
		self.learning_rate = learning_rate
		if preprocessor == None:
			self.preprocessor = identity_preprocessor()
		else:
			self.preprocessor = preprocessor

	def forward(self, state):
		"""Returns the value of the specified state."""
		state_index = self.preprocessor.process(state)
		return self.table[state_index]

	def step(self, state, error):
		"""Updates the value of the specified state."""
		state_index = self.preprocessor.process(state)
		self.table[state_index] += self.learning_rate * error


class two_action_policy_table(object):
	"""A tabular policy for environments where each state has two actions."""

	def __init__(self, dimensions, learning_rate, preprocessor = None):
		self.table = np.zeros(dimensions)
		self.learning_rate = learning_rate
		if preprocessor == None:
			self.preprocessor = identity_preprocessor()
		else:
			self.preprocessor = preprocessor

	def _forward(self, state):
		"""Calculates the probabilitiy of action 1."""
		state_index = self.preprocessor.process(state)
		exponentiated_potential = math.exp(-self.table[state_index])
		return 1/(exponentiated_potential+1)

	def action(self, state):
		"""Returns a random action according to the current policy."""
		action1_probability = self._forward(state)
		random = numpy.random.random() # pylint: disable = no-member
		if random < action1_probability:
			return 1, 1 - action1_probability
		else:
			return 0, -action1_probability
	
	def step(self, state, error, eligibility):
		"""Updates the potential for actions in the given state."""
		state_index = self.preprocessor.process(state)
		self.table[state_index] += self.learning_rate * error * eligibility


class cartpole_preprocessor(object):

	def __init__(self, dimensions, range_1, range_2):
		self.dimensions = dimensions
		self.limits = [2.4, range_1, 0.21, range_2]

	def table_index(self, state, component_index):
		shift = state[component_index] + self.limits[component_index]
		scale = shift / (2 * self.limits[component_index])
		return math.floor(self.dimensions[component_index] * scale)

	def process(self, state):
		processed_state = [0, 0, 0, 0]
		processed_state[0] = self.table_index(state, 0)
		if state[1] < -self.limits[1]:
			processed_state[1] = 0
		elif state[1] > self.limits[1]:
			processed_state[1] = self.dimensions[1]
		else:
			processed_state[1] = self.table_index(state, 1)
		processed_state[2] = self.table_index(state, 2)
		if state[3] < -self.limits[3]:
			processed_state[3] = 0
		elif state[3] > self.limits[3]:
			processed_state[3] = self.dimensions[3]
		else:
			processed_state[3] = self.table_index(state, 3)
		return tuple(processed_state)