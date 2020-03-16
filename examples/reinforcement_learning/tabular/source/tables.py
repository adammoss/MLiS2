import math
import numpy as np
import numpy.random

class value_table(object):
	"""A generic tabular state value function."""

	def __init__(self, dimensions, learning_rate):
		self.table = np.zeros(dimensions)
		self.learning_rate = learning_rate

	def forward(self, state):
		"""Returns the value of the specified state."""
		return self.table[state[0]][state[1]]

	def step(self, state, error):
		"""Updates the value of the specified state."""
		self.table[state[0]][state[1]] += self.learning_rate * error

class two_action_policy_table(object):
	"""A tabular policy for environments where each state has two actions."""

	def __init__(self, dimensions, learning_rate):
		self.table = np.zeros(dimensions)
		self.learning_rate = learning_rate

	def _forward(self, state):
		"""Calculates the probabilitiy of action 1."""
		exponentiated_potential = math.exp(-self.table[state[0]][state[1]])
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
		self.table[state[0]][state[1]] += self.learning_rate * error * eligibility