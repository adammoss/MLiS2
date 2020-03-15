import sys
import os
from matplotlib import pyplot
source_path = os.path.join("source/")
sys.path.insert(0,source_path)
import environments # pylint: disable = import-error
import algorithms # pylint: disable = import-error
import tables # pylint: disable = import-error

environment_parameters = dict(
	trajectory_length = 20, 
	positivity_bias = 1,
	target_bias = 2,
)
environment = environments.excursions(environment_parameters)

table_dimension = (environment_parameters['trajectory_length']*2 + 1, 
				   environment_parameters['trajectory_length'] + 1)
policy1 = tables.two_action_policy_table(table_dimension, 0.15)
values2 = tables.value_table(table_dimension, 0.6)
policy2 = tables.two_action_policy_table(table_dimension, 0.15)
values3 = tables.value_table(table_dimension, 0.6)
policy3 = tables.two_action_policy_table(table_dimension, 0.15)

algorithm_parameters1 = dict(
	environment = environment, 
	return_learning_rate = 0.1,
	policy = policy1,
)
algorithm_parameters2 = dict(
	environment = environment, 
	return_learning_rate = 0.1,
	values = values2,
	policy = policy2,
)
algorithm_parameters3 = dict(
	environment = environment, 
	return_learning_rate = 0.1,
	values = values3,
	policy = policy3,
)
agent1 = algorithms.monte_carlo_returns(algorithm_parameters1)
agent2 = algorithms.monte_carlo_value_baseline(algorithm_parameters2)
agent3 = algorithms.actor_critic(algorithm_parameters3)

initial_return = agent1.evaluate(1000)
agent2.average_return = initial_return
agent3.average_return = initial_return
print(initial_return)
episodes = 1000
agent1.train(episodes)
agent2.train(episodes)
agent3.train(episodes)

pyplot.figure()
pyplot.subplot(121)
pyplot.plot(agent1.returns, c = 'r')
pyplot.plot(agent2.returns, c = 'b')
pyplot.plot(agent3.returns, c = 'k')
pyplot.subplot(122)
pyplot.plot(agent1.average_returns, c = 'r')
pyplot.plot(agent2.average_returns, c = 'b')
pyplot.plot(agent3.average_returns, c = 'k')
pyplot.xscale('log')
pyplot.show()