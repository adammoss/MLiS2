import sys
import os
import numpy as np
from matplotlib import pyplot as plt
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
print("Initial return: %s"%(initial_return))
initial_samples = agent1.samples(30)

min_y = np.min(np.array(initial_samples)[:,:,0]) - 1
max_y = np.max(np.array(initial_samples)[:,:,0]) + 1

episodes = 1000
agent1.train(episodes)
agent2.train(episodes)
agent3.train(episodes)

plt.figure(figsize = (10, 3.5))
plt.subplot(121)
plt.plot(agent3.returns, c = 'g')
plt.plot(agent1.returns, c = 'b')
plt.plot(agent2.returns, c = 'm')
plt.xlabel("Episode")
plt.ylabel("Episodic returns")
plt.subplot(122)
plt.plot(agent3.average_returns, c = 'g')
plt.plot(agent1.average_returns, c = 'b')
plt.plot(agent2.average_returns, c = 'm')
plt.xlabel("Episode")
plt.ylabel("Running return")
plt.show()

final_return1 = agent1.evaluate(1000)
final_return2 = agent2.evaluate(1000)
final_return3 = agent3.evaluate(1000)
print("Initial return: %s, agent1's final return: %s, agent2's final return: %s, agent3's final return: %s"
%(initial_return, final_return1, final_return2, final_return3))
samples1 = agent1.samples(30)
samples2 = agent2.samples(30)
samples3 = agent3.samples(30)

plt.figure(figsize = (12, 3.5))
plt.subplot(131)
plt.plot(np.array(initial_samples)[:,:,0].T, c = 'k', alpha = 0.2)
plt.plot(np.array(samples1)[:,:,0].T, c = 'b', alpha = 0.2)
plt.scatter([20], [0], c = 'k', marker = 'o', s = 80)
plt.plot([-1, 21], [0, 0], lw = 2, c = 'r', ls = '--', alpha = 0.3)
plt.fill_between([-1, 21], [0, 0], [min_y, min_y], color = 'r', alpha = 0.1)
plt.xlim(-1, 21)
plt.ylim(min_y, max_y)
plt.title("Agent 1")
plt.xlabel("Time")
plt.ylabel("Position")
plt.subplot(132)
plt.plot(np.array(initial_samples)[:,:,0].T, c = 'k', alpha = 0.2)
plt.plot(np.array(samples2)[:,:,0].T, c = 'm', alpha = 0.2)
plt.scatter([20], [0], c = 'k', marker = 'o', s = 80)
plt.plot([-1, 21], [0, 0], lw = 2, c = 'r', ls = '--', alpha = 0.3)
plt.fill_between([-1, 21], [0, 0], [min_y, min_y], color = 'r', alpha = 0.1)
plt.xlim(-1, 21)
plt.ylim(min_y, max_y)
plt.title("Agent 2")
plt.xlabel("Time")
plt.subplot(133)
plt.plot(np.array(initial_samples)[:,:,0].T, c = 'k', alpha = 0.2)
plt.plot(np.array(samples3)[:,:,0].T, c = 'g', alpha = 0.2)
plt.scatter([20], [0], c = 'k', marker = 'o', s = 80)
plt.plot([-1, 21], [0, 0], lw = 2, c = 'r', ls = '--', alpha = 0.3)
plt.fill_between([-1, 21], [0, 0], [min_y, min_y], color = 'r', alpha = 0.1)
plt.xlim(-1, 21)
plt.ylim(min_y, max_y)
plt.title("Agent 3")
plt.xlabel("Time")

plt.show()