import sys
import os
from matplotlib import pyplot as plt
source_path = os.path.join("source/")
sys.path.insert(0,source_path)
import environments # pylint: disable = import-error
import algorithms # pylint: disable = import-error
import tables # pylint: disable = import-error
import gym # pylint: disable = import-error


env = gym.make('CartPole-v0')

dimensions = (10,10,10,10)
preprocessor = tables.cartpole_preprocessor(dimensions, 5, 5)
policy1 = tables.two_action_policy_table(dimensions, 0.05, preprocessor)
values2 = tables.value_table(dimensions, 0.2, preprocessor)
policy2 = tables.two_action_policy_table(dimensions, 0.05, preprocessor)
values3 = tables.value_table(dimensions, 0.2, preprocessor)
policy3 = tables.two_action_policy_table(dimensions, 0.2, preprocessor)
algorithm_parameters1 = dict(
	environment = env, 
	return_learning_rate = 0.1,
	policy = policy1,
)
algorithm_parameters2 = dict(
	environment = env, 
	return_learning_rate = 0.1,
    values = values2,
	policy = policy2,
)
algorithm_parameters3 = dict(
	environment = env, 
	return_learning_rate = 0.1,
    values = values3,
	policy = policy3,
)
agent1 = algorithms.monte_carlo_returns(algorithm_parameters1)
agent2 = algorithms.monte_carlo_value_baseline(algorithm_parameters2)
agent3 = algorithms.actor_critic(algorithm_parameters3)

observation = env.reset()
for i in range(5):
    done = False
    while not done:
        env.render()
        action = policy3.action(observation)[0]
        observation, reward, done, info = env.step(action)
    observation = env.reset()
env.close()

evaluations = 1000
initial_return = agent1.evaluate(evaluations)
agent2.average_return = initial_return
agent3.average_return = initial_return
print("Initial return: %s"%(initial_return))
episodes = 1000
agent1.train(episodes)
agent2.train(episodes)
agent3.train(episodes)
final_return1 = agent1.evaluate(evaluations)
final_return2 = agent2.evaluate(evaluations)
final_return3 = agent3.evaluate(evaluations)
print("Initial return: %s, agent1's final return: %s, agent2's final return: %s, agent3's final return: %s"
%(initial_return, final_return1, final_return2, final_return3))

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

observation = env.reset()
for i in range(1):
    done = False
    while not done:
        env.render()
        action = policy3.action(observation)[0]
        observation, reward, done, info = env.step(action)
    observation = env.reset()
env.close()