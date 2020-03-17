import sys
import os
from matplotlib import pyplot
source_path = os.path.join("source/")
sys.path.insert(0,source_path)
import environments # pylint: disable = import-error
import algorithms # pylint: disable = import-error
import tables # pylint: disable = import-error
import gym # pylint: disable = import-error


env = gym.make('CartPole-v0')

dimensions = (10,10,10,10)
preprocessor = tables.cartpole_preprocessor(dimensions, 5, 5)
values = tables.value_table(dimensions, 0.2, preprocessor = preprocessor)
policy = tables.two_action_policy_table(dimensions, 0.05, preprocessor = preprocessor)
algorithm_parameters = dict(
	environment = env, 
	return_learning_rate = 0.1,
	values = values,
	policy = policy,
)
agent = algorithms.actor_critic(algorithm_parameters)

evaluations = 1000
initial_return = agent.evaluate(evaluations)
print(initial_return)
episodes = 5000
agent.train(episodes)
final_return = agent.evaluate(evaluations)
print(final_return)

pyplot.figure()
pyplot.subplot(121)
pyplot.plot(agent.returns, c = 'r')
pyplot.subplot(122)
pyplot.plot(agent.average_returns, c = 'r')
pyplot.show()

tests = 1
for i in range(tests):
    observation = env.reset()
    done = False
    while not done:
        env.render()
        action = policy.action(observation)[0]
        observation, reward, done, info = env.step(action)
env.close()