import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gym

env = gym.make('Taxi-v3')
print("观测空间 = {}".format(env.observation_space))
print("动作空间 = {}".format(env.action_space))
print("状态数量 = {}".format(env.observation_space.n))
print("动作数量 = {}".format(env.action_space.n))

state, _ = env.reset()
taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
print(taxirow, taxicol, passloc, destidx)
print("的士位置 = {}".format(taxirow, taxicol))
print("乘客位置 = {}".format(env.unwrapped.locs[passloc]))
print("目的位置 = {}".format(env.unwrapped.locs[destidx]))
# env.render()
print(env.step(0))

# env.render()

# SARSA算法
print("SARSA算法")

class SARSAAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.2, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state): # 采用epsilon贪心策略，防止智能体陷入局部最优
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, terminated, truncated, next_action):
        u = reward + self.gamma * \
            self.q[next_state, next_action] * (1. - terminated)
        td_error = u - self.q[state, action]
        self.q[state,action] += self.learning_rate * td_error

# Agent 与环境进行交互一回合
def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation, _ = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation) # 终止状态此步没有意义
        if train:
            agent.learn(observation, action, reward, next_observation,
                        terminated, truncated, next_action)
        if terminated or truncated:
            break
        observation, action = next_observation, next_action
    return episode_reward

agent = SARSAAgent(env)

# 训练
episodes = 3000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)

# plt.plot(episode_rewards)
# plt.show()

# 测试
agent.epsilon = 0. # 取消探索

episode_rewards = [play_sarsa(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                           len(episode_rewards), np.mean(episode_rewards)))
# 显示最优价值估计
print(pd.DataFrame(agent.q))

# 显示最优策略估计
policy = np.eye(agent.action_n)[agent.q.argmax(axis=-1)]
print(pd.DataFrame(policy))

# 期望SARSA   先计算状态价值函数，然后再用状态价值函数来计算U回报
print("期望SARSA")
class ExpectedSARSAAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        self.action_n = env.action_space.n

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, terminated, truncated):
        v = (self.q[next_state].mean() * self.epsilon + \
               self.q[next_state].max() * (1. - self.epsilon))  # 采用epsilon策略，最大的动作价值函数的概率是（1-epsilon），剩余的动作概率均分，可以算出状态价值函数的值
        u = reward + self.gamma * v * (1. - terminated)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error

def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation, _ = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation,
                        terminated, truncated)
        if terminated or truncated:
            break
        observation = next_observation
    return episode_reward

agent = ExpectedSARSAAgent(env)

# 训练
episodes = 5000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)

# plt.plot(episode_rewards)
# plt.show()

# 测试
agent.epsilon = 0. # 取消探索

episode_rewards = [play_sarsa(env, agent) for _ in range(100)]
print("平均回合奖励 = {} / {} = {}".format(sum(episode_rewards), len(episode_rewards),
                                           np.mean(episode_rewards)))

# Q学习
print("Q学习")
class QLearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        self.action_n = env.action_space.n

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, ternimated, truncated):
        u = reward + self.gamma * self.q[next_state].max() * (1.0 - ternimated)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error

agent = QLearningAgent(env)

# 训练
episodes = 4000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)

# plt.plot(episode_rewards)
# plt.show()

# 测试
agent.epsilon = 0. # 取消探索

episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

# 双重Q学习
print("双重Q学习")
class DoubleQLearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q0 = np.zeros((env.observation_space.n, env.action_space.n))
        self.q1 = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = (self.q0 + self.q1)[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, terminated, truncated):
        if np.random.randint(2):
            self.q0, self.q1 = self.q1, self.q0
        a = self.q0[state, action].argmax()
        u = reward + self.gamma * self.q1[next_state, a] * (1. - terminated)
        td_error = u - self.q0[state, action]
        self.q0[state, action] += self.learning_rate * td_error

class DoubleQLearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q0 = np.zeros((env.observation_space.n, env.action_space.n))
        self.q1 = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = (self.q0 + self.q1)[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, terminated, truncated):
        if np.random.randint(2):
            self.q0, self.q1 = self.q1, self.q0
        a = self.q0[next_state].argmax()
        u = reward + self.gamma * self.q1[next_state, a] * (1. - terminated)
        td_error = u - self.q0[state, action]
        self.q0[state, action] += self.learning_rate * td_error

agent = DoubleQLearningAgent(env)

# 训练
episodes = 9000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)

# plt.plot(episode_rewards)
# plt.show()

# 测试
agent.epsilon = 0. # 取消探索

episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

# SARSA(lambda)算法
print("SARSA(lambda)算法")
class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env, lambd=0.6, beta=1.,
            gamma=0.9, learning_rate=0.1, epsilon=.01):
        super().__init__(env, gamma=gamma, learning_rate=learning_rate,
                epsilon=epsilon)
        self.lambd = lambd
        self.beta = beta
        self.e = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, terminated, truncated, next_action):
        # 更新资格迹
        self.e *= (self.lambd * self.gamma)
        self.e[state, action] = 1. + self.beta * self.e[state, action]

        # 更新价值
        u = reward + self.gamma * \
                self.q[next_state, next_action] * (1. - terminated)
        td_error = u - self.q[state, action]
        self.q += self.learning_rate * self.e * td_error
        if terminated or truncated:
            self.e *= 0.

agent = SARSALambdaAgent(env)

# 训练
episodes = 5000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)

# plt.plot(episode_rewards)
# plt.show()

# 测试
agent.epsilon = 0. # 取消探索

episode_rewards = [play_sarsa(env, agent, train=False) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))