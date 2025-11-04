import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v2 as tf
tf.random.set_seed(0)
from tensorflow import keras

class Chart:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)

    def plot(self, episode_rewards):
        self.ax.clear()
        self.ax.plot(episode_rewards)
        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('episode reward')
        self.fig.canvas.draw()


env = gym.make('CartPole-v0')

# 用简单策略地图算法寻找最优策略
class VPGAgent:
    def __init__(self, env, policy_kwargs, baseline_kwargs=None,
                 gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma

        self.trajectory = []

        self.policy_net = self.build_network(output_size=self.action_n,
                                             output_activation=tf.nn.softmax,
                                             loss=tf.losses.categorical_crossentropy,
                                             **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_network(**baseline_kwargs)

    def build_network(self, hidden_sizes, output_size=1,
                      activation=tf.nn.relu, output_activation=None,
            use_bias=False, loss=tf.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for hidden_size in hidden_sizes:
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, use_bias=use_bias))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation, use_bias=use_bias))
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        # 网络返回的是 动作的概率分布
        probs = self.policy_net.predict(observation[np.newaxis], verbose=0)[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, terminated, truncated):
        self.trajectory.append((observation, action, reward))

        if terminated or truncated:
            df = pd.DataFrame(self.trajectory,
                              columns=['observation', 'action', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series() # df.index.to_series()获取第几步
            df['discounted_reward'] = df['discount'] * df['reward'] # 折扣奖励
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()   # 倒序操作，累计求和
            df['psi'] = df['discounted_return'] # 样本权重

            x = np.stack(df['observation'])
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x, verbose=0)
                df['psi'] -= (df['baseline'] * df['discount']) # 用基线来进行修正
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)

            sample_weight = df['psi'].values[:, np.newaxis]
            y = np.eye(self.action_n)[df['action']]  # 离散动作变成独热编码
            self.policy_net.fit(x, y, sample_weight=sample_weight, verbose=0)

            self.trajectory = [] # 下一回合初始化经验列表

def play_montecarlo(env, agent, render=False, train=False):
    observation, _ = env.reset()
    episode_reward = 0.
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, terminated, truncated)
        if terminated or truncated:
            break
        observation = next_observation
    return episode_reward

# 不带基线的简单策略梯度算法
print('简单策略梯度算法')
policy_kwargs = {'hidden_sizes': [], 'learning_rate': 0.005}
agent = VPGAgent(env, policy_kwargs=policy_kwargs)

# 训练
episodes = 1000
episode_rewards = []
chart = Chart()
for episode in range(episodes):
    episode_reward = play_montecarlo(env, agent, train=True)
    episode_rewards.append(episode_reward)
    chart.plot(episode_rewards)

plt.show()

# 测试
episode_rewards = [play_montecarlo(env, agent, train=False)
        for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

# 带基线的简单策略梯度算法
print('简单策略梯度算法（带基线）')
policy_kwargs = {'hidden_sizes': [], 'learning_rate': 0.005}
baseline_kwargs = {'hidden_sizes': [], 'learning_rate': 0.01}
agent = VPGAgent(env, policy_kwargs=policy_kwargs,
                 baseline_kwargs=baseline_kwargs)

# 训练
episodes = 1000
episode_rewards = []
chart = Chart()
for episode in range(episodes):
    episode_reward = play_montecarlo(env, agent, train=True)
    episode_rewards.append(episode_reward)
    chart.plot(episode_rewards)

plt.show()

# 测试
episode_rewards = [play_montecarlo(env, agent, train=False) \
                   for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

# 异策策略梯度算法
print("异策策略梯度算法")
class RandomAgent:
    def __init__(self):
        self.action.n = env.action_space.n

    def decide(self, observation):
        action = np.random.choice(self.action_n)
        behavior = 1. / self.acton_n
        return action, behavior

class OffPolicyVPGAgent(VPGAgent):
    def __init__(self, env, policy_kwargs, baseline_kwargs=None,
            gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma

        self.trajectory = []

        def dot(y_true, y_pred): # 点积损失函数
            return -tf.reduce_sum(y_true * y_pred, axis=-1)

        self.policy_net = self.build_network(output_size=self.action_n,
                output_activation=tf.nn.softmax, loss=dot, **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_network(output_size=1,
                    **baseline_kwargs)

    def learn(self, observation, action, behavior, reward, terminated, truncated):
        self.trajectory.append((observation, action, behavior, reward))

        if terminated or truncated:
            df = pd.DataFrame(self.trajectory, columns=
                    ['observation', 'action', 'behavior', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = \
                    df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            x = np.stack(df['observation'])
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x, verbose=0)
                df['psi'] -= df['baseline'] * df['discount']
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)

            sample_weight = (df['psi'] / df['behavior']).values[:, np.newaxis] # 分子的Π(a_t | s_t) 已经在损失函数当中了
            y = np.eye(self.action_n)[df['action']]
            self.policy_net.fit(x, y, sample_weight=sample_weight, verbose=0) # sample_weights是加权到损失函数上

            self.trajectory = [] # 为下一回合初始化经验列表

# 不带基线的重要性采样策略梯度算法
policy_kwargs = {'hidden_sizes' : [], 'learning_rate' : 0.06}
agent = OffPolicyVPGAgent(env, policy_kwargs=policy_kwargs)
behavior_agent = RandomAgent(env)

# 训练
episodes = 1000
episode_rewards = []
chart = Chart()
for episode in range(episodes):
    observation, _ = env.reset()
    episode_reward = 0.
    while True:
        action, behavior = behavior_agent.decide(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        agent.learn(observation, action, behavior, reward, terminated, truncated)
        if terminated or truncated:
            break
        observation = next_observation

    # 跟踪监控
    episode_reward = play_montecarlo(env, agent, train=False)
    episode_rewards.append(episode_reward)
    chart.plot(episode_rewards)

plt.show()

# 测试
episode_rewards = [play_montecarlo(env, agent, train=False)
        for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

# 带基线的重要性采样策略梯度算法
policy_kwargs = {'hidden_sizes' : [], 'learning_rate' : 0.06}
baseline_kwargs = {'hidden_sizes' : [], 'learning_rate' : 0.1}
agent = OffPolicyVPGAgent(env, policy_kwargs=policy_kwargs,
        baseline_kwargs=baseline_kwargs)
behavior_agent = RandomAgent(env)

# 训练
episodes = 1000
episode_rewards = []
chart = Chart()
for episode in range(episodes):
    observation, _ = env.reset()
    episode_reward = 0.
    while True:
        action, behavior = behavior_agent.decide(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        agent.learn(observation, action, behavior, reward, terminated, truncated)
        if terminated or truncated:
            break
        observation = next_observation

    # 跟踪监控
    episode_reward = play_montecarlo(env, agent, train=False)
    episode_rewards.append(episode_reward)
    chart.plot(episode_rewards)

plt.show()

# 测试
episode_rewards = [play_montecarlo(env, agent, train=False)
        for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

env.close()