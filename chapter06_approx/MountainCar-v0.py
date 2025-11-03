import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow.compat.v2 as tf
tf.random.set_seed(0)
from tensorflow import keras

env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('位置范围 = {}'.format((env.unwrapped.min_position,
        env.unwrapped.max_position)))
print('速度范围 = {}'.format((-env.unwrapped.max_speed,
        env.unwrapped.max_speed)))
print('目标位置 = {}'.format(env.unwrapped.goal_position))

# 总是施加向右的力

positions, velocities = [], []
observation, _ = env.reset()
while True:
    positions.append(observation[0])
    velocities.append(observation[1])
    next_observation, reward, terminated, truncated, _ = env.step(2)
    if terminated or truncated:
        break
    observation = next_observation

if next_observation[0] > 0.5:
    print('成功到达')
else:
    print('失败退出')

# 绘制位置和速度图像
# fig, ax = plt.subplots()
# ax.plot(positions, label='position')
# ax.plot(velocities, label='velocity')
# ax.legend()
# plt.show()

# 用线性近似最优策略求解
# 砖瓦编码
class TileCoder:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features: # 冲突处理
            return hash(codeword) % self.features
        self.codebook[codeword] = count
        return count

    def __call__(self, floats=(), ints=()):
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer) /
                        self.layers) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features

# SARSA算法
print("SARSA算法")
class SARSAAgent:
    def __init__(self, env, layers=8, features=1893, gamma=1., learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n # 动作数
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - \
            env.observation_space.low    # 观测空间范围
        self.encoder = TileCoder(layers, features)  # 砖瓦编码器
        self.w = np.zeros(features) # 权重
        self.gamma = gamma # 折扣
        self.learning_rate = learning_rate # 学习率
        self.epsilon = epsilon # 探索

    def encode(self, observation, action): # 编码
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.encoder(states, actions)

    def get_q(self, observation, action): # 动作价值
        features = self.encode(observation, action)
        return self.w[features].sum()

    def decide(self, observation):  # 判决
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [self.get_q(observation, action) for action in
                  range(self.action_n)]
            return np.argmax(qs)

    def  learn(self, observation, action, reward, next_observation,
               terminated, truncating, next_action): # 学习
        u = reward + (1. - terminated) * self.gamma * \
                self.get_q(next_observation, next_action)
        td_error = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += (self.learning_rate * td_error)

def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation, _ = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)  # 终止状态时此步无意义
        if train:
            agent.learn(observation, action, reward, next_observation,
                        terminated, truncated, next_action)
        if terminated or truncated:
            break
        observation, action = next_observation, next_action
    return episode_reward

agent = SARSAAgent(env)

# 训练
episodes = 400
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)
# plt.plot(episode_rewards)
# plt.show()

# 测试
agent.epsilon = 0. # 取消探索
episode_rewards = [play_sarsa(env, agent) for _ in range(100)]
print("平均回合奖励 = {} / {} = {}".format(sum(episode_rewards),
                                          len(episode_rewards), np.mean(episode_rewards)))

# SARSA(lambda)算法
print("SARSA(lambda)算法")

class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env, layers=8, features=1893, gamma=1.,
                 learning_rate=0.03, epsilon=0.001, lambd=0.9):
        super().__init__(env=env, layers=layers, features=features,
                gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lambd = lambd
        self.z = np.zeros(features) # 初始化资格迹

    def learn(self, observation, action, reward, next_observation, terminated, truncated,
            next_action):
        u = reward
        if not terminated:
            u += (self.gamma * self.get_q(next_observation, next_action))
            self.z *= (self.gamma * self.lambd)
            features = self.encode(observation, action)
            self.z[features] = 1. # 替换迹
        td_error = u - self.get_q(observation, action)
        self.w += (self.learning_rate * td_error * self.z)
        if terminated or truncated:
            self.z = np.zeros_like(self.z) # 为下一回合初始化资格迹

agent = SARSALambdaAgent(env)

# 训练
episodes = 140
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

# 绘制价值和策略

poses = np.linspace(env.unwrapped.min_position,
                    env.unwrapped.max_position, 128)
vels = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, 128)
positions, velocities = np.meshgrid(poses, vels)

# 绘制动作价值估计
@np.vectorize
def get_q(position, velocity, action):
    return agent.get_q((position, velocity), action)

q_values = np.empty((len(poses), len(vels), 3))
for action in range(3):
    q_values[:, :, action] = get_q(positions, velocities, action)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for action, ax in enumerate(axes):
    c = ax.pcolormesh(positions, velocities, q_values[:, :, action], shading='auto')
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    fig.colorbar(c, ax=ax)
    ax.set_title('action = {}'.format(action))

# 绘制状态价值估计

v_values = q_values.max(axis=-1)

fig, ax = plt.subplots(1, 1)
c = ax.pcolormesh(positions, velocities, v_values, shading='auto')
ax.set_xlabel('position')
ax.set_ylabel('velocity')
fig.colorbar(c, ax=ax);
# plt.show()

# 绘制动作估计

@np.vectorize
def decide(position, velocity):
    return agent.decide((position, velocity))

q_values = np.empty((len(poses), len(vels), 3))
action_values = decide(positions, velocities)

fig, ax = plt.subplots()
c = ax.pcolormesh(positions, velocities, action_values, shading='auto')
ax.set_xlabel('position')
ax.set_ylabel('velocity')
fig.colorbar(c, ax=ax, boundaries=[-.5, .5, 1.5, 2.5], ticks=[0, 1, 2]);
# plt.show()

# 深度Q网络求解最优策略
print("深度Q网络")

class Chart:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)
        # plt.ion()

    def plot(self, episode_rewards):
        self.ax.clear()
        self.ax.plot(episode_rewards)
        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('episode reward')
        self.fig.canvas.draw()

# 经验回放 补充：代码和原理
class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward', 'next_observation', 'terminated'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = np.asarray(args, dtype=object)
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

# DQN
class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001,
                 replayer_capacity=10000, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity) # 经验回放

        self.evalute_net = self.build_network(input_size=observation_dim,
                                              output_size=self.action_n, **net_kwargs) # 评估网络
        self.target_net = self.build_network(input_size=observation_dim,
                                             output_size=self.action_n, **net_kwargs) # 目标网络
        self.target_net.set_weights(self.evalute_net.get_weights())

    def build_network(self, input_size, hidden_sizes, output_size,
                    activation=tf.nn.relu, output_activation=None,
                    learning_rate=0.01): # 构建网络
        model = keras.Sequential()
        for layer, hidden_size in  enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation)) # 输出层
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        self.replayer.store(observation, action, reward, next_observation,
                            terminated) # 存储经验
        observations, actions, rewards, next_observations, terminateds = \
                                    self.replayer.sample(self.batch_size) # 经验回放

        next_qs = self.target_net.predict(next_observations, verbose=0)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - terminateds) * next_max_qs
        targets = self.evalute_net.predict(observations, verbose=0)
        targets[np.arange(us.shape[0]), actions] = us
        self.evalute_net.fit(observations, targets, verbose=0)

        if terminated or truncated: # 更新目标网络
            self.target_net.set_weights(self.evalute_net.get_weights())

    def decide(self, observation): # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evalute_net.predict(observation[np.newaxis], verbose=0)
        return np.argmax(qs)

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


net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.001}
agent = DQNAgent(env, net_kwargs=net_kwargs)

# 训练
episodes = 500
episode_rewards = []
# chart = Chart()
for episode in range(episodes):
    print(f"第 {episode + 1}/{episodes} 轮，奖励：{episode_reward}")
    episode_reward = play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)
    # chart.plot(episode_rewards)

# 测试
agent.epsilon = 0. # 取消探索
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))

# 双重DQN
class DoubleDQNAgent(DQNAgent):
    def learn(self, observation, action, reward, next_observation, terminated, truncated):
        self.replaye.store(observation, action, reward, next_observation,
                           terminated) # 存储经验
        observations, actions, rewards, next_observations, terminateds = \
            self.replayer.sample(self.batch_size)  # 经验回放
        next_eval_qs = self.evaluate_net.predict(next_observations, verbose=0)
        next_actions = next_eval_qs.argmax(axis=-1)
        next_qs = self.target_net.predict(next_observations, verbose=0)
        next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions]
        us = rewards + self.gamma * next_max_qs * (1. - terminateds) # 换成用评估网络来更新价值
        targets = self.evaluate_net.predict(observations, verbose=0)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if terminated or truncated:
            self.target_net.set_weights(self.evaluate_net.get_weights())

net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.001}
agent = DoubleDQNAgent(env, net_kwargs=net_kwargs)

# 训练
episodes = 500
episode_rewards = []
chart = Chart()
for episode in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)
    chart.plot(episode_rewards)

# 测试
agent.epsilon = 0. # 取消探索
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))