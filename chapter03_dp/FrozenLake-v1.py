# 引入包
import numpy as np
np.random.seed(8)
import gym

# 环境使用
env = gym.make('FrozenLake-v1')
print("观测空间 = {}".format(env.observation_space))
print("动作空间 = {}".format(env.action_space))
print("观测空间大小 = {}".format(env.observation_space.n))
print("动作空间大小 = {}".format(env.action_space.n))
print("状态14的右移动力:", env.unwrapped.P[14][2])

# 进行一局
def play_once(env, policy, render=False):
    total_reward = 0.0
    observation, _ = env.reset()
    while True:
        if render:
            env.render()
        action = np.random.choice(env.action_space.n,
                                  p=policy[observation])
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

# 采用随机策略进行一轮
random_policy = \
        np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

episode_rewards = [play_once(env,random_policy) for _ in range(100)]
print("随机策略 平均奖励: {}".format(np.mean(episode_rewards)))

# 策略评估
def v2q(env, v, s=None, gamma=1.): # 根据状态价值函数计算动作价值函数
    if s is not None: # 针对单个状态求解
        q = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:
                q[a] += prob * \
                        (reward + gamma * v[next_state] * (1. - terminated))
    else:   #  针对所有状态进行求解
        q = np.zeros((env.observation_space.n, env.action_space.n))
        for s in range(env.observation_space.n):
            q[s] = v2q(env, v, s, gamma)
    return q

def evalute_policy(env, policy, gamma=1., tolerant=1e-6):   # 节省空间的做法
    v = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            vs = sum(policy[s] * v2q(env, v, s, gamma)) # 更新状态价值函数
            delta = max(delta, abs(v[s] - vs)) # 更新最大误差
            v[s] = vs # 更新状态价值函数
        if delta < tolerant:
            break
    return v

# 评估随机策略的价值函数
print("评估随机策略的价值函数")
print("状态价值函数：")
v_random = evalute_policy(env, random_policy)
print(v_random.reshape(4,4))

print("动作价值函数：")
q_random = v2q(env, v_random)
print(q_random)

# 策略改进
print("策略改进")
def improve_policy(env, v, policy, gamma=1.):
    optimal = True
    for s in range(env.observation_space.n):
        q = v2q(env, v, s, gamma)
        a = np.argmax(q)
        if policy[s][a] != 1:
            optimal = False
            policy[s] = 0
            policy[s][a] = 1.
    return optimal

# 对随机策略进行改进
policy = random_policy.copy()
optimal = improve_policy(env, v_random, policy)
if optimal:
    print("无更新，最优策略为：")
else:
    print("有更新，更新后的策略为：")
print(policy)

# 策略迭代
print("策略迭代")
def iterate_policy(env, gamma=1., tolerant=1e-6):
    # 初始化为任意一个策略
    policy = np.ones((env.observation_space.n, env.action_space.n)) \
            / env.observation_space.n
    while True:
        v = evalute_policy(env, policy, gamma, tolerant) # 策略评估
        if improve_policy(env, v, policy):
            break
    return policy, v

policy_pi, v_pi = iterate_policy(env)
print("状态价值函数：")
print(v_pi.reshape(4,4))
print("最优策略 =")
print(np.argmax(policy_pi, axis=1).reshape(4,4))

# 价值迭代
print("价值迭代")
def iterate_value(env, gamma=1., tolerant=1e-6):
    v = np.zeros(env.observation_space.n) # 初始化
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            vmax = max(v2q(env, v, s, gamma)) # 更新价值函数
            delta = max(delta, abs(v[s] - vmax))
            v[s] = vmax
        if delta < tolerant:    # 满足迭代要求
            break

    policy = np.zeros((env.observation_space.n, env.action_space.n))    # 计算最优策略
    for s in range(env.observation_space.n):
        a = np.argmax(v2q(env, v, s,gamma))
        policy[s][a] = 1.
    return policy, v

policy_vi, v_vi = iterate_value(env)
print("状态价值函数：")
print(v_vi.reshape(4,4))
print("最优策略为 =")
print(np.argmax(policy_vi, axis=1).reshape(4,4))