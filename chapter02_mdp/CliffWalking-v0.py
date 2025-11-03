import numpy as np
np.random.seed(0)
import scipy.optimize
import gym

env = gym.make('CliffWalking-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('状态数量 = {}, 动作数量 = {}'.format(env.nS, env.nA))
print('地图大小 = {}'.format(env.shape))

# 运行一回合
def play_once(env, policy):
    total_reward = 0
    state, _ = env.reset()
    while True:
        loc = np.unravel_index(state, env.shape)
        print('状态 = {}, 位置 = {}'.format(state, loc), end=' ')
        action = np.random.choice(env.nA, p=policy[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        print('动作 = {}, 奖励 = {}'.format(action, reward))
        total_reward += reward
        if terminated or truncated:
            break
        state = next_state
    return total_reward

actions = np.ones(env.shape, dtype=int)
actions[-1, :] = 0
# print(actions)
actions[:, -1] = 2
# print(actions)
optimal_policy = np.eye(4)[actions.reshape(-1)]
# print(optimal_policy)

total_reward = play_once(env, optimal_policy)
print('回合奖励 = {}'.format(total_reward))

# 求解Bellman期望方程
def evaluate_bellman(env, policy, gamma=1.):
    a, b = np.eye(env.nS), np.zeros((env.nS)) # b是常数项矩阵，a是系数矩阵
    for state in range(env.nS - 1):  # 假设最后一个动作是env.nS - 1，最后一个动作不需要遍历，因为价值为0
        for action in range(env.nA):    # 遍历所有的动作
            pi = policy[state][action]  # 获取策略选择该动作的概率
            for p, next_state, reward, terminated in env.P[state][action]: # env.P[state][action]存储了状态state下执行action后的所有可能转移
                a[state, next_state] -= (pi * gamma * p)
                b[state] += (pi * reward * p)
    v = np.linalg.solve(a, b)   # 计算状态价值
    q = np.zeros((env.nS, env.nA))  # 初始化动作价值
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, terminated in env.P[state][action]:
                q[state][action] += ((reward + gamma * v[next_state]) * p)
    return v, q

# 评估随机策略的价值
policy = np.random.uniform(size=(env.nS, env.nA))
policy = policy / np.sum(policy, axis=1)[:, np.newaxis]

state_values, action_values = evaluate_bellman(env, policy)
print('状态价值 = {}'.format(state_values))
print('动作价值 = {}'.format(action_values))

# 评估最优策略的价值

optimal_state_values, optimal_action_values = evaluate_bellman(env, optimal_policy)
print('最优状态价值 = {}'.format(optimal_state_values))
print('最优动作价值 = {}'.format(optimal_action_values))

# 求解Bellman最优方程
def optimal_bellman(env, gamma=1.):
    p = np.zeros((env.nS, env.nA, env.nS))
    r = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for prob, next_state, reward, terminated in env.P[state][action]:
                p[state, action, next_state] += prob
                r[state, action] += (reward * prob) # 计算期望奖励
    c = np.ones(env.nS) # 线性规划定义目标函数
    a_ub = gamma * p.reshape(-1, env.nS) - \
            np.repeat(np.eye(env.nS), env.nA, axis=0) # γ
    b_ub = -r.reshape(-1)
    a_eq = np.zeros((0, env.nS))
    b_eq = np.zeros(0)
    bounds = [(None, None),] * env.nS
    res = scipy.optimize.linprog(c, a_ub, b_ub, bounds=bounds,
            method='interior-point')
    v = res.x
    q = r + gamma * np.dot(p, v)
    return v, q

optimal_state_values, optimal_action_values = optimal_bellman(env)
print('最优状态价值 = {}'.format(optimal_state_values))
print('最优动作价值 = {}'.format(optimal_action_values))

optimal_actions = optimal_action_values.argmax(axis=1)
print('最优策略 = {}'.format(optimal_actions))

env.close()