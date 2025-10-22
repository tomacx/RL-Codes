import numpy as np
np.random.seed(0)
import pandas as pd
# import gymnasium as gym
import gym

space_names = ['观测空间', '动作空间', '奖励范围', '最大步数']
df = pd.DataFrame(columns=space_names)

for env_id in gym.envs.registry:
    try:
        env = gym.make(env_id)  # 当前环境
        observation_space = env.observation_space  # 观测空间
        action_space = env.action_space     # 动作空间
        reward_range = env.reward_range     # 奖励范围
        max_episode_steps = None
        if isinstance(env, gym.wrappers.time_limit.TimeLimit):
            max_episode_steps = env._max_episode_steps
        df.loc[env_id] = [observation_space, action_space, reward_range, max_episode_steps]
    except:
        pass

with pd.option_context('display.max_rows', None):
    print(df)