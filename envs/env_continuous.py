import gymnasium
from gymnasium import spaces
import numpy as np
from envs.env_core import EnvCore
# from envs.CityLearn.citylearn.citylearn import CityLearnEnv
from citylearn.citylearn import CityLearnEnv

class ContinuousActionEnv(object):
    """
    对于连续动作环境的封装
    Wrapper for continuous action environment.
    """

    def __init__(self):
        self.env = CityLearnEnv("citylearn_challenge_2022_phase_1", central_agent=False)
        dataset = "/home/wortel/Documents/citylearn_benchmark/benchmark/data/neighborhoods/tx_travis_county_neighborhood_10/schema.json"
        # self.env = CityLearnEnv(dataset, central_agent=False)

        self.num_agent = len(self.env.buildings)

        self.signal_obs_dim = self.env.observation_space[0].shape[0]

        self.signal_action_dim = self.env.action_space[0].shape[0]

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agent):
            # physical action space
            u_action_space = spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(self.signal_action_dim,),
                dtype=np.float32,
            )

            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码

        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of environment, there are 2 agents inside, and each agent's action is a 5-dimensional one_hot encoding
        """

        results = self.env.step(actions)
        # print(results)
        obs, rews, dones, trunc, infos = results

        return np.stack(obs), np.stack(rews), [dones] * self.num_agent, infos

    def reset(self):
        obs = self.env.reset()[0]
        # print(obs)
        # for i, o in enumerate(obs):
        #     print(f"Observation {i+1} shape:", len(o))
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass
