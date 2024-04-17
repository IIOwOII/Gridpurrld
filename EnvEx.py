#%%
"""
pip install tensorflow==2.3.0
pip install gym
pip install keras
pip install keras-r12
"""

#%%
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

#%%
class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38+random.randint(-3,3)
        self.shower_length = 60
        
    def step(self, action):
        # action = 0,1,2
        self.state += action-1
        self.shower_length -= 1
        
        # Reward
        if self.state >=37 and self.state <=39:
            reward = 1
        else:
            reward = -1
        
        if self.shower_length <= 0:
            done = True
        else:
            done = False
        
        # State noise
        self.state += random.randint(-1,1)
        # placeholder for info
        info = {}
        
        return self.state, reward, done, info
        
    def render(self):
        pass
    
    def reset(self):
        # Reset Env
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60
        return self.state

#%%
env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

print(actions)