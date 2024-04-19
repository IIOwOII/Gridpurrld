#%% 1. Test Random Environment with OpenAI Gym
import gym
from gym import spaces
import numpy as np
import random

#%%
class GCEnv(gym.Env):
    def __init__(self):
        self.render_mode = None
        self.action_space = spaces.Discrete(5) # 0:pick, (1,2,3,4):move
        #
    
    def step(self, action):
        pass
    
    def render(self, render_mode='human'):
        pass
    
    def reset(self):
        pass

