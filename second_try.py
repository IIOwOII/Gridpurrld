#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random, collections, torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


learning_rate   = .0005
gamma           = .98
buffer_limiit   = 50000
batch_size      = 1000


# In[3]:


class Gridworld_ItemCollect():
    def __init__(self, size = (4,4), start = (0,0)):
        self.size   = size
        self.max_x  = size[0] - 1
        self.max_y  = size[1] - 1
        self.start  = start
        self.x      = start[0]
        self.y      = start[1]
        self.item_x = random.randint(0, self.max_x)
        self.item_y = random.randint(0, self.max_y)
        self.item   = 0

    def action(self, a):
        if a == 0:
            self.pick_up()
        elif a == 1:
            self.move_left()
        elif a == 2:
            self.move_up()
        elif a == 3:
            self.move_right()
        elif a == 4:
            self.move_down()
        
        reward  = -1
        if self.item == 1:
            reward += 20
        done    = self.is_done()

        return np.array([self.x, self.y, self.item_x, self.item_y]), reward, done

    def move_right(self):  
        if self.y == self.max_y:
            pass
        else:
            self.y += 1
    
    def move_left(self):
        if self.y == 0:
            pass
        else:
            self.y -= 1
    
    def move_up(self):
        if self.x == 0:
            pass
        else:
            self.x -= 1
            
    def move_down(self):
        if self.x == self.max_x:
            pass
        else:
            self.x += 1

    def pick_up(self):
        if (self.x == self.item_x) and (self.y == self.item_y):
            self.item += 1
    
    def is_done(self):
        if self.item == 1:
            return True
        else:
            return False
        
    def get_state(self):
        return np.array([self.x, self.y, self.item_x, self.item_y])
    
    def draw_state(self):
        fig, ax = plt.subplots(1)

        ax.imshow(np.zeros(self.size), cmap = 'gray')
        ax.scatter(self.x, self.y, marker = 'o', color = [.8, .2, .2], s = 400)
        ax.scatter(self.item_x, self.item_y, marker = 'x', color = [.2, .8, .2], s = 400)

        ax.set_xticks(np.arange(self.size[1]) - .5)
        ax.set_yticks(np.arange(self.size[0]) - .5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color = 'white', linewidth = 2)

    def reset(self):
        self.x = self.start[0]
        self.y = self.start[1]
        self.item_x = random.randint(0, self.max_x)
        self.item_y = random.randint(0, self.max_y)
        self.item   = 0

        return np.array([self.x, self.y, self.item_x, self.item_y])


# In[4]:


class RepalyBuffer():
    def __init__(self, buffer_limit = 50000):
        self.buffer = collections.deque(maxlen = buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask_list.append([done_mask])

        return torch.tensor(s_list, dtype = torch.float), torch.tensor(a_list), torch.tensor(r_list), torch.tensor(s_prime_list, dtype = torch.float), torch.tensor(done_mask_list)
    
    def size(self):
        return len(self.buffer)    


# In[5]:


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,4)
        else:
            return out.argmax().item()


# In[6]:


def train(q, q_target, memory, optimzer, batch_size = 1000, gamma = .98, n_iter = 10):
    for i_iter in range(n_iter):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out       = q(s)
        q_a         = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target      = r + gamma * max_q_prime * done_mask
        loss        = F.smooth_l1_loss(q_a, target)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()


# In[7]:


def main(learning_rate = .0005, print_interval = 20, n_epi = 1000, trunc = 50, epsilon_min = .01, epsilon_start = .08, epsilon_step = 200, epsilon_grad = .01, min_memory = 2000, device = torch.device('cpu')):
    env = Gridworld_ItemCollect()

    q           = Qnet()
    q_target    = Qnet()
    q_target.load_state_dict(q.state_dict())

    memory = RepalyBuffer()

    score           = 0.0
    optimizer       = optim.Adam(q.parameters(), lr = learning_rate)

    for i_epi in range(n_epi):
        epsilon = max(epsilon_min, epsilon_start - epsilon_grad * (i_epi / epsilon_step))

        s       = env.reset()
        done    = False
        i_step  = 0

        while not done:
            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon)
            s_prime, r, done = env.action(a)
            done_mask = 0.0 if done else 1.0
            i_step += 1
    
            s = s_prime
            score += r
            memory.put((s, a, r, s_prime, done_mask))
            if i_step == trunc:
                break
        print('Episode{:05d} Done in {:05} steps!'.format(i_epi, i_step))

        if memory.size() > min_memory:
            train(q, q_target, memory, optimizer)

        if i_epi % print_interval == 0 and i_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print('Episode{:05d} - Score: {:.1f}, n_buffer: {:04d}, eps: {:.1f}%'.format(i_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0


# In[8]:


main()

