#%% python 3.9
import gym
from gym import spaces
import pygame as pg
import os
import numpy as np

#%%
class GCEnv(gym.Env):
    def __init__(self, render_mode='human'):
        ## 화면 출력 여부 설정
        self.render_mode = render_mode
        self.screen = None
        self.render_once = False # 한번이라도 렌더 되었는지 여부
        
        ## 게임 기본 요소 설정
        self.G_clock = None # Time
        self.grid_num = (9,7) # 맵의 크기 (나중에 고칠 것)
        
        # 내부 클래스 선언
        self.G_player = self.Player(self, (self.grid_num[0]//2,self.grid_num[1]//2)) # Player
        self.G_item = self.ItemCreator(0,0) # Item
        
        ## Action 유형 선언
        self.action_space = spaces.Discrete(5) # 0:pick, (1,2,3,4):move
        self.action = -1
   
    
    def step(self, action):
        # 플레이어 액션
        self.action = action
        self.G_player.action(self.action)
        
        # state
        CC = self.content_classifier()
        IR = self.iteminfo_returner()
        PR = self.playerpos_returner()
        self.observation = np.array(lst_flatten(
            [PR, self.G_player.inventory_num, CC, IR]))
        
        # placeholder
        self.info = {}
        
        # 아이템 3개를 모두 수집한 경우
        if (self.G_player.inventory_num == 3):
            self.reward = self.reward_check() # 인벤토리 검사
            self.done = True
        else:
            self.reward = -1
            self.done = False
        
        return self.observation, self.reward, self.done, self.info
    
    
    def content_classifier(self):
        """
        content는 1,2,3의 숫자를 사용하여 특성을 구분짓는다.
        다만 이 방식은 딥러닝의 input으로 쓰는데 문제가 있다.
        
        예를 들어, shape content를 생각해보자.
        사람은 content가 1인 item과 content가 3인 item을 원, 사각형과 같이 특성을 부여하는 index로써 이해한다.
        그러나 딥러닝에서는 단순히 <사각형은 원의 3배구나!> 와 같이 index가 아닌 하나의 양으로써 이해한다.
        이는 우리의 추구 방향과는 맞지 않으므로, 인벤토리 내의 아이템 content를 딥러닝의 input으로 사용할 때는
        단순히 on/off(1/0)값으로 분류해줄 필요가 있다.
        
        다만, 현재 인벤토리에 어떤 content가 특히 많은지를 나타낼 수 있도록,
        (RL agent가 현 인벤토리에 존재하는 아이템의 content가 얼마나 중복하게 가지고 있는지 구분하기를 기대하므로)
        CC의 index는 content의 종류, value는 content의 보유수로 설정하고자 한다.
        """
        inv = self.G_player.inventory
        CC = [0,0,0,0,0,0,0,0,0]
        for c in range(3):
            for i in range(3):
                CC_idx = inv[i].content[c]
                if CC_idx != 0:
                    CC[CC_idx+3*c-1] += 1
        return CC
    
    def content_multiplier(self):
        """
        인벤토리에 있는 각 아이템의 content를 곱하면, 아래와 같은 숫자를 반환할 것이다.
        
        모두 같은 content: 1^3=1, 2^3=8, 3^3=27
        각각 다른 content: 1*2*3=6
        존재하지 않는 content: 0을 적어도 1개 곱하므로 0
        
        여기서 reward를 줘야하는 상황은 바로, 세 아이템 모두 한 dimension에 대하여 
        각각 다른 content를 가지고 있거나, 모두 같은 content를 가지고 있는 상황이다.
        """
        inv = self.G_player.inventory
        CM = (inv[0].content[0]*inv[1].content[0]*inv[2].content[0],
             inv[0].content[1]*inv[1].content[1]*inv[2].content[1],
             inv[0].content[2]*inv[1].content[2]*inv[2].content[2])
        return CM
    
    
    def playerpos_returner(self):
        """
        플레이어의 위치를 onehot 형식으로 반환해준다.
        """
        # 그리드 크기만큼의 PR list 생성
        PR = [0 for idx in range(self.grid_num[0]*self.grid_num[1])]
        
        # 플레이어가 존재하는 좌표 찾고, PR list에 onehot
        player_pos = self.G_player.position[1]*self.grid_num[0]+self.G_player.position[0]
        PR[player_pos] = 1
        
        return PR
    
    def iteminfo_returner(self):
        """
        현재 맵에 존재하는 아이템의 정보를 반환해준다.
        1. gridworld size가 9*7이면, 총 63칸의 list를 만든다.
        2. 아이템이 (x,y)에 위치하고 content는 (a,b,c)라면, list의 x+y*9번째 칸에 cba(4진법)을 입력한다.
        예1_ 위치(4,0),content(2,0,0)[삼각형,색없음]: list[4]=2
        예2_ 위치(3,2),content(1,3,0)[원,파랑색]: list[21]=13
        """
        # 그리드 크기만큼의 IR list 생성
        IR = [0 for idx in range(self.grid_num[0]*self.grid_num[1])]
        
        # 아이템이 존재하는 좌표 찾고, IR list에 content 입력
        for item in self.G_item:
            if not item.collected:
                item_pos = item.position[1]*self.grid_num[0]+item.position[0]
                IR[item_pos] = item.content[0]+4*item.content[1]+16*item.content[2]
            
        return IR

    
    def reward_check(self):
        # 인벤토리 내부에는 3개의 아이템 content가 들어있을 것
        if (self.G_player.inventory_num != 3):
            raise SystemExit('Cannot find three items in inventory!')
        
        CM = self.content_multiplier()
        
        # CM를 통해 reward를 주는 방식은 아래 코드를 수정하여 고칠 수 있다.
        if any((CM[0]==1,CM[0]==8,CM[0]==27,CM[0]==6)):
            reward = 500
        else:
            reward = 100
        
        return reward
        
    
    def render_initialize(self):
        # 처음 렌더
        self.render_once = True
        
        # 그리드 설정
        self.grid_size = 64
        self.grid_SP = (96,128) # 그리드가 시작되는 위치, 즉 그리드 왼쪽 상단의 위치
        self.grid_EP = (self.grid_SP[0]+self.grid_num[0]*self.grid_size, 
                        self.grid_SP[1]+self.grid_num[1]*self.grid_size)
        
        # Inventory 설정
        self.Inv_SP = (self.G_resolution[0]-self.grid_size*3-128, 128) # 인벤토리 표시가 시작되는 위치(왼쪽 상단)
        self.Inv_height = 32 # 인벤토리 텍스트 표시 폭
        
        # 스프라이트 생성
        self.G_background = pg.transform.scale(load_image("BG.png"), self.G_resolution)
        self.grid_spr = pg.transform.scale(load_image("Grid.png"), (self.grid_size,self.grid_size))

        # 텍스트 설정
        self.G_font = pg.font.Font(None, 24)
        self.text_inventory = self.G_font.render("Inventory", True, C_black) # render(text, antialias, color)
        self.text_inventory_rect = self.text_inventory.get_rect()
        self.text_inventory_rect.center = (self.Inv_SP[0]+self.grid_size*3//2, 
                                           self.Inv_SP[1]+self.Inv_height//2)
        
        # 플레이어 설정
        self.G_player.spr = pg.transform.scale(load_image("Agent.png"),
                                               (self.grid_size,self.grid_size)) # 그리드 1칸 크기만큼 캐릭터 사이즈 조절
        self.G_player.rect = self.G_player.spr.get_rect() # 스프라이트 히트박스
        
        # 렌더 리셋
        self.render_reset() 

    
    def render_reset(self):
        if self.render_once:
            # 플레이어 위치 리셋
            self.G_player.rect.topleft = (self.grid_SP[0]+self.G_player.position[0]*self.grid_size, 
                                          self.grid_SP[1]+self.G_player.position[1]*self.grid_size)     
            # 아이템 리셋
            """
            렌더 이전에 수집된 아이템은 위치를 잘 반영하지 못하는 버그 존재
            """
            for item in self.G_item:
                item.spr = pg.Surface((self.grid_size,self.grid_size), pg.SRCALPHA)
                item.rect = item.spr.get_rect()
                item.rect.topleft = (self.grid_SP[0]+item.position[0]*self.grid_size, 
                                     self.grid_SP[1]+item.position[1]*self.grid_size)
                item.draw()
        
    
    def render(self):
        # 렌더 모드를 설정하지 않은 경우
        if self.render_mode is None:
            gym.logger.warn("You have to specify the render_mode. etc: human")
            return
        
        # 렌더 모드가 agent인 경우
        if (self.render_mode == 'agent'):
            return
        
        # 화면에 띄운 창이 없는 경우
        if self.screen is None:
            pg.init() # 창 띄우기
            pg.display.set_caption('Hello!') # Title
            # 해상도 설정
            self.G_resolution = (1280,720)
            if (self.render_mode == "human"):
                self.screen = pg.display.set_mode(self.G_resolution) # Display Setting
            else:  # mode == "rgb_array"
                self.screen = pg.Surface(self.G_resolution)
            self.render_initialize() # 렌더 초기화 (창 켜질 때 한번만 구동) (스테이지 바뀌어서 맵크기 변하면 코드 수정해야함)
            # 배경 그리기
            self.screen.blit(self.G_background, self.G_background.get_rect())
        
        # clock이 없는 경우
        if self.G_clock is None:
            self.G_clock = pg.time.Clock() # Time
        
        ### Draw in Display
        ## 그리드 그리기
        for row in range(self.grid_num[0]):
            for col in range(self.grid_num[1]):
                self.screen.blit(self.grid_spr,(self.grid_SP[0]+row*self.grid_size,
                                                self.grid_SP[1]+col*self.grid_size, 
                                                self.grid_size, self.grid_size))
        ## 인벤토리 그리기
        # 상단
        pg.draw.rect(self.screen, C_lightgreen, 
                     (self.Inv_SP[0], self.Inv_SP[1], self.grid_size*3, self.Inv_height))
        pg.draw.rect(self.screen, C_black, 
                     (self.Inv_SP[0], self.Inv_SP[1], self.grid_size*3, self.Inv_height), 2)
        self.screen.blit(self.text_inventory, self.text_inventory_rect)
        # 하단 (내용물)
        for idx in range(3):
            pg.draw.rect(self.screen, C_inside, 
                         (self.Inv_SP[0]+idx*self.grid_size, self.Inv_SP[1]+self.Inv_height, 
                          self.grid_size, self.grid_size))
            pg.draw.rect(self.screen, C_border, 
                         (self.Inv_SP[0]+idx*self.grid_size, self.Inv_SP[1]+self.Inv_height, 
                          self.grid_size, self.grid_size), 2)
        ## 아이템 그리기
        for item in self.G_item:
            self.screen.blit(item.spr, item.rect)
        ## 플레이어 그리기
        self.G_player.rect.topleft = (self.grid_SP[0]+self.G_player.position[0]*self.grid_size, 
                                      self.grid_SP[1]+self.G_player.position[1]*self.grid_size)
        self.screen.blit(self.G_player.spr, self.G_player.rect)
        
        if (self.render_mode == "human"):
            self.G_clock.tick(G_FPS)
            pg.display.update()
        elif (self.render_mode == "rgb_array"):
            return np.transpose(np.array(pg.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        
    
    # 한 에피소드가 끝났을 때
    def reset(self):
        # Reset
        self.G_item = self.ItemCreator(0,0) # 아이템 생성 (나중에 고칠 것)
        self.G_player.reset()
        self.render_reset()
        self.done = False

        # State
        CC = self.content_classifier()
        IR = self.iteminfo_returner()
        PR = self.playerpos_returner()
        self.observation = np.array(lst_flatten(
            [PR, self.G_player.inventory_num, CC, IR]))
        
        return self.observation
    
    
    # 게임 종료
    def close(self):
        pg.quit()
    
    
    ## Game Component
    # 플레이어 (Agent)
    class Player:
        def __init__(self, environment, position):
            self.environment = environment # 플레이어가 위치한 환경
            self.position = position # position=(4,6)이라면, 5열 7행에 플레이어 배치
            self.inventory = [environment.Item(environment),
                              environment.Item(environment),
                              environment.Item(environment)]
            self.inventory_num = 0
            self.spr = None
            self.rect = None # 스프라이트 히트박스
        
        def action(self, command):
            if (command == 0): # 아이템 먹기
                self.pick()
            elif (command in (1,2,3,4)): # 이동
                self.move(command)
            elif (command == -1): # Null (아무 행동하지 않음)
                pass
            else: # 혹시나 버그로 인해 command가 이상한 값으로 설정되어 있을 때
                pg.exit()
                raise SystemExit('Cannot find action command')
        
        # 아이템 수집
        def pick(self):
            environment = self.environment
            for item in self.environment.G_item:
                # 플레이어와 위치도 겹치고, 아직 수집 되지 않은 아이템인 경우
                if (item.position == self.position) and not item.collected:
                    if environment.render_once:
                        item.rect.topleft = (environment.Inv_SP[0]+self.inventory_num*environment.grid_size, 
                                             environment.Inv_SP[1]+environment.Inv_height)
                    item.collected = True
                    self.inventory[self.inventory_num] = item
                    self.inventory_num += 1
        
        # 이동
        def move(self, direction):
            if (direction == 1) and (self.position[0] < self.environment.grid_num[0]-1): # Right
                self.position = (self.position[0]+1, self.position[1])
            elif (direction == 2) and (self.position[1] > 0): # Up
                self.position = (self.position[0], self.position[1]-1)
            elif (direction == 3) and (self.position[0] > 0): # Left
                self.position = (self.position[0]-1, self.position[1])
            elif (direction == 4) and (self.position[1] < self.environment.grid_num[1]-1): # Down
                self.position = (self.position[0], self.position[1]+1)
            
        # 리셋
        def reset(self, position=None):
            # 플레이어가 위치한 환경 정의
            environment = self.environment
            
            # 위치 설정
            if position is None: # 입력 변수가 없다면 현재 자리 그대로
                pass
            else:
                self.position = position
            
            # 인벤토리 비우기
            self.inventory = [environment.Item(environment),
                              environment.Item(environment),
                              environment.Item(environment)]
            self.inventory_num = 0


    # 아이템
    class Item:
        """
        content: tuple (shape, color, texture)
        shape: 1=circle, 2=triangle, 3=square
        color: (0=None) 1=Red, 2=Green, 3=Blue
        texture: (0=None) 1=?, 2=?, 3=?
        """
        def __init__(self, environment, content=(0,0,0), position=(0,0)):
            self.environment = environment # 아이템이 위치하는 환경
            self.position = position # pos=(4,6)이라면, 5열 7행에 아이템 배치
            self.content = content
            self.collected = False # 인벤토리에 수집된 여부
            self.spr = None
            self.rect = None
            
            """
            self.spr = pg.Surface((grid_size,grid_size), pg.SRCALPHA)
            self.rect = self.spr.get_rect()
            self.rect.topleft = (grid_SP[0]+pos[0]*grid_size, grid_SP[1]+pos[1]*grid_size)
            """
        
        def draw(self):
            # Dimension: 색 설정
            if self.content[1]==0:
                color = (0,0,0)
            elif self.content[1]==1:
                color = (255,0,0)
            elif self.content[1]==2:
                color = (0,255,0)
            elif self.content[1]==3:
                color = (0,0,255)
            else:
                pg.exit()
                raise SystemExit('Cannot find color content')
            # Dimension: 모양 설정
            grid_size = self.environment.grid_size
            if self.content[0]==1:
                pg.draw.circle(self.spr, color, (grid_size//2,grid_size//2), grid_size//2-8)
            elif self.content[0]==2:
                pg.draw.polygon(self.spr, color, ((8,grid_size-8),(grid_size//2,8),(grid_size-8,grid_size-8)))
            elif self.content[0]==3:
                pg.draw.rect(self.spr, color, (8,8,grid_size-16,grid_size-16))
        

    ## 
    def ItemCreator(self, mode_dimension=0, mode_position=0):
        ItemSet = [self.Item(self,(1,1,0),(0,0)), 
                   self.Item(self,(2,2,0),(3,2)), 
                   self.Item(self,(3,3,0),(4,6)), 
                   self.Item(self,(2,1,0),(7,1)), 
                   self.Item(self,(2,3,0),(6,1))]
        return ItemSet
    
    
#%% Game System
def load_image(file):
    """loads an image, prepares it for play"""
    file = os.path.join(dir_main, "Data", file)
    try:
        surface = pg.image.load(file)
    except:
        pg.quit()
        raise SystemExit(f'Could not load image "{file}"')
    return surface.convert_alpha()

def load_sound(file):
    """because pygame can be compiled without mixer."""
    if not pg.mixer:
        return None
    file = os.path.join(dir_main, "Data", file)
    try:
        sound = pg.mixer.Sound(file)
        return sound
    except pg.error:
        print(f"Warning, unable to load, {file}")
    return None


#%% 
# 다중 리스트를 단일 리스트로 변환
def lst_flatten(lst):
    res = []
    for ele in lst:
        if isinstance(ele, list) or isinstance(ele, tuple): # 원소가 리스트 또는 튜플일 경우
            res.extend(lst_flatten(ele))
        else:
            res.append(ele)
    return res


#%% Main

# 파일 경로 설정
dir_main = os.path.split(os.path.abspath(__file__))[0]

# 게임 상수 설정
G_FPS = 100
C_white = (255,255,255)
C_black = (0,0,0)
C_border = (96,96,96)
C_inside = (192,192,192)
C_lightgreen = (170,240,180)


# #%% 사람 플레이
# from pynput import keyboard

# G_switch = True
# key_switch = False
# act = -1

# def key_press(key):
#     global key_switch, act
#     if not key_switch:
#         if key == keyboard.Key.space:
#             act = 0
#         elif key == keyboard.Key.right:
#             act = 1
#         elif key == keyboard.Key.up:
#             act = 2
#         elif key == keyboard.Key.left:
#             act = 3
#         elif key == keyboard.Key.down:
#             act = 4
#         key_switch = True

# def key_release(key):
#     global key_switch, G_switch, act
#     key_switch = False
#     if key == keyboard.Key.esc:
#         G_switch = False
#         return False

# steps = int(input('플레이 스텝을 입력(step): '))

# listener = keyboard.Listener(on_press=key_press, on_release=key_release)
# listener.start()

# # 환경 구성
# env = GCEnv()
# env.reset()

# # 게임 플레이
# step = 0

# while G_switch and (step<steps):
#     if act != -1:
#         info = env.step(act)
#         step += 1
#         act = -1
#         print(info)
#     else:
#         pass
#     env.render()

# del listener
# env.close()


#%% keras 모델링
import tensorflow.keras as tfk
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt

# Q network의 output은 각 action에 대한 Q값이다.
def Q_network(shape_state, shape_action):
    Layers = [tfk.layers.Dense(units=256, name='State', activation='relu', input_dim=shape_state),
              tfk.layers.Dense(units=256, name='Hidden', activation='relu'),
              tfk.layers.Dense(units=shape_action, name='Action', activation='linear')]
    q_net = tfk.models.Sequential(Layers, name='NN')
    q_net.compile(optimizer='Adam', loss='mse')
    return q_net


class ReplayBuffer:
    def __init__(self, buffer_size):
        # 가장 오래된 replay memory부터 지우는 deque로 정의
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
    
    def sample(self, batch_size):
        # batch_size만큼 replay memory 샘플링
        if (len(self.buffer) > batch_size):
            size = batch_size
        else:
            size = len(self.buffer)
        return random.sample(self.buffer, size)
    
    def clear(self):
        # buffer 제거
        self.buffer.clear()
    
    def append(self, transition):
        self.buffer.append(transition)
    
    def __len__(self):
        return len(self.buffer)


class Agent_DQN:
    def __init__(self, env, q_net, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                  buffer_size=2000, batch_size=64, episode_limit=20, step_truncate=1000, step_target_update=100):
        # 환경 입력
        self.env = env
        
        # policy 초기화
        self.Qnet_behavior = q_net
        self.Qnet_target = q_net
        
        # state와 action의 dimension
        self.S_dim = q_net.layers[0].input.shape[1]
        self.A_dim = q_net.layers[len(q_net.layers)-1].output.shape[1]
        
        # Discount factor
        self.gamma = gamma
        
        # Exploration factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # DQN 특성
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.step_target_update = step_target_update # target network를 업데이트하는 주기
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # 딥러닝
        self.loss_func = tfk.losses.MeanSquaredError() # loss는 MSE로 계산
        self.optimizer = tfk.optimizers.Adam() # 그래디언트 디센트 기법은 Adam으로
        
        # 최대 에피소드 & 한 에피소드 당 최대 스텝 수 설정
        self.step_truncate = step_truncate
        self.episode_limit = episode_limit
        
    
    @tf.function
    def learn(self, S, A, S_next, R, done):
        """
        Q 함수를 예측하는 Q net을 학습
        S = state(t)
        A = action(t)
        S_next = state(t+1)
        R = reward(t+1)
        """
        Q_target = R + (1-done)*self.gamma*tf.reduce_max(self.Qnet_target(S_next), axis=1, keepdims=True)
        
        with tf.GradientTape() as tape:
            Q_behavior = self.Qnet_behavior(S)
            A_onehot = tf.one_hot(tf.cast(tf.reshape(A, [-1]), tf.int32), self.A_dim)
            Q_behavior = tf.reduce_sum(Q_behavior*A_onehot, axis=1, keepdims=True)
            loss = self.loss_func(Q_target, Q_behavior)
        grads = tape.gradient(loss, self.Qnet_behavior.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.Qnet_behavior.trainable_weights))
    
    
    # state가 주어졌을 때 action을 선택하는 방식
    def epsilon_greedy(self, S):
        # 탐색률 업데이트 (최소 0.01까지만)
        self.epsilon = max(self.epsilon, 0.01)
        if np.random.rand() <= self.epsilon: # 탐색
            return np.random.randint(self.A_dim)
        else:
            A = self.Qnet_behavior(S[np.newaxis])
            #return np.random.choice(range(self.A_dim), p=A[0].numpy())
            return np.argmax(A[0])
    
    
    def target_update(self):
        self.Qnet_target.set_weights(self.Qnet_behavior.get_weights())
            
    
    def train(self):
        """
        S = state(t)
        A = action(t)
        S_next = state(t+1)
        R = reward(t+1)
        G = return
        """
        epochs = 0
        global_step = 0
        G_list = []
        while epochs < self.episode_limit:
            S = env.reset()
            G = 0
            step = 0
            while step < self.step_truncate:
                A = self.epsilon_greedy(S) # 
                S_next, R, done, _ = env.step(A)
                
                # Replay Buffer 생성
                transition = (S, A, S_next, R, done)
                self.replay_buffer.append(transition)
                
                # return(한 에피소드 내 모든 보상, 따라서 gamma decay 안함)과 state update
                G += R
                S = S_next
                
                self.env.render()
                
                if (global_step%100)==0:
                    print(global_step)
                
                # replay buffer가 충분히 모이면 Qnet 학습 시작
                if global_step > 1000:
                    transitions = self.replay_buffer.sample(batch_size=self.batch_size)
                    self.learn(*map(lambda x: np.vstack(x).astype('float32'), np.transpose(transitions)))
                
                # step update
                step += 1
                global_step += 1
                
                # 
                if done:
                    break
            
            #
            G_list.append(G)
            self.target_update()
            print(f'episode:{epochs+1}, score:{G}, step:{step}')
            
            # episode & epsilon update
            epochs += 1
            self.epsilon *= self.epsilon_decay
        
        # 학습 종료
        env.close()
        
        # reward 출력
        plt.title('Composition Gridworld')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.plot(G_list)
        plt.show()


#%% CNN
import cv2

# This function can resize to any shape, but was built to resize to 84x84
def process_frame(frame, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """

    # cv2를 통한 데이터 감소를 이용하기 위해선 반드시 데이터 형을 np.uint8로 만들어야함
    frame = frame.astype(np.uint8)

    # RGB -> GRAY로 색 변경
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # 프레임을 crop한다 ([0~34, 160 ~ 160+34] 범위 삭제 )
    frame = frame[34:34+160, :160]

    # 크기를 인자로 전달된 shape로 리사이징 한다
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)

    # reshape한다
    frame = frame.reshape((*shape, 1))

    return frame


#%% GPU 사용 가능 여부 확인
"""
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0","/GPU:1"])
gpus = tf.config.experimental.list_physical_devices('GPU') # 호스트 러나임에 표시되는 GPU 장치 목록 반환
"""

#%% DQN 학습
# 환경 구성
env = GCEnv(render_mode='human')

# 모델 구성
shape_state = env.grid_num[0]*env.grid_num[1]*2 + 10
shape_action = env.action_space.n
q_net = Q_network(shape_state, shape_action)
q_net.summary() # 모델 정보

# 모델 구조 시각화
#tfk.utils.plot_model(q_net, show_shapes=True, show_layer_activations=True)

agent = Agent_DQN(env=env, q_net=q_net, step_truncate=1000, episode_limit=1000)
agent.train()