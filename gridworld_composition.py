#%% python 3.9
import gym
from gym import spaces
import pygame as pg
import os
import numpy as np

#%%
class GCEnv(gym.Env):
    def __init__(self, render_mode='human', stage_total=1):
        ## 화면 출력 여부 설정
        self.render_mode = render_mode
        self.screen = None
        self.render_once = False # 한번이라도 렌더 되었는지 여부
        
        ## 게임 기본 요소 설정
        self.G_clock = None # Time
        self.G_stage_total = stage_total # 한 에피소드 당 스테이지 수
        self.grid_num = (9,7) # 맵의 크기 (나중에 고칠 것)
        
        # 내부 클래스 선언
        self.G_player = self.Player(self, (self.grid_num[0]//2,self.grid_num[1]//2)) # Player
        self.G_item = self.ItemCreator(0,0) # Item
        
        ## Action 유형 선언
        self.action_space = spaces.Discrete(5) # 0:pick, (1,2,3,4):move
   
    
    def step(self, action):
        # 플레이어 액션
        self.G_player.action(action)
        
        CC = self.content_classifier()
        
        # 아이템 3개를 모두 수집한 경우
        if (self.G_player.inventory_num == 3):
            self.reward = self.reward_check() # 인벤토리 검사
            self.G_player.reset() # 플레이어 리셋
            self.G_stage_change = True # 다음 스테이지로
            self.G_stage_number += 1 # 다음 스테이지로
        else:
            self.reward = 0
        
        # 스테이지가 처음 시작되었을 경우
        if self.G_stage_change:
            self.G_item = self.ItemCreator(0,0) # 아이템 생성 (나중에 고칠 것)
            self.G_stage_change = False
            self.render_reset()
        
        # state : [X,Y,Action,]
        # numpy는 리소스 너무 많이 잡아먹어서 튜플로?
        #self.observation = np.array([])
        self.observation = (self.G_player.position[0], self.G_player.position[1], action,
                            self.G_player.inventory_num, CC)
        
        # placeholder
        self.info = {}
        
        # 모든 스테이지가 끝난 경우 (한 에피소드가 끝난 경우)
        if (self.G_stage_number >= self.G_stage_total):
            self.done = True
        else:
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
    
    def reward_check(self):
        # 인벤토리 내부에는 3개의 아이템 content가 들어있을 것
        if (self.G_player.inventory_num != 3):
            raise SystemExit('Cannot find three items in inventory!')
        
        CM = self.content_multiplier()
        
        # CM를 통해 reward를 주는 방식은 아래 코드를 수정하여 고칠 수 있다.
        if any((CM[0]==1,CM[0]==8,CM[0]==27,CM[0]==6)):
            reward = 1
        else:
            reward = -1
        
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
            pg.event.pump()
            self.G_clock.tick(G_FPS)
            pg.display.update()
        elif (self.render_mode == "rgb_array"):
            return np.transpose(np.array(pg.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        
    
    # 한 에피소드가 끝났을 때
    def reset(self):
        # 게임 요소 설정
        self.G_stage_change = True # 새 스테이지 시작 여부
        self.G_stage_number = 0 # 현재 스테이지
        #
        self.G_player.reset()
        self.render_reset()
    
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
        def reset(self):
            environment = self.environment # 플레이어가 위치한 환경
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


#%% Main

# 파일 경로 설정
dir_main = os.path.split(os.path.abspath(__file__))[0]

# 게임 상수 설정
G_FPS = 30
C_white = (255,255,255)
C_black = (0,0,0)
C_border = (96,96,96)
C_inside = (192,192,192)
C_lightgreen = (170,240,180)


#%% 사람 플레이

from pynput import keyboard

G_switch = True
key_switch = False
act = -1

def key_press(key):
    global key_switch, act
    if not key_switch:
        if key == keyboard.Key.space:
            act = 0
        elif key == keyboard.Key.right:
            act = 1
        elif key == keyboard.Key.up:
            act = 2
        elif key == keyboard.Key.left:
            act = 3
        elif key == keyboard.Key.down:
            act = 4
        key_switch = True

def key_release(key):
    global key_switch, G_switch, act
    key_switch = False
    if key == keyboard.Key.esc:
        G_switch = False
        return False

steps = int(input('플레이 스텝을 입력(step): '))

listener = keyboard.Listener(on_press=key_press, on_release=key_release)
listener.start()

# 환경 구성
env = GCEnv()
env.reset()

# 게임 플레이
step = 0

while G_switch and (step<steps):
    if act != -1:
        info = env.step(act)
        step += 1
        act = -1
        print(info)
    else:
        pass
    env.render()

del listener
env.close()


#%% 

