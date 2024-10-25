#%% python 3.9
import pygame as pg
import os

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


#%% Game Component
## 플레이어 (Agent)
class Player:    
    def __init__(self, pos):
        self.pos = pos
        self.spr = pg.transform.scale(load_image("Agent.png"),(grid_size,grid_size)) # 그리드 1칸 크기만큼 캐릭터 사이즈 조절
        self.rect = self.spr.get_rect() # 스프라이트 히트박스
        self.rect.topleft = (grid_SP[0]+(grid_num[0]//2)*grid_size, grid_SP[1]+(grid_num[1]//2)*grid_size)
        self.inventory = []
    
    def action(self, command):
        if (command == 0): # 아이템 먹기
            self.pick()
        elif (command in (1,2,3,4)): # 이동
            self.move(command)
        else: # 혹시나 버그로 인해 command가 이상한 값으로 설정되어 있을 때
            pg.exit()
            raise SystemExit('Cannot find action command')
    
    # 아이템 수집
    def pick(self):
        item_idx = self.rect.collidelist(G_item)
        if (item_idx != -1): # 플레이어가 아이템과 닿아있는 상태인 경우
            self.inventory.append(G_item[item_idx].taken())
    
    # 이동
    def move(self, direction):
        pos = self.rect.center
        if (direction == 1) and (self.rect.right < grid_EP[0]): # Right
            pos = (pos[0]+grid_size, pos[1])
        elif (direction == 2) and (self.rect.top > grid_SP[1]): # Up
            pos = (pos[0], pos[1]-grid_size)
        elif (direction == 3) and (self.rect.left > grid_SP[0]): # Left
            pos = (pos[0]-grid_size, pos[1])
        elif (direction == 4) and (self.rect.bottom < grid_EP[1]): # Down
            pos = (pos[0], pos[1]+grid_size)
        self.rect.center = pos
        
    # 리셋
    def reset(self):
        self.rect.topleft = (grid_SP[0]+(grid_num[0]//2)*grid_size, grid_SP[1]+(grid_num[1]//2)*grid_size)
        self.inventory = []

## 아이템
class Item:
    """
    content: tuple (shape, color, texture)
    shape: 1=circle, 2=triangle, 3=square
    color: (0=None) 1=Red, 2=Green, 3=Blue
    texture: (0=None) 1=?, 2=?, 3=?
    """
    def __init__(self, content=(1,0,0), pos=(0,0)):
        self.spr = pg.Surface((grid_size,grid_size), pg.SRCALPHA)
        self.content = content
        self.rect = self.spr.get_rect()
        # pos=(4,6)이라면, 5열 7행에 아이템 배치
        self.rect.topleft = (grid_SP[0]+pos[0]*grid_size, grid_SP[1]+pos[1]*grid_size)
        
        # Dimension: 색 설정
        if content[1]==0:
            color = (0,0,0)
        elif content[1]==1:
            color = (255,0,0)
        elif content[1]==2:
            color = (0,255,0)
        elif content[1]==3:
            color = (0,0,255)
        else:
            pg.exit()
            raise SystemExit('Cannot find color content')
        
        # Dimension: 모양 설정
        if content[0]==1:
            pg.draw.circle(self.spr, color, (grid_size//2,grid_size//2), grid_size//2-8)
        elif content[0]==2:
            pg.draw.polygon(self.spr, color, ((8,grid_size-8),(grid_size//2,8),(grid_size-8,grid_size-8)))
        elif content[0]==3:
            pg.draw.rect(self.spr, color, (8,8,grid_size-16,grid_size-16))
    
    # 아이템이 수집됨 (아이템 정보 반환)
    def taken(self):
        # 아이템이 인벤토리로 옮겨짐
        self.rect.topleft = (Inv_SP[0]+len(G_player.inventory)*grid_size, Inv_SP[1]+Inv_height)
        return self.content

## 아이템 생성자
def ItemCreator(mode_dimension=0, mode_position=0):
    ItemSet = [Item((1,1,0),(0,0)), Item((2,2,0),(3,2)), Item((3,3,0),(4,6)), Item((2,1,0),(7,1)), Item((2,3,0),(6,1))]
    return ItemSet
    
#%% Main
#def main():
# pygame 기능 사용을 시작
pg.init()

# 게임 정보 설정
pg.display.set_caption('Hello!') # Title

# 게임 파일이 있는 경로
dir_main = os.path.split(os.path.abspath(__file__))[0]

# 게임 상수 설정
G_FPS = 30
G_resolution = (1280,720)
C_white = (255,255,255)
C_black = (0,0,0)
C_border = (96,96,96)
C_inside = (192,192,192)
C_lightgreen = (170,240,180)

# 그리드 설정
grid_num = (9,7)
grid_size = 64
grid_SP = (96,128) # 그리드가 시작되는 위치, 즉 그리드 왼쪽 상단의 위치
grid_EP = (grid_SP[0]+grid_num[0]*grid_size, grid_SP[1]+grid_num[1]*grid_size)

# Inventory 설정
Inv_SP = (G_resolution[0]-grid_size*3-128, 128) # 인벤토리 표시가 시작되는 위치(왼쪽 상단)
Inv_height = 32 # 인벤토리 텍스트 표시 폭

# 텍스트 설정
G_font = pg.font.Font(None, 24)
text_inventory = G_font.render("Inventory", True, C_black) # render(text, antialias, color)
text_inventory_rect = text_inventory.get_rect()
text_inventory_rect.center = (Inv_SP[0]+grid_size*3//2, Inv_SP[1]+Inv_height//2)

# 게임 요소 설정
G_clock = pg.time.Clock() # Time
G_surf = pg.display.set_mode(G_resolution) # Display Setting
G_background = pg.transform.scale(load_image("BG.png"), G_resolution)
G_surf.blit(G_background, G_background.get_rect()) # BG
G_grid = pg.transform.scale(load_image("Grid.png"), (grid_size,grid_size))

# 게임 실행
G_switch = True # 게임 On/Off
G_stagechange = True # 새 스테이지 시작 여부
G_player = Player((grid_num[0]//2,grid_num[1]//2)) # Player


while G_switch:
    G_clock.tick(G_FPS) # 설정한 FPS마다 루프 실행
    
    # 스테이지가 처음 시작되었을 경우
    if G_stagechange:
        G_item = ItemCreator(0,0) # 아이템 생성
        G_stagechange = False
    
    # 아이템 3개를 모두 수집한 경우
    if (len(G_player.inventory) == 3):
        G_player.reset()
        G_stagechange = True # 다음 스테이지로
    
    # 발생한 이벤트 수집
    for event in pg.event.get():
        if event.type == pg.QUIT: # 창 닫기
            G_switch = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE: # ESC
                G_switch = False
            # 플레이어 조작
            if event.key == pg.K_SPACE:
                G_player.action(0)
            elif event.key == pg.K_RIGHT:
                G_player.action(1)
            elif event.key == pg.K_UP:
                G_player.action(2)
            elif event.key == pg.K_LEFT:
                G_player.action(3)
            elif event.key == pg.K_DOWN:
                G_player.action(4)
    
    ### Draw in Display
    ## 그리드 그리기
    for row in range(grid_num[0]):
        for col in range(grid_num[1]):
            G_surf.blit(G_grid,(grid_SP[0]+row*grid_size, grid_SP[1]+col*grid_size, grid_size, grid_size))
    ## 인벤토리 그리기
    # 상단
    pg.draw.rect(G_surf, C_lightgreen, (Inv_SP[0], Inv_SP[1], grid_size*3, Inv_height))
    pg.draw.rect(G_surf, C_black, (Inv_SP[0], Inv_SP[1], grid_size*3, Inv_height), 2)
    G_surf.blit(text_inventory, text_inventory_rect)
    # 하단 (내용물)
    for idx in range(3):
        pg.draw.rect(G_surf, C_inside, (Inv_SP[0]+idx*grid_size, Inv_SP[1]+Inv_height, grid_size, grid_size))
        pg.draw.rect(G_surf, C_border, (Inv_SP[0]+idx*grid_size, Inv_SP[1]+Inv_height, grid_size, grid_size), 2)
    ## 아이템 그리기
    for idx in range(len(G_item)):
        G_surf.blit(G_item[idx].spr, G_item[idx].rect)
    ## 캐릭터 그리기
    G_surf.blit(G_player.spr, G_player.rect)
    
    pg.display.update() # 화면 업데이트
        
pg.quit() # 임시 코드

'''
# call the "main" function if running this script
if __name__ == "__main__":
    main()
    pg.quit() # 게임 끝
'''