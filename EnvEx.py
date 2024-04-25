from pynput import keyboard
import time

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
    global key_switch, G_switch
    key_switch = False
    if key == keyboard.Key.esc:
        G_switch = False
        return False

listener = keyboard.Listener(on_press=key_press, on_release=key_release)
listener.start()

while G_switch:
    print(act)
    act = -1
    time.sleep(0.1)

del listener

#%%

a=2

def Test():
    if a ==2:
        return 'no'
        print('a')
    return a

print(Test())