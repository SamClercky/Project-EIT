from gamestate import *
from setscore import *
from pynput.keyboard import Key, Controller
from pynput.keyboard import Key, Listener
 
dictt = {1 : ["Player 1", 10], 2 : ["Player 2", 100], 3 : ["Player 3", 100]}
game = GameState()
score = MyScore()
array=[100,651321,651321,4656]

currentstage=True

def on_press(key):
    if key == Key.space:
        currentstage=False

def on_release(key):                                                                    
    if key == Key.esc:
        # Stop listener
        return False                

# Collect events until released
with Listener(
        on_release=on_release) as listener:
    listener.join() 
print(score.scale(array))
