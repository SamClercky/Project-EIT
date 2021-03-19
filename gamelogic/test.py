from gamestate import *
from setscore import *
from pynput.keyboard import Key, Controller
from pynput.keyboard import Key, Listener
from playsound import playsound 
dictt = {1 : ["Player 1", 100], 2 : ["Player 2", 120], 3 : ["Player 3", 130]}
game = GameState()
score = MyScore()
array=[100,651321,651321]
test=[-10,-315]
score.update(dictt,score.scale(array))
print(dictt)
print(score.heightscale(dictt,True))
print(4%4)