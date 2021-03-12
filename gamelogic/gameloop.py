from gamelogic.setscore import *
from gamelogic.gamestate import *
from pynput.keyboard import Key, Listener
from Camera.Image_processing_integrated import *
from playsound import playsound 

#line 7-10 from the source https://pythonhosted.org/pynput/keyboard.html
def on_release(key):                                                                    
    if key == Key.esc:
        # Stop listener
        return False 

game = GameState()
score = MyScore()
cam = CameraControl()
n=0
game.start()


cam.run_code("getting_target")

while(not game.end):
    print("Round "+str(n+1))
    newscore=[]
    for i in range(1,len(game.scoresheet)+1):
        print(game.scoresheet.get(i)[0]+" to throw.")
        with Listener(
            on_release=on_release) as listener:
                listener.join()
        distance=min([abs(x) for x in cam.run_code("getting_data")])
        newscore.append(distance)
        
        

    newscore = score.scale(newscore)
    score.update(game.scoresheet, newscore)
    print("Scoreboard")
    for i in range(1,len(game.scoresheet)+1):
        print(game.scoresheet.get(i)[0]+": "+game.scoresheet.get(i)[1])
    n+=1
    if(n==3):
        game.endstate(game.scoresheet)
playsound("gamelogic/play.wav")    
cam.stop_pipline()
    
