from setscore import *
from gamestate import *
from pynput.keyboard import Key, Listener
#from Image_processing_integrated import *

#line 7-10 from the source https://pythonhosted.org/pynput/keyboard.html
def on_release(key):                                                                    
    if key == Key.n:
        # Stop listener
        return False 

game = GameState()
score = MyScore()
#cam = CameraControl()
n=0
game.start()


cam.run_code("getting_target")

while(not game.end):
    print("Round "+str(n+1))
    newscore=[]
    for i in range(1,len(game.scoresheet)+1):
        print(game.scoresheet.get(i)[0]+" to throw.")
        distance=min(cam.run_code("getting_data"))
        newscore.append(distance)
        
        with Listener(
            on_release=on_release) as listener:
                listener.join()

    newscore = score.scale(newscore)
    score.update(game.scoresheet, newscore)
    n+=1
    if(n==4):
        game.endstate(game.scoresheet)
    
    
    
