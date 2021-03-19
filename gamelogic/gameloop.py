from gamelogic.setscore import *
from gamelogic.gamestate import *
from pynput.keyboard import Key, Listener
from Camera.Image_processing_failsafe import *
from playsound import playsound
from pcserial.pcserial import * 

#line 8-11 from the source https://pythonhosted.org/pynput/keyboard.html
def on_release(key):                                                                    
    if key == Key.esc:
        # Stop listener
        return False 
game = GameState()
score = MyScore()
cam = CameraControl()
pcs = PcSerial()
n=0
game.start()
height=score.heightscale(game.scoresheet,game.end)
light=score.lightscale(game.scoresheet,game.end)

cam.run_code("getting_target",pcs)

while(not game.end):
    print("\nRound "+str(n+1)+"\n\n")
    newscore=[]
    for i in range(1,len(game.scoresheet)+1):
        print(game.scoresheet.get(i)[0]+" to throw.")
        
        pcs.set_height(height[i-1])
        pcs.set_led_state(light[i-1])

        with Listener(
            on_release=on_release) as listener:
                listener.join()
        distance=min([abs(x) for x in cam.run_code("getting_data",pcs)])
        newscore.append(distance)


        
        

    #newscore = score.scale(newscore)
    newscore = score.demoscale(newscore)
    score.update(game.scoresheet, newscore)
    
    height=score.heightscale(game.scoresheet,game.end)
    
    
    print("\n\nScoreboard\n")
    for i in range(1,len(game.scoresheet)+1):
        pcs.set_led()
        print(game.scoresheet.get(i)[0]+": "+str(game.scoresheet.get(i)[1])+"\n")
    n+=1
    if(n==2):
        game.endstate(game.scoresheet)
cam.stop_pipline()
pcs.set_height(255)
pcs.set_led_state(60)
playsound("gamelogic/play.wav")      
