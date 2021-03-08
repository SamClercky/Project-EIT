from setscore import *
from gamestate import *

game = GameState()
score = MyScore()
n=0
game.start()

while(not game.end):
    print("Round "+str(n+1))
    newscore=[]
    for i in range(1,len(game.scoresheet)+1):
        print(game.scoresheet.get(i)[0]+" to throw.")
        #newscore.append(?)    
    newscore = score.scale(newscore)
    score.update(game.scoresheet, newscore)
    n+=1
    if(n==4):
        game.endstate(game.scoresheet)
    
    
    
