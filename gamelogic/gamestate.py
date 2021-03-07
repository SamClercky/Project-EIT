class GameState():
    """If one player succeeded in scoring 100 or more then he is declared the winner of the match. If multiple players scored over 100, 
    then the player with the most points wins. If multiple players have the same score that is above or equal to 100 then the game goes 
    into sudden death. 
    (sudden death : the game continues, where only the players with the highest scores go into the next round untill one player remains. 
    The remaining player is the winner. ) """

    end = False
    scoresheet = None
    
    def _init_(self):
        self.end = end
        self.scoresheet = scoresheet
        
    def start(self):
        print("How many players are there ?")
        self.scoresheet = {}
        for i in range(int(input())):
            print("Player "+str(i+1)+": ")
            self.scoresheet[i+1]=[str(input()),0]


    """Will check if there's someone with a score equal to or higher than 100. If so then the winner will be printed in the console
    and the end variable will be set to true and the gameloop will break out of the while loop. If there's sudden death then 
    scoresheet will be replaced by the players going to the next round untill there's one winner"""
    def endstate(self,scoresheet):
        winners=[]
        losers=[]
        maxscore=0
        for i in range(1,len(scoresheet)+1):
            winners.append([i,scoresheet.get(i)[1],scoresheet.get(i)[0]])
            if scoresheet.get(i)[1]>maxscore:
                maxscore=scoresheet.get(i)[1]
        if maxscore<100:
            return
        for x in winners:
            if x[1]!=maxscore:
                losers.append(x)
        for x in losers:
            winners.remove(x)
    
        if len(winners)>1:
            print("The following players will now move on to the sudden death round:")
            for x in winners:
                print(scoresheet.get(x[0])[0])
            scoresheet={}
            i=1
            for  x in winners:
                scoresheet[i]=[x[2],x[1]]
                i+=1
            self.scoresheet = scoresheet
        else:
            self.end = True
            print("The winner is "+str(scoresheet.get(winners[0][0])[0]))

    

            
            