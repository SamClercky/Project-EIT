class MyScore():
    """The class function gets a list of players, which will be put in a dictionary with the keys representing the player numbers
    or indexes and the value is a list with the first element the player name and the second one their score."""
    def scale(self, array):
        newscore=[]
        for x in array:
            newscore.append([0,x])
        array=set(array)
        array=list(array)
        array.sort()
        arrayscale=[]
        for x in newscore:
            i=0
            for y in array:
                if x[1]==y:
                    x[0]=i+1
                i+=1
        for x in newscore:
            if 40-x[0]*5>= 0:
                arrayscale.append(35-x[0]*5)
            else:
                arrayscale.append(0)
        print(newscore)
        return arrayscale

    def update(self,scoresheet, newscore):
        for i in range(1,len(newscore)+1):
            scoresheet.get(i)[1]=scoresheet.get(i)[1]+newscore[i-1]
             