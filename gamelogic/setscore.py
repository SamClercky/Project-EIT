class MyScore():
    """The class function will update the scoresheet, scale the outputs for the score, for the height of the ball and led lights."""
    def scale(self, array: list):
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
            if 45-x[0]*5>= 0:
                arrayscale.append(45-x[0]*5)
            else:
                arrayscale.append(0)
        return arrayscale

    def update(self,scoresheet: dict, newscore: list):
        for i in range(1,len(newscore)+1):
            scoresheet.get(i)[1]=scoresheet.get(i)[1]+newscore[i-1]
    
    def demoscale(self, array: list):
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
            if 55-x[0]*5>= 0:
                arrayscale.append(55-x[0]*5)
            else:
                arrayscale.append(0)
        return arrayscale
    
    def heightscale(self, scoresheet: dict, end: bool):
        array=[]
    
        for i in range(1,len(scoresheet)+1):
            if(end):
                array.append(int((((scoresheet.get(i)[1]-100)/100)*175)+80))
            else:
                array.append(int(((scoresheet.get(i)[1]/100)*175)+80))

        print(array)
        return array

    def lightscale(self,scoresheet: dict,end: bool):
        array=[]
        for i in range(1,len(scoresheet)+1):
            if(end):
                array.append(int(((scoresheet.get(i)[1]-100)/100)*80))
            else:
                array.append(int((scoresheet.get(i)[1]/100)*80))

        print(array)
        return array
                