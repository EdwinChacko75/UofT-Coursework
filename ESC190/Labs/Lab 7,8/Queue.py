class Queue:
    def __init__(self):
        self.data = [None, None, 10, 12, None, None]
        self.start = 2
        self.end = 3

    def enqueue(self, item):
        if self.end < len(self.data)-1:
            self.end+=1
        elif self.end  == len(self.data)-1:
            self.end =0
        self.data[self.end] = item   # O(n) worst-case, usually O(1)


    def dequeue(self):
        ret_val = self.data[self.start]
        self.data[self.start] = None
        if self.start == len(self.data) -1:
            self.start = 0
        else:
            self.start +=1
        return ret_val    
    def __str__(self):
        print(self.data,f'{self.start}:{self.end}',sep='\n')

    def __lt__(self): 

        self.data = [str(x) for x in self.data]  
        self.data = sorted(self.data)  
        for i,x in enumerate(self.data):
            if x != "None":
                self.data[i] = int(x)
            else:
                self.data[i] = None
        for i in range(self.start):
            self.data.insert(0,None)
            self.data.pop()
            
        return self.data
        

q = Queue()
q.enqueue(5)
q.__str__()
q.enqueue(6)
q.__str__()

q.enqueue(7)
q.__str__()

q.dequeue()
q.__str__()
q.dequeue()
q.__str__()
q.dequeue()
q.__str__()
q.enqueue(90000)
q.__str__()
q.enqueue(1)
q.__str__()
q.enqueue(2)
q.__str__()
q.dequeue()
q.__str__()
q.dequeue()
q.__str__()

q.__lt__()
q.__str__()
