from point import Point
import random as rand
import matplotlib.pyplot as plt
import numpy as np
import sys

class Generator(object):
    ''' Random generate some points on a coordinate plane within an rectangle '''
    
    def __init__(self, startX=0, startY=0, width=100, height=100):
        self.startX = startX
        self.startY = startY
        self.width = width
        self.height = height
        # store the latest generated batch of points 
        self.latest = []    

    def next(self):
        ''' Generate the next random point '''
        x = self.startX + self.width * rand.random()
        y = self.startY + self.height * rand.random()
        return Point(x, y)

    def next_batch(self, batch_size=1000):
        ''' Generate a batch of points '''
        l = list()
        if batch_size < 0:
            sys.stderr.write('Batch size can not be negative\n')
            return None
        for i in range(batch_size):
            l.append(self.next())
        self.latest = np.array(l)
        return self.latest

    def show(self):
        ''' Show the latest generated batch of points '''
        f = plt.figure(str(len(self.latest)) + ' Points')
        pl = plt.scatter([p.X for p in self.latest], 
                         [p.Y for p in self.latest],
                         marker = 'o',
                         color = 'r',
                         s = 1)
        f.show()
