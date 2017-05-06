import numpy as np
import random as rand
import utils
import sys

class Generator(object):
    ''' Random generator the points and set '''

    def __init__(self, X_n=100, F_n=100):
        ''' X_n - number of points  F_n - Set size '''
        if X_n <=20 or F_n <= 0:
            sys.stderr.write('Invalid size of points or set\n')
            return 
        # generate points
        self.X = set(range(X_n))
        self.F = list()
        s_union = set()                 # current sampled points union
        # generate set
        s0 = utils.sample(self.X, 20)   # first set
        self.F.append(s0)
        s_union = s_union.union(s0)
        while len(self.X - s_union) >= 20:
            n = rand.randint(1, 20)
            x = rand.randint(1, n)
            si = utils.sample(self.X-s_union,x).union(utils.sample(s_union,n-x))
            self.F.append(si)
            s_union = s_union.union(si)
            if len(self.F) == F_n - 1:  # enough set
                self.F.append(self.X - s_union)
                s_union = self.X
                break
        if len(self.X - s_union) > 0:
            self.F.append(self.X - s_union)
        while len(self.F) < F_n:   # the rest
            self.F.append(utils.sample(self.X, rand.randint(1,20)))
        
    def getX(self):
        return self.X

    def getF(self):
        return self.F
