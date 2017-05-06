import sys
from scipy.optimize import linprog
import numpy as np

class SetCover(object):
    ''' For solving Set Cover problem '''

    def __init__(self, X, F, strategy = 'greedy'):
        ''' X - point set  F - subset list
            valid strategy value:
            'greedy': using greedy algorithm
            'rounding': rounding method using linear programming
            '''
        if len(X) <= 0 or len(F) <= 0:
            sys.stderr.write('Invalid X or F\n')
            return
        s = set()
        for f in F: s = s.union(f)
        if not X.issubset(s):
            sys.stderr.write('No valid solution\n')
            return
        self.X = X
        self.F = F
        self.strategy = strategy

    def start(self):
        if self.strategy == 'greedy':
            self.__greedy()
        elif self.strategy == 'rounding':
            self.__rounding()
        else:
            sys.stderr.write('Invalid strategy\n')
            return None
        return self.C

    def __greedy(self):
        U = self.X
        self.C = list()
        while len(U) > 0:
            max_cover = 0
            max_f = set()
            for f in self.F:
                inter = f.intersection(U)
                if len(inter) > max_cover:
                    max_cover = len(inter)
                    max_f = f
            U = U - max_f
            self.C.append(max_f)

    def __rounding(self):
        ''' linear programming method
            goal: min Z =  x1 + x2 + ... + xn
            expression:
               A(F1, e1)x1 + A(F2, e1)x2 + ... + A(Fn, e1)xn  >=  1
               A(F1, e2)x1 + A(F2, e2)x2 + ... + A(Fn, e2)xn  >=  1
               ......
               A(F1, em)x1 + A(F2, em)x2 + ... + A(Fn, em)xn  >=  1
            using scipy linear programming package, transfer to:
             - A(F1, e1)x1 - A(F2, e1)x2 - ... - A(Fn, e1)xn  <= -1
             - A(F1, e2)x1 - A(F2, e2)x2 - ... - A(Fn, e2)xn  <= -1
             ......
             - A(F1, em)x1 - A(F2, em)x2 - ... - A(Fn, em)xn  <= -1
            such that
                          0 <= x1, x2, ...., xn <= 1
        '''
        m = len(self.X) # number of points
        n = len(self.F) # number of subsets
        # goal
        c = np.ones(n)
        # expression coefficent
        A = np.array([[e in f for f in self.F] for e in self.X],dtype=np.float32)
        f = np.max(np.sum(A, axis=1))
        A_ub = -A
        b_ub = -np.ones(m)
        bounds = [(0, 1)] * n   # bounds for x1 .... xn
        # linear programming
        res = linprog(c, A_ub, b_ub, bounds=bounds, options={"disp": True})
        # rounding
        threshold = 1 / f
        self.C = [self.F[i] for i in range(n) if res.x[i] > threshold]
            
         
