from generator import Generator
from collections import OrderedDict
from point import Point
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys

class ConvexHull(object):
    ''' Algorithms for solving convex hull problems '''

    def __init__(self, points=[], strategy='graham-scan'):
        '''
        parameters:
          points: points for find a convex hull
          strategy: different strategy for solving this problem
          available value:
            'brute-force': brute force method which own O(n^3)
            'graham-scan': graham scan algorithm which own O(nlgn)
            'divide-conqure': divide and conqure method which own O(nlgn)
        '''
        self.points = points
        self.strategy = strategy

    def __brute_force(self): 
        self.latest = np.array(self.points)
        num_points = len(self.latest)
        if num_points <= 3: return None
        # Find the least y coordinate
        for i in range(num_points):
            if self.latest[i].Y < self.latest[0].Y:
                tmp = self.latest[i]
                self.latest[i] = self.latest[0]
                self.latest[0] = tmp
        P0 = self.latest[0]
        mark = np.ones(num_points, np.int8)
        for i in range(1, num_points):
            for j in range(1, num_points):
                if i == j: continue
                for k in range(1, num_points):
                    if i == k or j == k: continue
                    if self.latest[i].in_triangle([self.latest[j],self.latest[k],P0]):
                        mark[i] = 0
                    if self.latest[j].in_triangle([self.latest[i],self.latest[k],P0]):
                        mark[j] = 0
                    if self.latest[k].in_triangle([self.latest[i],self.latest[j],P0]):
                        mark[k] = 0
        self.latest = self.latest[mark == 1]
        # Sort the point array
        self.latest = sorted(self.latest, key=lambda p:p.X)
        num_points = len(self.latest)
        if num_points > 0:
            X_min = self.latest[0]
            X_max = self.latest[-1]
            above_line = list()
            under_line = list()
            for p in self.latest:
                if p.lie_line([X_min, X_max]) > 0: above_line.append(p)
                else: under_line.append(p)
            under_line = sorted(under_line, key=lambda p:p.X)
            above_line = sorted(above_line, key=lambda p:p.X, reverse=True)
            self.latest = np.array(under_line + above_line)
        return self.latest

    def __graham_scan(self):
        self.latest = np.array(self.points)
        num_points = len(self.latest)
        if num_points <= 3: return None
        # Find the least y coordinate
        for i in range(num_points):
            if self.latest[i].Y < self.latest[0].Y:
                tmp = self.latest[i]
                self.latest[i] = self.latest[0]
                self.latest[0] = tmp
        P0 = self.latest[0]
        # Sorted by polar angle
        self.latest = sorted(self.latest[1:],key=lambda p:((p.X-P0.X)/p.dis(P0),-abs(p.X-P0.X)),reverse=True)
        # Remove the same polar angle, near P0 points
#        d = OrderedDict()
#        for p in self.latest:
#            d[(p.X-P0.X)/p.dis(P0)] = p
#        cur = 0
#        for key in d:
#            self.latest[cur] = d[key]
#            cur += 1
        # Scan
        num_points = len(self.latest)
        stack = list()
        stack.append(P0)
        stack.append(self.latest[0])
        stack.append(self.latest[1])
        for i in range(2, num_points):
            while P0.lie_line([self.latest[i], stack[-1]]) * \
                stack[-2].lie_line([self.latest[i], stack[-1]]) < 0:
                stack.pop()
            stack.append(self.latest[i])
        self.latest = np.array([P0] + stack)

    def start(self):
        start_time = datetime.now()
        if self.strategy == 'brute-force':
            self.__brute_force()
        elif self.strategy == 'graham-scan':
            self.__graham_scan()
        elif self.strategy == 'divide-conqure':
            pass
        else:
            sys.stderr.write('Invliad strategy\n')
            return None
        end_time = datetime.now()
        return end_time - start_time
        
    def show(self):
        ''' Show the latest convex hull '''
        # Show the result
        f = plt.figure('Convex Hull - ' + str(len(self.points)) + ' Points')
        pl = plt.scatter([p.X for p in self.points], 
                         [p.Y for p in self.points],
                         marker = '.',
                         color = 'b',
                         s = 1)
        
        for i in range(len(self.latest) - 1):
            A = self.latest[i]
            B = self.latest[i+1]
            plt.plot([A.X, B.X],
                     [A.Y, B.Y], linewidth=1, color='g')
        plt.plot([self.latest[0].X, self.latest[-1].X],
                 [self.latest[0].Y, self.latest[-1].Y],
                 linewidth=1, color='g')
        f.show()
