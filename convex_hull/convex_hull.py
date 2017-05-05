from generator import Generator
from collections import OrderedDict
from point import Point
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import utils
import sys

class Pair(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

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
        ''' Graham scan algorithms, argument sort determine whether need to sort or not'''
        self.latest = self.__do_graham_scan(np.array(self.points))
        
    def __do_graham_scan(self, points, sort=True):
        ''' Graham scan algorithms, argument sort determine whether need to sort or not'''
        num_points = len(points)
        if num_points <= 3:
            if num_points < 3: return np.array(points)
            for i in range(1, num_points):
                if points[i].Y < points[0].Y:
                    tmp = points[i]
                    points[i] = points[0]
                    points[0] = tmp
            if points[2].lie_line([points[0], points[1]]) > 0:
                return np.array(points)
            else: return np.array([points[0],points[2],points[1]])
        # Find the least y coordinate
        for i in range(num_points):
            if points[i].Y < points[0].Y:
                tmp = points[i]
                points[i] = points[0]
                points[0] = tmp
        P0 = points[0]
        if sort:
            # Sorted by polar angle
            points = sorted(points[1:],key=lambda p:((p.X-P0.X)/p.dis(P0),-abs(p.X-P0.X)),reverse=True)
            # Remove the same polar angle, near P0 points
            d = OrderedDict()
            for p in points:
                d[(p.X-P0.X)/p.dis(P0)] = p
            cur = 0
            for key in d:
                points[cur] = d[key]
                cur += 1
        else:
            points = points[1:]
        # Scan
        num_points = len(points)
        stack = list()
        stack.append(P0)
        stack.append(points[0])
        stack.append(points[1])
        for i in range(2, num_points):
            while P0.lie_line([points[i], stack[-1]]) * \
                stack[-2].lie_line([points[i], stack[-1]]) < 0:
                stack.pop()
            stack.append(points[i])
        return np.array(stack)

    def __divide_conquer(self):
        self.latest = self.__do_divide_conquer(np.array(self.points))
        
    def __do_divide_conquer(self, P):
        if len(P) <= 3: return self.__do_graham_scan(P)
        # divide
        m = utils.median([p.X for p in P])
        PL = [p for p in P if p.X <= m]
        PR = [p for p in P if p.X > m]
        QL = self.__do_divide_conquer(PL)
        QR = self.__do_divide_conquer(PR)
        # calculate polar angle
        i = np.argmin([p.Y for p in QL])
        j = np.argmin([p.Y for p in QR])
        if QL[i].Y > QR[j].Y:
            tmp = QL
            QL = QR
            QR = tmp
            i = j
        X = Point(QL[i].X + 1, QL[i].Y)
        O = QL[i]
        QL_pa = utils.calc_polar_angle(O, X, QL)
        QR_pa = utils.calc_polar_angle(O, X, QR)
        s = np.argmin(QR_pa)  # min polar angle in QR
        t = np.argmax(QR_pa)  # max polar angle in QR
        
        # merge
        QL = np.concatenate((QL[i:], QL[:i]))   # arrange in ascending polar angle order
        QL_pa = np.concatenate((QL_pa[i:], QL_pa[:i]))
        if s < t:
            QR_1 = QR[s:t]
            QR_2 = np.concatenate((QR[t:], QR[:s]))[::-1]
            QR_pa_1 = QR_pa[s:t]
            QR_pa_2 = np.concatenate((QR_pa[t:], QR_pa[:s]))[::-1]
        else:
            QR_1 = np.concatenate((QR[s:], QR[:t]))
            QR_2 = QR[t:s][::-1]
            QR_pa_1 = np.concatenate((QR_pa[s:], QR_pa[:t]))
            QR_pa_2 = QR_pa[t:s][::-1]

        l_len = len(QL)
        r_len_1 = len(QR_1)
        r_len_2 = len(QR_2)
        l_it = 0
        r_it_1 = 0
        r_it_2 = 0
        
        W = list()
        while True:
            if l_it >= l_len:
                if r_it_1 >= r_len_1: # extend r_2
                    W.extend(QR_2[r_it_2:])
                    r_it_2 = r_len_2
                    break
                elif r_it_2 >= r_len_2:   # extend r_1
                    W.extend(QR_1[r_it_1:])
                    r_it_1 = r_len_1
                    break
                else:   # append r_1 and r_2
                    if QR_pa_1[r_it_1] < QR_pa_2[r_it_2]:
                        W.append(QR_1[r_it_1])
                        r_it_1 += 1
                    else:
                        W.append(QR_2[r_it_2])
                        r_it_2 += 1
            elif r_it_1 >= r_len_1:
                if r_it_2 >= r_len_2:   # extend l
                    W.extend(QL[l_it:])
                    l_it = l_len
                    break
                else:   # append l and r_2
                    if QL_pa[l_it] < QR_pa_2[r_it_2]:
                        W.append(QL[l_it])
                        l_it += 1
                    else:
                        W.append(QR_2[r_it_2])
                        r_it_2 += 1
            elif r_it_2 >= r_len_2:
                if QL_pa[l_it] < QR_pa_1[r_it_1]:
                    W.append(QL[l_it])
                    l_it += 1
                else:
                    W.append(QR_1[r_it_1])
                    r_it_1 += 1
            else:   # append l, r_1 and r_2
                if QL_pa[l_it] < QR_pa_1[r_it_1] and QL_pa[l_it] < QR_pa_2[r_it_2]:
                    W.append(QL[l_it])
                    l_it += 1
                elif QR_pa_1[r_it_1] < QL_pa[l_it] and QR_pa_1[r_it_1] < QR_pa_2[r_it_2]:
                    W.append(QR_1[r_it_1])
                    r_it_1 += 1
                else:
                    W.append(QR_2[r_it_2])
                    r_it_2 += 1
        return self.__do_graham_scan(W, sort=False)
        

    def start(self):
        start_time = datetime.now()
        if self.strategy == 'brute-force':
            self.__brute_force()
        elif self.strategy == 'graham-scan':
            self.__graham_scan()
        elif self.strategy == 'divide-conqure':
            self.__divide_conquer()
        else:
            sys.stderr.write('Invalid strategy\n')
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
