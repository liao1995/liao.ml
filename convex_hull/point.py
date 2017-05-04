import math
import sys

class Point(object):
    '''Creates a point on a coordinate plane with values x and y.'''

    def __init__(self, x=0, y=0):
        '''Defines x and y variables'''
        self.X = x
        self.Y = y

    def __str__(self):
        return "(%s,%s)"%(self.X, self.Y)

    def minus(self, p):
        return Point(self.X - p.X, self.Y - p.Y)
    
    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def cross2d(self, p):
        ''' 2-D cross product '''
        res = self.X * p.Y - self.Y * p.X
        if res > 0: return 1
        if res == 0: return 0
        return -1

    def dis(self, p):
        ''' Return the Eculidean distance between this and p'''
        return math.sqrt((self.X - p.X)**2 + (self.Y - p.Y)**2)
        
    def in_triangle(self, triangle):
        ''' Test if this point in triangle, given the triangle as points array '''
        if len(triangle) < 3:
            sys.stderr.write('Invalid triangle\n')
            return None
        A = triangle[0]
        B = triangle[1]
        C = triangle[2]
        P = self
        # check if P lie in space between AB and AC
        PA_AB_cross = A.minus(P).cross2d(B.minus(A))
        PA_AC_cross = A.minus(P).cross2d(C.minus(A))
        if PA_AB_cross * PA_AC_cross > 0: return False
        # check if P lie in space between BA and BC
        PB_BA_cross = B.minus(P).cross2d(A.minus(B))
        PB_BC_cross = B.minus(P).cross2d(C.minus(B))
        if PB_BA_cross * PB_BC_cross > 0: return False
        # check if P lie in space between CA and CB
        PC_CA_cross = C.minus(P).cross2d(A.minus(C))
        PC_CB_cross = C.minus(P).cross2d(B.minus(C))
        if PC_CA_cross * PC_CB_cross > 0: return False
        return True

    def lie_line(self, line):
        ''' Return this point's relative position to line, 1 stands for up(left),
            0 stands for lie in line, -1 stands for down(right), line is points array '''
        if len(line) < 2:
            sys.stderr.write('Invalid line\n')
            return None
        res = line[0].minus(self).cross2d(line[1].minus(line[0]))
        if res > 0: return 1
        if res < 0: return -1
        return 0

Vector = Point

    
