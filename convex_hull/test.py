from generator import Generator
from convex_hull import ConvexHull
from point import Point
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# test generator
g = Generator()
a = g.next_batch(100)
#g.show()

c = ConvexHull(a, 'brute-force')
#c = ConvexHull(a, 'graham-scan')
time = c.start()
print('Escape time ' + str(time))
c.show()

##
### test in triangle
##triangle = [Point(10,10), Point(60,20), Point(100,80)]
##in_ = list()
##out_ = list()
##for p in a:
##    if p.in_triangle(triangle): in_.append(p)
##    else: out_.append(p)
##f = plt.figure('Test Triangle')
##plt.plot([triangle[0].X, triangle[1].X],
##         [triangle[0].Y, triangle[1].Y], linewidth=1, color='g')
##plt.plot([triangle[1].X, triangle[2].X],
##         [triangle[1].Y, triangle[2].Y], linewidth=1, color='g')
##plt.plot([triangle[2].X, triangle[0].X],
##         [triangle[2].Y, triangle[0].Y], linewidth=1, color='g')
##p1 = plt.scatter([p.X for p in in_],
##                 [p.Y for p in in_],
##                 marker = 'o',
##                 color = 'r',
##                 s = 1)
##p2 = plt.scatter([p.X for p in out_],
##                 [p.Y for p in out_],
##                 marker = 'x',
##                 color = 'b',
##                 s = 1)
##f.show()
