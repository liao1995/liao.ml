from generator import Generator
from convex_hull import ConvexHull
from point import Point
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# test generator
def test_generator(points=100):
    g = Generator()
    a = g.next_batch(points)
    g.show()

# test calculate polar angle
def test_calc_polar_angle(points=50):
    from utils import calc_polar_angle
    g = Generator()
    a = g.next_batch(points)
    X = Point(50,50)
    P = Point(75,75)
    s_pa = calc_polar_angle(X, P, a)
    print(s_pa)
    f = plt.figure('Test Polar Angle')
    p1 = plt.scatter([p.X for p in a],
                     [p.Y for p in a],
                     marker = '.',
                     color = 'r',
                     s = 1)
    for i in range(len(a)):
        plt.annotate("{:.1f}".format(s_pa[i]), [a[i].X,a[i].Y])
    plt.plot([50,100],[50,100], color='g', linewidth=2)
    f.show()

# test convex hull
def test_convex_hull(stratery='graham-scan', points=10000):
    g = Generator()
    a = g.next_batch(points)
    c = ConvexHull(a, stratery)
    time = c.start()
    print('Escape time ' + str(time))
    c.show()
    
#test_convex_hull('divide-conqure', 10000)
#test_convex_hull(points=1000)

# test insert sort
def test_insert_sort(points=10000):
    from utils import insert_sort
    from datetime import datetime
    import numpy.random as rand
    s = rand.randint(1,100000000,points).tolist()
    start = datetime.now() 
    insert_sort(s)
    end = datetime.now()
    print ('insert sort escape time: ' + str(end-start))
    start = datetime.now() 
    sorted(s)
    end = datetime.now()
    print ('sorted escape time: ' + str(end-start))


# test linear select
def test_linear_select(points=10000000):
    from utils import linear_select
    import numpy.random as rand
    import numpy as np
    from datetime import datetime
    s = rand.randint(1,100000000,points).tolist()
    k = 3
    print ('len(s) = ' + str(len(s)) + ' k = ' + str(k))
    start = datetime.now()
    res = linear_select(s, k)
    end = datetime.now()
    print(' linear select time: ' + str(end-start))
    start = datetime.now()
    s = sorted(s)
    end = datetime.now()
    print(' sort time: ' + str(end-start))
    start = datetime.now()
    m = np.median(s)
    end = datetime.now()
    print(' median time: ' + str(end-start))


# test in triangle
def test_in_triangle(points=1000):
    g = Generator()
    a = g.next_batch(points)
    triangle = [Point(10,10), Point(60,20), Point(100,80)]
    in_ = list()
    out_ = list()
    for p in a:
        if p.in_triangle(triangle): in_.append(p)
        else: out_.append(p)
    f = plt.figure('Test Triangle')
    plt.plot([triangle[0].X, triangle[1].X],
         [triangle[0].Y, triangle[1].Y], linewidth=1, color='g')
    plt.plot([triangle[1].X, triangle[2].X],
         [triangle[1].Y, triangle[2].Y], linewidth=1, color='g')
    plt.plot([triangle[2].X, triangle[0].X],
         [triangle[2].Y, triangle[0].Y], linewidth=1, color='g')
    p1 = plt.scatter([p.X for p in in_],
                 [p.Y for p in in_],
                 marker = 'o',
                 color = 'r',
                 s = 1)
    p2 = plt.scatter([p.X for p in out_],
                 [p.Y for p in out_],
                 marker = 'x',
                 color = 'b',
                 s = 1)
    f.show()

