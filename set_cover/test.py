from generator import Generator
from set_cover import SetCover
from datetime import datetime

# test generator
def test_generator():
    g = Generator()
    print ('points: ' + str(g.X))
    print ('sets:')
    for s in g.F:
        print(s)

# test greedy algorithm for set cover
def test_set_cover(strategy='rounding'):
    #X_F_n_l = [100, 1000, 5000, 10000]
    X_F_n_l = [10000]
    print ('strategy: ' + strategy)
    for x in X_F_n_l:
        g = Generator(x, x)
        sc = SetCover(g.X, g.F, strategy=strategy)
        start = datetime.now()
        c = sc.start()
        end = datetime.now()
        s = set()
        for f in c: s = s.union(f)
        print ('\tset size: ' + str(x) + ' cover size: ' + str(len(c)))
        print ('\tpoints: '+str(x)+' cover points: ' + str(len(s)))
        #print ('points: ' + str(g.X))
        print ('\tEscape time: ' + str(end - start))
        print ('\t===================================')

#test_set_cover(strategy='greedy')
test_set_cover(strategy='rounding')
#test_set_cover(strategy='primal-dual')
