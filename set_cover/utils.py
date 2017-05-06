import random as rand

def sample(S, num = 10):
    ''' Random select num different points from set S,
        return the sampled set '''
    return set(rand.sample(S, num))

def select(S):
    ''' Random select one point from set S '''
    return rand.sample(S, 1)[0]
