import numpy as np
eps = 0.0000001
def linear_select(s, k):
    ''' Select the k-th small number of s in linear time '''
    # Split s to n / 5 groups, sort and get median
    num = len(s)
    if num < 20:
        s = sorted(s)
        return s[k-1]
    group = list()
    m = list()  # median list
    group = np.reshape(s[:num//5*5], [num//5, 5])
    m = np.median(group, axis=1)
#    for i in  range(num//5):
#        _l = s[i*5:(i+1)*5]
#        _l = sorted(_l)
#        group.append(_l)
#        m.append(_l[2])
#    if num % 5 != 0: group.append(s[num//5*5:])
    # Recusive call, divide-and-conquer
    M = linear_select(m, len(m)//2)  # median
    s = np.array(s)
    lt_M = s[s<M]
    eq_M = s[s==M]
    gt_M = s[s>M]
    if len(lt_M) >= k: return linear_select(lt_M, k)
    elif len(lt_M)+len(eq_M)<k: return linear_select(gt_M,k-len(lt_M)-len(eq_M))
    else: return M

def median(s):
    ''' Return the median of s '''
    if len(s) % 2 == 0:
        return (linear_select(s, len(s)//2+1)+linear_select(s,len(s)//2))/2
    return linear_select(s, len(s)//2+1)

def calc_polar_angle(X, P, s):
    ''' Calculate the polar angle for s given origin X and axis X->P '''
    XP = P.minus(X) # vector X->P
    s_pa = np.arccos([XP.dot2d(p.minus(X))/(XP.norm2()*p.minus(X).norm2()+eps) for p in s]) 
    for j in range(len(s_pa)): # fix the polar angle large than pi based on cross
       s_pa[j] = s_pa[j] * 180 / np.pi
       if X.minus(s[j]).cross2d(XP)<0: s_pa[j] = 360 - s_pa[j]
       if s[j].X == X.X and s[j].Y == X.Y: s_pa[j] = 0
    return np.array(s_pa)

def insert_sort(s):
    for i in range(1,len(s)):
        key = s[i]
        j = i - 1
        while j >= 0 and s[j] > key:
            s[j+1] = s[j]
            j -= 1
        s[j+1] = key
    return s
