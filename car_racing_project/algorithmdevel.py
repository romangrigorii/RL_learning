import numpy as np
from itertools import compress

def distance_estimate(d):
    q_tot = 0
    q_d = 128
    sgn = 1
    hit = 1
    while abs(q_d)>0:
        q_tot += (sgn*q_d)
        if q_tot>d:
            hit = 1
            sgn = -1
            q_d //=2
        else:
            if not hit:
                q_d*=2
            else:
                sgn = 1
                q_d/=2
    return q_tot

print(distance_estimate(95))


a = [1,2,3,4,5]
b = [0,0,0,1,0]

def extract_list(a,b):
    out = []
    for i, q in enumerate(b):
        if q!=0: out.append(a[i])
    return out

print(list(compress(a, b)))