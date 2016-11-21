import numpy as np

def randfunc(p):
    f = np.array(p)
    f = f / f.sum()
    for i in range(1,len(f)):
        f[i] = f[i] + f[i-1]
    r = np.random.rand()
    return np.searchsorted(f,r)

def randfunc2(f):
    f = np.array(f)
    r = np.random.rand()
    return np.searchsorted(f,r)
