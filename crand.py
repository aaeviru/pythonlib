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

def zipf_init(num):
    f = []
    for i in range(num):
        f.append(1.0/(i+1.0))
    f = np.array(f)
    f = f / f.sum()
    for i in range(1,num):
        f[i] = f[i] + f[i-1]
    return f
