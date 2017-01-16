#!/usr/bin/env python
# -*- coding: utf-8 -*-
def coef(p,q):
    tmp = len([val for val in p if val in q])
    return 2.0*tmp/(len(p)+len(q))


def simatt(q,Pu,alpha = 0.5):
    cc = []
    for p in Pu:
        cc.append(coef(p,q))
    cc = sorted(cc)
    for i in range(1,len(cc)):
        cc[i] = alpha * cc[i] + (1-alpha) * cc[i-1]
    return cc[-1]
    
def sis(q,Pu):
    cc = set()
    for p in Pu:
        cc = cc.union([val for val in p if val in q])
    return cc

    
