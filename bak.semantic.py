#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import re
import numpy as np

classname = '/home/ec2-user/git/tfidf/result/classname.txt'#literate classname

def vecof(lines,a,wtol,kk):#cal lsa topic-vec of terms
    vec = np.zeros(kk)
    for line in lines:
        line = line.strip('\n')
        vec = vec + a[:,wtol[line]]
    return vec


def vecof2(lines,idf,a,wtol,kk):#cal lsa topic-vec of weighted(tfidf) terms
    vec = np.zeros(kk)
    for line in lines:
        line = line.strip('\n')
        line = line.split(' ')
        if line[1] in wtol:
            vec = vec + a[:,wtol[line[1]]] * idf[line[1]] * int(line[0])
    return vec/np.linalg.norm(vec)

def vecof3(lines,a,s,wtol,kk):#cal domain class of terms using lda
    vec = 1
    for line in lines:
        line = line.strip('\n')
        vec = vec * a[:,wtol[line]] * 62300
    vec = vec * s * 100000
    return vec

def readwl(wlpath):#read word-lsanum matrix
    fwl = open(wlpath,"r")
    wtol = {}
    itow = {}
    i = 0
    for line in fwl:
        line = line.strip('\n')
        wtol[line] = i
        i = i + 1
    fwl.close()
    return wtol

def readcll(cllfile):
    cll = {}
    if cllfile == 'rand':
        for i in range(0,kk):
            cll[i] = np.random.randint(kk,size=3)
    else:
        fcl = open(cllfile,'r')
        for line in fcl:
            line = line.strip(' \n')
            line = line.split(' ')
            for w in line:
                ww = int(w)
                cll[ww] = list(line)
                cll[ww].remove(w)
        fcl.close()

def readcll2(cllfile):
    cll = {}
    if cllfile == 'rand':
	fcl = open(classname,'r')
	acl = []
	for line in fcl:
	    line = line.strip(' \n')
	    line = line.split(' ')
	    for w in line:
		acl.append(w)
	for i in range(0,623):
	    tmpc = []
	    for j in range(3):
		tmpc.append(acl[np.random.randint(623)])
	    cll[acl[i]] = list(tmpc)
    else:
	fcl = open(cllfile,'r')
	for line in fcl:
	    line = line.strip(' \n')
	    line = line.split(' ')
	    for w in line:
		cll[w] = list(line)
		cll[w].remove(w)
	fcl.close()
    return cll



def classof(lines,a,wtol,kk):#cal domain class of terms using lsa
    vec = np.zeros(kk)
    for line in lines:
        line = line.strip('\n')
        vec = vec + a[:,wtol[line]]
    return vec.argmax()

def classof2(lines,a,s,wtol,kk):#cal domain class of terms using lda
    vec = 1
    for line in lines:
        line = line.strip('\n')
        vec = vec * a[:,wtol[line]]
    vec = vec * s
    return vec.argmax()

def dg(root,name,cll,clpath,zipf,a,wtol,kk):#gen dummy query
    filename = root + '/' + name
    w = []
    fin  = open(filename+'.txt','r')
    lines = fin.readlines()
    fin.close()
    cl = classof(lines,a,wtol,kk)
    fcl = open(clpath+'/'+str(cl))
    tmp = fcl.readlines()
    fcl.close()
    for line in lines:
        if line in tmp[0:10000]:
            w.append(tmp.index(line))
    rq = np.array(w)
    mean = rq.mean()
    std = rq.std()
    #print mean,std
    if cl not in cll:
        return None
    ttmmpp = list(tmp)
    del tmp
    result = []
    for tcl in cll[cl]:
        fcl = open(clpath+'/'+str(tcl))
        tmp = fcl.readlines()    
        fcl.close()
        rr =  set()
        q = []
        #qlen = abs(len(w)+np.random.normal(0,2,1))
        qlen = len(w)
        while len(rr) < qlen:
            dp = int(np.random.zipf(zipf,1))
            if dp < len(tmp) and dp not in rr:
                rr.add(dp)
                q.append(tmp[int(dp)].strip('\n'))
            else:
                continue
        result.append(q)
    q = []
    t = '!'
    for i in w:
        tw = ttmmpp[i].strip('\n')
        q.append(tw)
        t = t + tw + ' '
    result.append(q)
    result.append(t)
    return result


def dg2(filename,cll,clpath,a = None,s = None,wtol = None,kk = None,zipf = 1.03,type = 0):
#dummpy query generation using tfidf
    fin  = open(filename+'.txt','r')
    lines = fin.readlines()
    fin.close()
    if type == 0 or type == 1:#tfidf
        fin = open(filename,'r')
        temp = fin.read()
        fin.close()
        cl = re.findall(r'【国際特許分類第.*版】.*?([A-H][0-9]+?[A-Z])',temp,re.DOTALL)
        if(len(cl) < 1):
            print 'cl<1:',filename
            return None
        cl = cl[0]
        cl = cl[0] + str(int(cl[1:len(cl)-1])) +cl[len(cl)-1]
	clf = clpath+'/'+cl[0]+'/'+cl+'.txt.fq.tfidfn'
	if type == 1:#tfidf2
	    clf = clf + '2'
    elif type == 2 or type == 3:#lsa,lda
        cl = classof(lines,a,wtol,kk)
	clf = clpath+'/'+str(cl)

    w = []
    fcl = open(clf,'r')
    tmp = fcl.readlines()
    fcl.close()
    for line in lines:
	if line in tmp[0:10000]:
	    w.append(tmp.index(line))
    r = []
    t = '!'
    result = []
    #rq = np.array(w)
    #mean = rq.mean()
    #std = rq.std()
    #print mean,std
    if cl not in cll:
        return None
    ttmmpp = list(tmp)
    del tmp
    for tcl in cll[cl]:
        if type == 0 or type == 1:
            clf = clpath+'/'+cl[0]+'/'+cl+'.txt.fq.tfidfn'
            if type == 1:
                clf = clf + '2'
        if type == 2 or type == 3:
            cl = classof(lines,a,wtol,kk)
            clf = clpath+'/'+str(cl)
	fcl = open(clf,'r')
	tmp = fcl.readlines()[0:10000]
	fcl.close()

	rr =  set()
        qtem = []
	#qlen = abs(len(w)+np.random.normal(0,2,1)) #random querry length
	qlen = len(w)
        if zipf <= 1:
            for i in w:
		if i < len(tmp):
		    qtem.append(tmp[i].strip('\n'))
	else:
	    while len(rr) < qlen:
		dp = int(np.random.zipf(zipf,1))
		if dp < len(tmp) and dp not in rr:
		    rr.add(dp)
		    qtem.append(tmp[int(dp)].strip('\n'))
		else:
		    continue
        result.append(list(qtem))
    for i in w:
	tw = ttmmpp[i].strip('\n')
	qtem.append(tw)
	t = t + tw + ' '
    result.append(list(qtem))
    result.append(t)
    return result

def dg3(root,name,cll,clpath,zipf,a,s,wtol,kk):#gen dummy query using lda
    filename = root + '/' + name
    w = []
    fin  = open(filename+'.txt','r')
    lines = fin.readlines()
    fin.close()
    cl = classof2(lines,a,s,wtol,kk)
    fcl = open(clpath+'/'+str(cl))
    tmp = fcl.readlines()
    fcl.close()
    for line in lines:
        if line in tmp[0:10000]:
            w.append(tmp.index(line))
    #rq = np.array(w)
    #mean = rq.mean()
    #std = rq.std()
    #print mean,std
    if cl not in cll:
        return None
    ttmmpp = list(tmp)
    del tmp
    result = []
    for tcl in cll[cl]:
        fcl = open(clpath+'/'+str(tcl))
        tmp = fcl.readlines()    
        fcl.close()
        rr =  set()
        q = []
        #qlen = abs(len(w)+np.random.normal(0,2,1))
        qlen = len(w)
        while len(rr) < qlen:
            dp = int(np.random.zipf(zipf,1))
            if dp < len(tmp) and dp not in rr:
                rr.add(dp)
                q.append(tmp[int(dp)].strip('\n'))
            else:
                continue
        result.append(q)
    q = []
    t = '!'
    for i in w:
        tw = ttmmpp[i].strip('\n')
        q.append(tw)
        t = t + tw + ' '
    result.append(q)
    result.append(t)
    return result

