#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import re
import random
import numpy as np
from scipy import spatial
import crand

classname = '/home/ec2-user/git/tfidf/result/classname.txt'#literate classname
wordnet_wordlist = '/home/ec2-user/git/statresult/wordnet_wordlist.txt'#wordlist for wordnet
#fldawl = open('/home/ec2-user/git/statresult/wordslist_top10000_dsw.txt','r')#wordlist for lda
#i = 0
#ltow = {}
#for line in fldawl:
#    line = line.strip('\n')
#    ltow[i] = line
#    i = i + 1
#fldawl.close()

alpha = 0.01 #parameter for rd

def vecof0(lines,a,s,wtol,kk):
    if s == None:
        return vecof(lines,a,wtol,kk)
    else:
        return vecof3(lines,a,s,wtol,kk)


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

def vecof3(lines,a,s,wtol,kk):#cal lda topic-vec
    vec = 1
    for line in lines:
        line = line.strip('\n')
        if line in wtol:
            vec = vec * 1000.0 * a[:,wtol[line]] 
            vec = vec/vec.sum()
    vec = vec * 100000 * s
    return vec*10000/vec.sum()

def prq(lines,a,s,wtol,kk):#cal lda topic-vec
    vec = 1
    for line in lines:
        line = line.strip('\n')
        if line in wtol:
            vec = vec * 1000.0 * a[:,wtol[line]] 
    vec = vec * s
    return vec.sum()


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

def readcll0(cllfile,kk,stype):
    if stype in (2,3):
        return readcll(cllfile,kk)
    else:
        return readcll2(cllfile,kk)

def readcll(cllfile,kk):
    cll = {}
    if cllfile[0:4] == 'rand':
        if len(cllfile) > 4:
            dummylen = int(cllfile[4:])
        else:
            dummylen = 3
        for i in range(0,kk):
            cll[i] = np.random.randint(kk,size=dummylen)
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
    return cll

def readcll2(cllfile,kk):
    cll = {}
    if cllfile[0:4] == 'rand':
        if len(cllfile) > 4:
            dummylen = int(cllfile[4:])
        else:
            dummylen = 3
	fcl = open(classname,'r')
	acl = []
	for line in fcl:
	    line = line.strip(' \n')
	    line = line.split(' ')
	    for w in line:
		acl.append(w)
	for i in range(0,623):
	    tmpc = []
	    for j in range(dummylen):
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


def classof0(lines,a,s,wtol,kk):
    if s == None:
        return classof(lines,a,wtol,kk)
    else:
        return classof2(lines,a,s,wtol,kk)

def classof(lines,a,wtol,kk):#cal domain class of terms using lsa
    if len(lines) < 1:
        return -1
    vec = np.zeros(kk)
    for line in lines:
        line = line.strip('\n')
        vec = vec + a[:,wtol[line]]
    return vec.argmax()

def classof2(lines,a,s,wtol,kk):#cal domain class of terms using lda
    if len(lines) < 1:
        return -1
    vec = 1
    for line in lines:
        line = line.strip('\n')
        if line in wtol:
            vec = vec * a[:,wtol[line]] * 62300
    vec = vec * s
    return vec.argmax()

def classoft(filename,clpath,stype):
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
    if stype == 1:#tfidf2
        clf = clf + '2'
    return (cl,clf)

def simcheck(q,P):
    d = 2
    for p in P:
        sys.stdout.flush()
        tmpd = spatial.distance.cosine(p,q)
        if tmpd < d :
            d = tmpd
    return d

def dg(filename,cll,clpath,a = None,s = None,wtol = None,kk = None,zipf = 1.03,stype = 0):
#dummpy query generation using tfidf
#stype:0 tfidf/1 tfidf2/2 lsa/3 lda
#s for lda only
    fin  = open(filename+'.txt','r')
    lines = fin.readlines()
    fin.close()
    if stype == 0 or stype == 1:#tfidf
        cl,clf = classoft(filename,clpath,stype)
    elif stype == 2 or stype == 3:#lsa,lda
        cl = classof0(lines,a,s,wtol,kk)
	clf = clpath+'/'+str(cl)

    w = []
    fcl = open(clf,'r')
    tmp = fcl.readlines()
    fcl.close()
    for line in lines:
	if line in tmp[0:10000]:
	    w.append(tmp.index(line))
        else:
            print tmp.index(line)
    r = []
    t = '!'
    result = []
    #rq = np.array(w)
    #mean = rq.mean()
    #std = rq.std()
    #print mean,std
    if cl not in cll:
        cl = cl[:-1]
        if cl not in cll:
            cl = cl[0]
            if cl not in cll:
                return None

    ttmmpp = list(tmp)
    del tmp
    for tcl in cll[cl]:
	rr =  set()
        qtem = []
	#qlen = abs(len(w)+np.random.normal(0,2,1)) #random querry length
	qlen = len(w)

	if type(zipf) == float and zipf < 0 and stype == 3:
	    while len(rr) < qlen:
                dp = crand.randfunc(a[int(tcl)])
                if dp not in rr:
                    rr.add(dp)
                    qtem.append(ltow[int(dp)])
                else:
                    continue
            result.append(list(qtem))
            continue

        if stype == 0 or stype == 1:
            clf = clpath+'/'+tcl[0]+'/'+tcl+'.txt.fq.tfidfn'
            if stype == 1:
                clf = clf + '2'
        if stype == 2 or stype == 3:
            clf = clpath+'/'+str(tcl)
	fcl = open(clf,'r')
	tmp = fcl.readlines()[0:10000]
	fcl.close()

        if type(zipf) == float and zipf <= 1:
            for i in w:
		if i < len(tmp):
		    qtem.append(tmp[i].strip('\n'))
	else:
	    while len(rr) < qlen:
		dp = int(np.random.zipf(zipf,1))
                #dp = crand.randfunc2(zipf)
		if dp < len(tmp) and dp not in rr:
		    rr.add(dp)
		    qtem.append(tmp[int(dp)].strip('\n'))
		else:
		    continue
        result.append(list(qtem))
    qtem = []
    for i in w:
	tw = ttmmpp[i].strip('\n')
	qtem.append(tw)
	#t = t + tw + ' '
    result.append(list(qtem))
    x = range(len(result))
    random.shuffle(x)
    result = list(np.array(result)[x])
    result.append(str(x.index(len(result)-1)))
    return result


def dg2(filename,cll,clpath,a = None,s = None,wtol = None,kk = None,zipf = 1.03,P = None,stype = 0):
#dummpy query generation using tfidf
#stype:0 tfidf/1 tfidf2/2 lsa/3 lda
#s for lda only
    fin  = open(filename+'.txt','r')
    lines = fin.readlines()
    fin.close()
    if stype == 0 or stype == 1:#tfidf
        cl,clf = classoft(filename,clpath,stype)
    elif stype == 2 or stype == 3:#lsa,lda
        cl = classof0(lines,a,s,wtol,kk)
	clf = clpath+'/'+str(cl)

    w = []
    fcl = open(clf,'r')
    tmp = fcl.readlines()
    fcl.close()
    for line in lines:
	if line in tmp[0:10000]:
	    w.append(tmp.index(line))
    rd = simcheck(vecof0(lines,a,s,wtol,kk),P)
    rd = abs(rd + alpha * (np.random.random_sample() - 0.5))
    r = []
    t = '!'
    result = []
    if cl not in cll:
        return None
    ttmmpp = list(tmp)
    del tmp
    for tcl in cll[cl]:
        beta = alpha
        if stype == 0 or stype == 1:
            clf = clpath+'/'+tcl[0]+'/'+tcl+'.txt.fq.tfidfn'
            if stype == 1:
                clf = clf + '2'
        if stype == 2 or stype == 3:
            clf = clpath+'/'+str(tcl)
	fcl = open(clf,'r')
	tmp = fcl.readlines()[0:10000]
	fcl.close()
	#qlen = abs(len(w)+np.random.normal(0,2,1)) #random querry length
	qlen = len(w)
        sc = False
        it = 0 
        while sc == False:
            it = it + 1
            if it > 1000:
                it = 1
                beta = beta * 2
            rr =  set()
            qtem = []
            if type(zipf) == float and zipf < 0 and stype == 3:
                while len(rr) < qlen:
                    dp = crand.randfunc(a[int(tcl)])
                    if dp not in rr:
                        rr.add(dp)
                        qtem.append(ltow[int(dp)])
                    else:
                        continue
            else:
                while len(rr) < qlen:
                    dp = int(np.random.zipf(zipf,1))
                    #dp = crand.randfunc2(zipf)
                    if dp < len(tmp) and dp not in rr:
                        rr.add(dp)
                        qtem.append(tmp[int(dp)].strip('\n'))
                    else:
                        continue
            stm = simcheck(vecof0(qtem,a,s,wtol,kk),P)
            #print it,rd,stm
            if  abs(stm - rd) < beta * 0.5:
                sc = True
        result.append(list(qtem))
    qtem = []
    for i in w:
	tw = ttmmpp[i].strip('\n')
	qtem.append(tw)
	#t = t + tw + ' '
    result.append(list(qtem))
    #result.append(t)
    x = range(len(result))
    random.shuffle(x)
    result = list(np.array(result)[x])
    result.append(str(x.index(len(result)-1)))

    return result

def dg3(filename,cll,a = None,s = None,p = None,wtol = None,ltow = None,kk = None):
#dummpy query generation using lda
    fin  = open(filename+'.txt','r')
    lines = fin.readlines()
    fin.close()
    cl = classof2(lines,a,s,wtol,kk)

    w = []
    result = []
    if cl not in cll:
        return None
    for tcl in cll[cl]:
	#qlen = abs(len(w)+np.random.normal(0,2,1)) #random querry length
        qtem = []
	qlen = len(lines)
        while classof2(qtem,a,s,wtol,kk) != int(tcl):
            rr =  set()
            qtem = []
            while len(rr) < qlen:
                dp = crand.randfunc2(p[int(tcl)])
                if dp not in rr:
                    rr.add(dp)
                    qtem.append(ltow[int(dp)])
                else:
                    continue
            #print len(qtem),classof2(qtem,a,s,wtol,kk) , int(tcl)
        result.append(list(qtem))
    qtem = []
    for line in lines:
	qtem.append(line.strip('\n'))
    result.append(list(qtem))
    x = range(len(result))
    random.shuffle(x)
    result = list(np.array(result)[x])
    result.append(str(x.index(len(result)-1)))
    return result


def dg4(filename,cll,clpath,a = None,s = None,p = None,wtol = None,ltow = None,kk = None,P = None):
#dummpy query generation using tfidf
#stype:0 tfidf/1 tfidf2/2 lsa/3 lda
#s for lda only
    fin  = open(filename+'.txt','r')
    lines = fin.readlines()
    fin.close()
    cl = classof0(lines,a,s,wtol,kk)

    rd = simcheck(vecof0(lines,a,s,wtol,kk),P)
    rd = abs(rd + alpha * (np.random.random_sample() - 0.5))
    result = []
    if cl not in cll:
        return None
    for tcl in cll[cl]:
        beta = alpha
	qlen = len(lines)
        sc = False
        it = 0 
        while sc == False:
            it = it + 1
            if it > 1000:
                it = 1
                beta = beta * 2
            rr =  set()
            qtem = []
            while classof2(qtem,a,s,wtol,kk) != int(tcl):
                while len(rr) < qlen:
                    dp = crand.randfunc(a[int(tcl)])
                    if dp not in rr:
                        rr.add(dp)
                        qtem.append(ltow[int(dp)])
                    else:
                        continue
            stm = simcheck(vecof0(qtem,a,s,wtol,kk),P)
            if  abs(stm - rd) < beta * 0.5:
                sc = True
        result.append(list(qtem))
    qtem = []
    for line in lines:
	qtem.append(line.strip('\n'))
    result.append(list(qtem))
    x = range(len(result))
    random.shuffle(x)
    result = list(np.array(result)[x])
    result.append(str(x.index(len(result)-1)))

    return result

def dg5(filename,clpath,cl = None,clf = None,dummylen = 3,a = None,s = None,wtol = None,kk = None,stype = 0,dtype = 1):
    if type(filename) == str:
        fin  = open(filename+'.txt','r')
        lines = fin.readlines()
        fin.close()
    elif type(filename) == list:
        lines = filename
    else:
        return None

    if cl == None and clf == None:
        if stype == 0 or stype == 1:#tfidf
            cl,clf = classoft(filename,clpath,stype)
        elif stype == 2 or stype == 3:#lsa,lda
            cl = classof0(lines,a,s,wtol,kk)
            clf = clpath+'/'+str(cl)

    w = []
    fcl = open(clf,'r')
    tmp = fcl.readlines()
    fcl.close()
    for line in lines:
	if line in tmp[0:10000]:
	    w.append(tmp.index(line))
    w = sorted(w)
    r = []
    t = '!'
    result = []
    for ii in range(dummylen):
	rr =  set()
        qtem = []
	qlen = len(w)
        if dtype > 1:
            for i in w:
		if i < len(tmp):
		    qtem.append(tmp[i].strip('\n'))
        else:    
            for i in w:
                dp = i + random.randint(-1 * 2,2)
                while dp < 0 or dp > len(tmp) or dp in rr:
                    #print 'w:',w
                    #print dp
                    dp = i + random.randint(-1 * 2,2)
                rr.add(dp)
                qtem.append(tmp[int(dp)].strip('\n'))
        result.append(list(qtem))
    qtem = []
    for i in w:
	tw = tmp[i].strip('\n')
	qtem.append(tw)
    result.append(list(qtem))
    x = range(len(result))
    random.shuffle(x)
    result = list(np.array(result)[x])
    result.append(str(x.index(dummylen)))
    return result

def dg6(filename,clpath,cl = None,clf = None,ti = None,dummylen = 3,a = None,s = None,wtol = None,kk = None,stype = 0,dtype = 1):
    if type(filename) == str:
        fin  = open(filename+'.txt','r')
        lines = fin.readlines()
        fin.close()
    elif type(filename) == list:
        lines = filename
    else:
        return None

    if cl == None and clf == None:
        if stype == 0 or stype == 1:#tfidf
            cl,clf = classoft(filename,clpath,stype)
        elif stype == 2 or stype == 3:#lsa,lda
            cl = classof0(lines,a,s,wtol,kk)
            clf = clpath+'/'+str(cl)

    w = []
    fcl = open(clf,'r')
    tmp = fcl.readlines()
    fcl.close()
    for line in lines:
	if line in tmp[0:10000]:
	    w.append(tmp.index(line))
    w = sorted(w)
    r = []
    t = '!'
    result = []
    for ii in range(dummylen):
	rr =  set()
        qtem = []
	qlen = len(w)
        if dtype > 1:
            for i in w:
		if i < len(tmp):
		    qtem.append(tmp[i].strip('\n'))
        else:    
            for i in w:
                if tmp[i].strip('\n') in ti:
                    dp = i
                else:
                    dp = i + random.randint(-1 * dummylen,dummylen)
                    while dp < 0 or dp == i or dp > len(tmp) or dp in rr:
                        #print 'w:',w
                        #print dp
                        dp = i + random.randint(-1 * dummylen,dummylen)
                rr.add(dp)
                qtem.append(tmp[int(dp)].strip('\n'))
        result.append(list(qtem))
    qtem = []
    for i in w:
	tw = tmp[i].strip('\n')
	qtem.append(tw)
    result.append(list(qtem))
    x = range(len(result))
    random.shuffle(x)
    result = list(np.array(result)[x])
    result.append(str(x.index(dummylen)))
    return result

def dg7(filename,clpath,cl = None,clf = None,dummylen = 3,a = None,s = None,wtol = None,kk = None,stype = 0,dtype = 1):
    if type(filename) == str:
        fin  = open(filename+'.txt','r')
        lines = fin.readlines()
        fin.close()
    elif type(filename) == list:
        lines = filename
    else:
        return None

    if cl == None and clf == None:
        if stype == 0 or stype == 1:#tfidf
            cl,clf = classoft(filename,clpath,stype)
        elif stype == 2 or stype == 3:#lsa,lda
            cl = classof0(lines,a,s,wtol,kk)
            clf = clpath+'/'+str(cl)

    w = []
    fcl = open(clf,'r')
    tmp = fcl.readlines()
    fcl.close()
    for line in lines:
	if line in tmp[0:10000]:
	    w.append(tmp.index(line))
    w = sorted(w)
    r = []
    t = '!'
    result = []
    offset = random.randint(-1 * dummylen/2,dummylen/2)
    for ii in range(dummylen+1):
	rr =  set()
        qtem = []
	qlen = len(w)
        if dtype > 1:
            for i in w:
		if i < len(tmp):
		    qtem.append(tmp[i].strip('\n'))
        else:    
            if offset + ii - dummylen/2 == 0:
                continue
            for i in w:
                dp = i + offset + ii - dummylen/2
                if dp < 0:
                    dp == 0
                qtem.append(tmp[int(dp)].strip('\n'))
        result.append(list(qtem))
    qtem = []
    for i in w:
	tw = tmp[i].strip('\n')
	qtem.append(tw)
    result.append(list(qtem))
    x = range(len(result))
    random.shuffle(x)
    result = list(np.array(result)[x])
    result.append(str(x.index(dummylen)))
    return result
