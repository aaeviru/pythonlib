import sys
import os
from time import gmtime, strftime

def logger(outf,inputform):
    outft = outf
    i = 0
    while os.path.isfile(outf+'.txt'):
	i = i + 1
	outf = outft+'-v'+str(i)
    fout = open(outf+'.txt','w')
    flog = open(outf+'.log', 'w')
    sys.stdout = flog
    sys.stderr = flog

    inputform = inputform.split(',')
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    for i in range(1,len(sys.argv)):
	print inputform[i-1]+':'+sys.argv[i]
    print
    return fout


