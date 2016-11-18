import sys
import os
from time import gmtime, strftime

def logger(outf,inputform):
    if outf[0:6] == 'stdout':
        return sys.stdout
    outft = outf
    i = 0
    while os.path.isfile(outf+'.txt'):
	i = i + 1
	outf = outft+'-v'+str(i)
    fout = open(outf+'.txt','w',0)
    flog = open(outf+'.log', 'w',0)
    sys.stdout = flog
    sys.stderr = flog

    inputform = inputform.split(',')
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    for i in range(1,len(sys.argv)):
	print inputform[i-1]+':'+sys.argv[i]
    print
    sys.stdout.flush()
    return fout


