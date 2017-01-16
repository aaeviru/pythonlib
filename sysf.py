import sys
import os
from datetime import datetime, timedelta

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
    

    current_time_in_utc = datetime.utcnow()
    print current_time_in_utc + timedelta(hours=9)
    print
    inputform = inputform.split(',')
    for i in range(1,len(sys.argv)):
	print inputform[i-1]+':'+sys.argv[i]
    print
    sys.stdout.flush()
    return fout

def pend():
    print
    print datetime.utcnow() + timedelta(hours=9)
