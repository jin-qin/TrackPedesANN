import time
import os


log_cache = []
log_name = "log"

def logSetName(pName):
    global log_name
    log_name = pName

def log(pMsg, pConsole=True, pFile=True):
    global log_cache

    msg = time.strftime('%X') + ": " + pMsg
    if pConsole:
        print(msg)
    if pFile:
        log_cache.append(time.strftime('%x') + ' ' + msg)


def logSave(directory, net=None):
    global log_cache, log_name

    #create dir if it does not exist yet
    if not os.path.exists(directory):
        os.makedirs(directory)

    if net is None:
        prefix = '{}'.format( time.time())
    else:
        prefix = net.get_session_name()

    #write to file
    f = open(directory + '/' + prefix + '-' + log_name + '.txt', 'w')
    f.write('\n'.join(log_cache))
    f.close()

    #automatically clear log after file save
    logClear()

def logClear():
    global log_cache
    log_cache = []