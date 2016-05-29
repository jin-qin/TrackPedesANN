import time
import os


log_cache = []
log_name = "log"

def logSetName(pName):
    global log_name
    log_name = pName

def log(pMsg, pConsole=True, pFile=True):
    msg = time.strftime('%X') + ": " + pMsg
    if pConsole:
        print(msg)
    if pFile:
        log_cache.append(time.strftime('%x') + ' ' + msg)


def logSave(directory):

    #create dir if it does not exist yet
    if not os.path.exists(directory):
        os.makedirs(directory)

    #write to file
    f = open(directory + '/{}'.format( time.time()) + '-' + log_name + '.txt', 'w')
    f.write('\n'.join(log_cache))
    f.close()

    #automatically clear log after file save
    logClear()

def logClear():
    log_cache.clear()