"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 20 Dec 2020

@author: frederic
"""
__updated__ = "2021-08-19 12:56:06"

import time
import string
import random
import logging

#===============================================================================
#
#    setup logging
#
logit = dict([(lvl, True) for lvl in logging._nameToLevel])

logging.basicConfig(
    level=logging.DEBUG, 
    filename='rfCommon.log',
    filemode = 'w',
    format='%(levelname)-8s - %(filename)-20s %(funcName)-15s'
           ' [%(lineno)5d]: %(message)s'
)

tLogger = logging.getLogger('pyRFtk2')
tLogger.info(f'{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())} -- '
             f'logging level {logging.getLevelName(logging.getLogger().level)}')

#=-----------------------------------------------------------------------------#

def setLogLevel(level):
    
    newLvl = level if isinstance(level, int) else logging._nameToLevel(level)
    
    logit['DEBUG'] and tLogger.debug(
        f'Change loglevel from {logging.getLevelName(logging.getLogger().level)} '
        f'to {logging.getLevelName(newLvl)}')
    
    tLogger.setLevel(newLvl)
    
    for lvl, name in logging._levelToName.items():
        logit[name] = tLogger.isEnabledFor(lvl)
    
    if logit['DEBUG']:
        tLogger.debug('SetLogLevel: DEBUG')
        for s in logit:
            tLogger.debug(f'   {s:10}  {logit[s]}   {logging._nameToLevel[s]}')

#=-----------------------------------------------------------------------------#

setLogLevel(logging.DEBUG)


#===============================================================================
#
#    setup random ID function
#
# from 
#    https://www.geeksforgeeks.org/python-generate-random-string-of-given-length/
#

N = 6
def _newID():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k = N))

#===============================================================================
#
#    setup defaults
#

rcparams = {
    'Zbase'  : 50,       # Ohm
    'funit'  : 'Hz',     # 'Hz' | 'kHz' | 'MHz' | 'GHz' # keep to Hz for the moment
    'fs'     : (35e6, 60e6, 251),
    'interp' : 'MA',     # 'MA' | 'RI' | 'dB'
    'interpkws' : {'kind': 3, 'fill_value':'extrapolate'},
}

FUNITS = {'HZ':1., 'KHZ':1e3, 'MHZ':1e6, 'GHZ':1e9}
fscale = lambda frm, to='Hz': FUNITS[frm.upper()]/FUNITS[to.upper()]
