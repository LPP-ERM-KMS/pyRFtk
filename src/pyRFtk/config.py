__updated__ = "2023-04-20 08:40:02"

import time
import string
import random
import logging
import inspect

import os
import atexit

LOGFILE = 'rfCommon.log'
#===============================================================================
#
#    setup logging
#
logit = dict([(lvl, True) for lvl in logging._nameToLevel])

tLogger = logging.getLogger('pyRFtk2')

logging.basicConfig(
    level = logging.CRITICAL, 
    filename = LOGFILE,
    filemode = 'w',
    format = '%(levelname)-8s - %(filename)-20s %(funcName)-15s'
             ' [%(lineno)5d]: %(message)s'
)

# tLogger = logging.getLogger('pyRFtk2')
tLogger.info(f'{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())} -- '
             f'logging level {logging.getLevelName(logging.getLogger().level)}')

#=-----------------------------------------------------------------------------#

def setLogLevel(level):
    
    newLvl = level if isinstance(level, int) else logging._nameToLevel[level]
    
    (logit['DEBUG'] or newLvl == logging.DEBUG) and tLogger.debug(
        f'Change loglevel from {logging.getLevelName(logging.getLogger().level)} '
        f'to {logging.getLevelName(newLvl)}')
    
    tLogger.setLevel(newLvl)
    
    for lvl, name in logging._levelToName.items():
        logit[name] = tLogger.isEnabledFor(lvl)
    
    # if logit['DEBUG']:
    #     tLogger.debug('SetLogLevel: DEBUG')
    #     for s in logit:
    #         tLogger.debug(f'   {s:10}  {logit[s]}   {logging._nameToLevel[s]}')

#=-----------------------------------------------------------------------------#

setLogLevel(logging.CRITICAL)

def CleanUpLogFile():
    try:
        s = os.path.getsize(LOGFILE)
        if s == 0:
            # print(f'pyRFtk.config.CleanUpLogFile: deleting {LOGFILE} [{s} bytes]')
            os.remove(LOGFILE)
        else:
            print(f'pyRFtk.config.CleanUpLogFile: keeping {LOGFILE} [{s} bytes]')
    except FileNotFoundError:
        pass

atexit.register(CleanUpLogFile)

#=-----------------------------------------------------------------------------#

def ident(message,d=0):
    try:
        ident.N
    except AttributeError:
        ident.N = 0
    
    ident.N += d if d < 0 else 0
    S = f'[{ident.N:3d}] ' + '| '*ident.N + message
    ident.N += d if d > 0 else 0
    
    return S

#=-----------------------------------------------------------------------------#

def get_class_that_defined_method(meth):
    # meth must be a bound method
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
            
def logident(message, printargs=False, stacklev=2):
    try:
        logident.N
    except AttributeError:
        logident.N = 0
        logident.stack = []
        
    def notset(*args, **kwargs):
        pass
    
    def strval(val):
        try:
            s = str(val)
        except:
            s = f'<{type(val).__name__}>'
        s = s.split('\n')
        return s[0] if len(s) < 2 else s[0]+' ...'
            
    flog = {
        'NOTSET'    : notset,
        'DEBUG'    : tLogger.debug,
        'INFO'     : tLogger.info,
        'WARNING'  : tLogger.warning,
        'ERROR'    : tLogger.error,
        'CRITICAL' : tLogger.critical
    }[logging._levelToName[tLogger.getEffectiveLevel()]]
        
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, stacklev)

    if message[0] == '>':
        msg = ident(
            f'> '+calframe[stacklev-1][3]+' '+(message[1:].lstrip().split() + [''])[0], +1)
        logident.stack.append(calframe[stacklev-1][3])
        flog(msg, stacklevel=stacklev)
        
        if printargs:
            args, varargs, kwargs, values = inspect.getargvalues(calframe[1][0])
            if args:
                for i in args:
                    s =f'    {i} = '
                    # print("#######\n#\n#",type(values[i]),'\n#',str(values[i]),"\n#\n#####")
                    if True or values[i] != '_debug_': # because of the matra: _debug_ and logident(...)
                        try:
                            lines = str(values[i]).split('\n')
                        except:
                            lines = [type(values[i]).__name__]
                        for _, aline in enumerate(lines):
                            if aline:
                                if _:
                                    flog(ident(f'{" ":{len(s)}}{aline}'), stacklevel=stacklev)
                                else:
                                    flog(ident(f'{s}{aline}'), stacklevel=stacklev)
            if varargs:
                for karg, targ in enumerate(values[varargs],1):
                    flog(ident(f'    #{karg} : {strval(targ)}'), stacklevel=stacklev)
                
            if kwargs:
                for kw in values[kwargs]:
                    flog(ident(f'    {kw} : {strval(values[kwargs][kw])}'), stacklevel=stacklev)
            
    elif message[0] == '<':
        clr = '< '+ (logident.stack.pop() if len(logident.stack) else '**stack err**')
        msg = ident(clr, -1)
        flog(msg, stacklevel=stacklev)
        
    else:
        msg = ident(message)
        flog(msg, stacklevel=stacklev)

#===============================================================================
#
#    setup random ID function
#
# from 
#    https://www.geeksforgeeks.org/python-generate-random-string-of-given-length/
#

N = 6
def _newID():
    try:
        _newID.N += 1
    except:
        _newID.N = 0
    return f'{_newID.N}'
    # return ''.join(random.choices(string.ascii_uppercase + string.digits, k = N))

#===============================================================================
#
# indentlevel
#
INDENTLEVEL = 0

#===============================================================================
#
#    setup defaults
#

#TODO #FIXME: leave funit to HZ for the moment

rcparams = {
    'Zbase'  : 50,       # Ohm
    'funit'  : 'Hz',     # 'Hz' | 'kHz' | 'MHz' | 'GHz' # keep to Hz for the moment
    'fs'     : (35e6, 60e6, 251),
    'interp' : 'MA',     # 'MA' | 'RI' | 'dB'
    'interpkws' : {'kind': 3, 
                   'fill_value':'extrapolate'
                  },
}

FUNITS = {'HZ':1., 'KHZ':1e3, 'MHZ':1e6, 'GHZ':1e9}
fscale = lambda frm, to='Hz': FUNITS[frm.upper()]/FUNITS[to.upper()]
