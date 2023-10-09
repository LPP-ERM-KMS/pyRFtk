import ast
from .findpath import findpath

## -----------------------------------------------------------------------------
#  R e a d D i c t D a t a
#
def ReadDictData(fpath, userdirs=['.'], envvar='', appdirs=[],
                 missingOK=True, verbose=False):
    """
    read a text file representing a model (.mod4PJ, .model, .ciamod, ...)
    and return the dict object
    """
    
    try:
        with open(findpath(fpath, envvar=envvar, userdirs=userdirs, appdirs=appdirs,
                       missingOK=missingOK, verbose=verbose), 'r') as f :

            return ast.literal_eval(f.read())
    except :
        raise
    
## -----------------------------------------------------------------------------
#  R e a d D i c t S t r i n g
#
def ReadDictString(s):
    try :
        return ast.literal_eval(s)
    except :
        raise
   
## -----------------------------------------------------------------------------
    
if __name__ == '__main__' :
    print(ReadDictData('ILA_Class3.rc', userdirs=['.']))
