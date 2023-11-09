#+-------------+----------------------------------------------------------------+
#| Date        | Comment                                                        |
#+=============+================================================================+
#| 2015-Aug-27 | made userdirs the second argument and set its default to ['.'] |
#|             | this allows to partially keep the former call signature        |
#|             | e.g. findpath(fpattern,'../../../')                            |
#+-------------+----------------------------------------------------------------+
#| 2015-Aug-26 | modified from the original findpath                            |
#+-------------+----------------------------------------------------------------+

from glob import glob
import os

#===============================================================================
#
# f i n d p a t h
#
def findpath(fpattern,
             userdirs=['.'],
             envvar='',
             appdirs=[],
             missingOK=True,
             verbose=False):
    
    """findpath(fpattern, userdirs, envar, appdirs, missingOK, verbose)
    
       fpattern : the file's basename to search for
       
       envvar   : if an environment variable is set with a : separated set of 
                  directories then these directories will be searched first
                  in the order of appearance in the environment variable
                  (default : '')
                  
       usersdirs : list of directories searched next in order of appearance
                   (default ['.'])
        
       appdirs : list of directories searched finally in order of appearance
                 (default [])
       
       missingOK : do not raise IOError when a directory does not exist
       
       verbose : print debugging messages
       
    """
    
    def listdirs(name, dirlist):
        s = '    %s :\n' % name
        if dirlist:
            for tdir in dirlist:
                sdir = os.path.expanduser(tdir)
                stat = 'missing' if not os.path.exists(sdir) else 'exists'
                s += '        %-8s %r\n' % (stat, tdir)
        else:
            s += '        (empty)\n'
        return s
    
    makelist = lambda x : x if isinstance(x, list) else [x]\

    #- ----------------------------------------------------------------------- -
    
    SEARCH_DIRS = []
    if envvar:
        envvardirs = os.getenv(envvar, '')
        if verbose:
            print('\nknown environment variables\n')
            keys = sorted([s for s in os.environ.keys()])
            for k in keys:
                print('%-30s : %r' % (k, os.environ[k]))
            print('\n')
        if envvardirs:
            SEARCH_DIRS = envvardirs.split(':')
            if verbose:
                print('findpath: $%s :' % envvar, SEARCH_DIRS)
                
    userdirs, appdirs = makelist(userdirs), makelist(appdirs)
    
    SEARCH_DIRS += userdirs + appdirs
    
    fname = None
    for tdir in SEARCH_DIRS:
        sdir = os.path.expanduser(tdir)
        if verbose:
            print('findpath : looking in %r' % sdir)
        if not os.path.exists(sdir):
            if verbose:
                print('  (missing)')
            if not missingOK:
                raise IOError('findpath : non-existing search directory %r' % sdir)
        else:
            for root, dirs, files in os.walk(sdir):  # @UnusedVariable
                if verbose:
                    print('  %r' % ('.' + root[len(sdir):]))
                    for f in files:
                        print('   %r' % f)
                fnames = glob(os.path.join(root, fpattern).replace('[', '[[]'))
                if fnames:
                    fname = sorted(fnames)[-1]  # take the first / last
                    break
            if fname:
                break
            
    if not fname:
        s = ('findpath: could not find single matching candidate for '
               '%r in : \n' % fpattern)
        if envvar in os.environ:
            s += listdirs('$' + envvar, envvardirs.split(':'))
        else:
            s += '    $%s : (not set)\n' % envvar
        s += listdirs('userdirs', userdirs)
        s += listdirs('appdirs', appdirs)
         
        raise IOError(s)
    
    if verbose:
        print('findpath ==> %r' % fname)
        
    return fname

#===============================================================================
#
# e x i s t s p a t h
#
def existspath(fpattern,
               userdirs=['.'],
               envvar='',
               appdirs=[],
               missingOK=True,
               verbose=False):
    
    """same as findpath but does not raise an exception : just return an empty
        string
    """
    try:
        fpath = findpath(fpattern, userdirs, envvar, appdirs, missingOK, verbose)
    except IOError:
        fpath = ''
    
    return fpath
    
#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == "__main__":
    print(findpath('* -- Level1.res',
                   '../CYCLE_2018'))
