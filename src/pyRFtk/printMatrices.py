import numpy as np

#===============================================================================
#
# s t r M
#

def strM(M,
         pfmt='%7.3f%+7.3fj, ',
         pfun=lambda x : (np.real(x), np.imag(x)),
         printzeros='0'):
    """
    pretty 'str' a complex matrix
    """
    
    if not isinstance(M, np.ndarray):
        M = np.array(M)
        
    s = '[%d,%d]\n\n' % M.shape
    for k1 in range(M.shape[0]):
        for k2 in range(M.shape[1]):
            s1 = pfmt % pfun(M[k1, k2])
            if  (printzeros) and np.abs(M[k1, k2]) < 1e-7 :
                s += printzeros.center(len(s1))
            else :
                s += s1
        s += '\n'
    s += '\n'
    
    return s

#===============================================================================
#
# p r i n t M
#

def printM(M,
           pfmt='%7.3f%+7.3fj, ',
           pfun=lambda x : (np.real(x), np.imag(x)),
           printzeros='0'):
    """
    pretty str a complex matrix to print or write to a file
    """
    print(strM(M, pfmt=pfmt, pfun=pfun, printzeros=printzeros))
    
#===============================================================================
#
# p r i n t R I
#

def printRI(C, pfmt='%7.3f%+7.3fj, ', printzeros='0') :
    """
    pretty print a complex matrix
    """
    print(strM(C,
               pfmt,
               pfun=lambda x : (np.real(x), np.imag(x)),
               printzeros=printzeros))

#===============================================================================
#
# p r i n t M A
#

def printMA(C, pfmt='%8.5f %+7.2f'+u'\N{DEGREE SIGN}'+', ', printzeros='0') :
    """
    pretty print a complex matrix
    """
    print(strM(C,
               pfmt,
               pfun=lambda x : (np.abs(x), np.angle(x, deg=1)),
               printzeros=printzeros))

#===============================================================================
#
# p r i n t D B
#

def printDB(C, pfmt='%7.3f%+7.1f'+u'\N{DEGREE SIGN}'+', ', printzeros='0') :
    """
    pretty print a complex matrix
    """
    print(strM(C,
               pfmt,
               pfun=lambda x : (20 * np.log10(np.abs(x)), np.angle(x, deg=1)),
               printzeros=printzeros))

#===============================================================================
#
# p r i n t R 
#

def printR(R, pfmt='%7.3f, ', printzeros='0'):
    """
    pretty print (the real part) of a (possibly complex) matrix
    """
    print(strM(R, pfmt, pfun=lambda x : x.real, printzeros=printzeros))
 
