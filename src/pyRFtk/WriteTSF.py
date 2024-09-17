__updated__ = "2024-08-29 15:46:08"

import numpy as np
import re
import time

#===============================================================================
#
# W r i t e T S F 
#
#===============================================================================

def WriteTSF(object, path2tsf=None,
           fs=[], 
           tsf_format='GHZ S DB R 50', 
           num_format=None,
           comments = ''):
    
    """WriteTSF:
    
        object     : a pyFtk object (rfCircuit, fObject, rfTRL, ...)
        path2tsf   : path to file: if it is None or an empty string the TSF is
                     returned as a string
        tsf_format : the touchstone version 1.1 string
        num_format : tuple for resp. the freq,  reat and imag number format
        comments   : multi-line string
        
        
        note (I believe) the touchstone specification 1.1 transposes the 
             the coefficients for .s2p (2-ports): this is not implemented here
             as it is (apparently) not done in ReadTSF. 
        
    """

    s  = '! TouchStone file (pyRFtk.WriteTSF version %s)\n' % __updated__
    s += '! Date : %s \n\n' % time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    
    scomments = comments.split('\n')
    if len(scomments) > 0 :
        for line in scomments :
            s += '! %s\n' % line
        
    s += '\n# %s \n\n' % tsf_format
    
    fscale = {'HZ': 1e0, 'KHZ':1e3,'MHZ':1e6, 'GHZ':1e9, 'THZ':1e12}[
        re.findall('^.*?([KMG]?HZ).*$',tsf_format.upper())[0]]

    try:
        zref = float(
                re.findall(
                    '^.*?R\s*([-+]?(?:\d*[\.])?\d+(?:[eE][-+]?\d+)?).*?$',
                    tsf_format
                )[0]
               )
    except:
        zref = 50.0
        
    fmt = re.findall('(RI|MA|DB)',tsf_format.upper())
    if len(fmt) == 0:
        fmt = 'DB'  # default
    elif len(fmt) == 1:
        fmt = fmt[0]
    else:
        raise ValueError(
            'WriteTSF: one one of DB, RI or MA allowed in format string'
            )
        
    try:
        if len(object.ports) > 0 :
            s += '!PORTS ' + ', '.join(object.ports) + '\n\n'
    except AttributeError:
        pass
    
    if len(fs) == 0:
        
        # try an find a way to a list of meaning full frequencies
        try:
            if len(object.fs) > 0:
                fs = object.fs
        except AttributeError:
            pass
    
    if len(fs) == 0:
        raise ValueError(
            'could not find list of frequencies in object and no list was supplied'
            )
    nports = len(object)
    
    if num_format is None:
        if fmt in ['DB','MA']:
            num_format = ('8.4f', '10.7f', '+6.1f')
        else:
            num_format = ('8.4f', '10.7f', '10.7f')
    
    tnumfmt = f'%{num_format[1]} %{num_format[2]} '
    
    for k,fk in enumerate(fs):
        s += (f'%{num_format[0]}' % (fk / fscale)) + ' '
        d = object.getS(fk, Zbase=zref)
        for kr in range(nports):
            for kc in range(nports):
                dij = d[kr,kc]
                if fmt == 'RI':
                    s += tnumfmt % (np.real(dij),np.imag(dij))
                    
                elif fmt == 'DB':
                    s += tnumfmt% (20.*np.log10(np.abs(dij)),
                                               np.angle(dij,1))
                    
                else: # can only be MA
                    s += tnumfmt % (np.abs(dij),np.angle(dij,1))
                    
                if (kc % 4 == 3) and (kc is not nports-1):
                    if kc == 3:
                        s += ' !    row %d' % (kr+1)
                    s += '\n'+' '*10
                    
            s += '\n'
            if kr != nports-1:
                s += ' '*10
                
        
    if path2tsf:
        with open(path2tsf,'w') as f:
            f.write(s)
                     
    else:
        return s

     
#===============================================================================
#
# _ _ m a i n _ _
#
#===============================================================================
if __name__ == '__main__':
    from scipy.io import loadmat
    
    R = loadmat('/home/frederic/workspace/TWA/S22_TWA.npy')
    print(R)
    
    