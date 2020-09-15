################################################################################
#                                                                              #
# Copyright 2018-2020                                                          #
#                                                                              #
#                   Laboratory for Plasma Physics                              #
#                   Royal Military Academy                                     #
#                   Brussels, Belgium                                          #
#                                                                              #
#                   ITER Organisation                                          #
#                                                                              #
# Author : frederic.durodie@rma.ac.be                                          #
#                          @gmail.com                                          #
#                          @ccfe.ac.uk                                         #
#                          @telenet.be                                         #
#                          .lpprma@tlenet.be                                   #
#                                                                              #
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by   #
# the European Commission - subsequent versions of the EUPL (the "Licence");   #
#                                                                              #
# You may not use this work except in compliance with the Licence.             #
# You may obtain a copy of the Licence at:                                     #
#                                                                              #
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-11-12                  #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the Licence is distributed on an "AS IS" basis,            #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the Licence for the specific language governing permissions and          #
# limitations under the Licence.                                               #
#                                                                              #
################################################################################

__updated__ = '2020-04-30 00:15:59'

"""
Created on 29 Apr 2020

@author: frederic
"""

import numpy as np
import re

from Utilities.getlines import getlines

#===============================================================================
#
#  r e a d _ t s f 
#
def read_tsf(src, **kwargs):
    """read_tsf
    
    read a touchstone file version 1 
    
    special comments:
        !PART       (TOP|BOTTOM)?(LEFT|RIGHT)?
        !SHAPE      tuple of int
        !NUMBERING  (UP|DOWN)?
        !REM
        !MARKERS    list of frequencies
    """
    
    #TODO: implement ports, funit
    
    TZbase = kwargs.pop('Zbase', 50.)
    Tfunit = kwargs.get('funit', 'GHz')
    Tports = kwargs.get('ports', '%2d')
    Tcomments = kwargs.get('comments',[])
            
    # process the source input 
        
    comments, fs = [], []
    coefs, Ss = np.array([]).reshape((0,)), []
    lineno = 0
    got_marker = 0
    tformat = None
    
    for aline in getlines(src)():
        lineno += 1
        aline = aline.strip()
        if aline:
            if aline[0] == '!':
                comments.append(aline)
            elif aline[0] == '#':
                if tformat is not None:
                    raise ValueError('multiple format lines detected')
                tformat = aline
            else:
                # floatRE = '([+-]?\d+(?:\.\d*)?(?:[eEdD][+-]\d+)?)'
                try:
                    items = [float(s) for s in aline.split('!',1)[0].split()]
                except:
                    # maybe we want to catch something, but what ?
                    raise
                odd = len(items) % 2
                if odd:
                    # odd number of floats this is a frequency line
                    # closeout the running coefs and append it to the
                    # coefss list. save the new frequency
                    if len(coefs):
                        Ss.append(coefs)
                        coefs = np.array([],dtype=float).reshape((0,))
                    fs.append(items[0])
                    
                coefs = np.append(
                           coefs,
                           np.array(items[odd::2], dtype=float) 
                           + 1j * np.array(items[odd+1::2], dtype=float)
                        )
                        
                if not got_marker:
                    comments.append('*** DATA MARKER ***')
                    got_marker = len(comments)

    
    # collect the last set of coefs                
    Ss.append(coefs)
    
    # reshape the collected coeffcients
    N = int(round(np.sqrt( len(Ss[0]) )))
    Ss = np.array(Ss).reshape((len(fs), N, N))
    
    # analyse the format string
    items = tformat[1:].upper().split()[::-1]
    datatype, datafmt, funit, Zbase = 'S', 'MA', 'GHZ', None 
    while len(items):
        kw = items.pop()
        if kw in ['S', 'Y', 'Z']:
            datatype = kw
        elif kw in ['MA', 'RI', 'DB']:
            datafmt = kw
        elif kw in ['HZ','KHZ','MHZ','GHZ']:
            funit = kw
        elif kw == 'R':
            Zbase = float(items.pop())
        else:
            raise ValueError('Unrecognized format specification: %r' % kw)
        
    # detect special comments
    ports = None
    part = None
    numbering = None
    Tcomments = []
    markers = None
    shape = None
    
    for aline in comments[:got_marker]:
            
        tlist = re.findall('!REM (.+)',aline)
        if len(tlist) > 0:
            Tcomments.append(tlist[0])
            
        tlist = re.findall('!PORTS (.+)',aline)
        if len(tlist) > 0:
            ports = [name.strip() for name in tlist[0].split(',')]
            
        tlist = re.findall('!MARKERS (.+)',aline)
        if len(tlist) > 0:
            markers = np.array([float(x) for x in tlist[0].split()])
            
        tlist = re.findall('!SHAPE (.+)',aline.upper())
        if len(tlist) > 0:
            shape = tuple([int(x) for x in tlist[0].split()[:2]])
            if len(shape) is 1:
                shape.append(0)
            
        tlist = re.findall('!PART (.+)',aline)
        if len(tlist) > 0:
            part = tuple([x.lower() for x in tlist[0].split()[:2]])
            if len(part) is 1:
                part = part[0]
        
        tlist = re.findall('!NUMBERING (.+)',aline)
        if len(tlist) > 0:
            numbering = tuple([x.lower() for x in tlist[0].split()[:2]])
            if len(shape) < 2:
                shape = shape[0] # so numbering becomes type str rather than tuple
            
        
    # detect the source of the touchstone

    pvars = {}
    GENERAL_MIXED = False
    source = re.findall(
        '(?i)! Touchstone file exported from HFSS (\d{4}\.\d+\.\d+)',
        comments[0])
    
    if source:
        # print('HFSS  %r' % source)
        lineno = 0
        while True:
            if lineno < got_marker:
                lineno += 1
            else:
                break
            try:
                aline = comments[lineno]
                if aline == '*** DATA MARKER ***':
                    break
                
                if re.findall('(?i)^!Data is not renormalized', aline):
                    GENERAL_MIXED = True
                    
                if re.findall('(?i)^! Variables:',aline):
                    lineno += 1
                    aline = comments[lineno]
                    while aline.strip() != '!':
                        varline = re.findall('!\s*([A-Za-z0-9_]+)\s*=\s*(.+)',
                                              aline)
                        if varline:
                            pvars[varline[0][0]] = varline[0][1]
                        else:
                            print('? %r %r' % (aline, varline))
                        
                        lineno += 1
                        aline = comments[lineno]

                    # print('got out variable loop grace fully')
                            
            except IndexError:
                break
        
        # pprint(pvars)
        # print('GENERAL_MIXED:', GENERAL_MIXED)
        
    # analyse the comments: detect if port impedances and gammas were present

    Gms, Gmsk = [], []
    Zcs, Zcsk = [], []
    lastline = ''
    
    re_compile = True # check if compiling regexes is faster ... not really
    if re_compile:
        reGline = re.compile('(?i)\s*(?:Gamma)\s+!(.+)')
        reZline = re.compile('(?i)\s*(?:Port Impedance)(.+)')
    
    for aline in comments[got_marker:]:
        # check for gamma or port impedance lines

        parts = aline.split('!', 1)
        if len(parts) > 1:
            aline, rest = tuple(parts)

        else:
            aline, rest = parts[0], ''
        
        if re_compile:
            Gline = reGline.search(rest)
            Zline = reZline.search(rest)
        else:
            Gline = re.findall('(?i)\s*(Gamma)\s+!(.+)',rest)
            Zline = re.findall('(?i)\s*(Port Impedance)(.+)',rest)
        
        
        if Gline or (lastline == 'gamma' and not Zline):
            
            if lastline != 'gamma':
                if len(Gmsk):
                    Gms.append(Gmsk)
                Gmsk = []
                if re_compile:
                    rest = Gline.group(1)
                else:
                    rest = Gline[0][1]
                    
            r =  rest.split()
            Gmsk += [float(r[i])+1j*float(r[i+1]) for i in range(0,len(r),2)]
            lastline='gamma'
        
        elif Zline or (lastline == 'impedance' and not aline):
            
            if lastline != 'impedance':
                if Zcsk:
                    Zcs.append(Zcsk)
                Zcsk = []
                if re_compile:
                    rest = Zline.group(1)
                else:
                    rest = Zline[0][1]
                    
            r =  rest.split()
            Zcsk += [float(r[i])+1j*float(r[i+1]) for i in range(0,len(r),2)]
            lastline = 'impedance'
                        
    # append the last Gms and Zcs
    if len(Gmsk):
        Gms.append(Gmsk)
    if len(Zcsk):
        Zcs.append(Zcsk)
        
    Gms = np.array(Gms)
    Zcs = np.array(Zcs)
    
    if len(Gms)>0 or len(Zcs)>0:
        if (Gms.shape != Zcs.shape or           # data mismatches
            Gms.shape[0] != Ss.shape[0] or      # frequencies mismatch
            Gms.shape[1] != N):                 # ports mismatch
            print('coefss:', Ss.shape)
            print('Gms:', Gms.shape)
            print('Zcs:', Zcs.shape)
            raise IOError('Complex Port Impedance and Gamma data mismatch')
    
    # get the frequencies and scale for the units
    fs = np.array(fs, dtype=float)
    
    # set the coefficient to RI from the format recieved
    if datafmt == 'MA':
        Ss = Ss.real * np.exp(1j * Ss.imag * np.pi / 180.)
    elif datafmt == 'DB':
        Ss = 10**(Ss.real/20.) * np.exp(1j * Ss.imag * np.pi / 180.)
    else: # datafmt == 'RI':
        pass
        
    # if the data type is Z or Y we need to convert to S
    if datatype == 'Z':
        Ss = S_from_Z(Ss, Zbase if Zbase else TZbase)
        
    elif datatype == 'Y':
        Ss = S_from_Y(Ss, Zbase if Zbase else TZbase)
    
    elif datatype == 'S' and Zbase is not None:
        if Zbase != TZbase:
            Ss = convert_general(TZbase, Ss, Zbase, 'V', 'V')
    
    else: # datatype == 'S' but the ports may not be normalized
        if Zbase is None and len(Zcs) == N:
            Ss = convert_general(TZbase, Ss, Zcs, 'P', 'V')
            
    # set the portnames
    
    if 'ports'in kwargs:
        # caller supplied portnames (or equivalent format string)
        if isinstance(Tports,str):
            ports = [Tports % k for k in range(1,N+1)]
            
        elif isinstance(Tports, list) and len(Tports) == N:
            ports = Tports
        
        else:
            print()
    
    elif ports and len(ports) == N: 
        # these are the ports that were found in the tsf
        ports = ports
        
    else:
        # supplied or found ports are not suitable
        ports = ['%02d' % k for k in range(1,N+1)]
            
    return {        
        "ports"     : ports,
        "fs"        : fs,
        "funit"     : funit,
        "Ss"        : Ss, 
        "Zbase"     : TZbase,
        "part"      : part,
        "shape"     : shape,
        "numbering" : numbering,
        "markers"   : markers,
        "variables" : None,
    }

#===============================================================================
#
# S _ f r o m _ Z 
#
def S_from_Z(Z, Zbase=50.):
    """
    returns the scattering matrix S defined on Zbase [Ohm] from an 
    impedance matrix Z
    
    always returns an numpy.ndarray unless the Z is a scalar
    """
    L, Z = _check_3D_shape_(Z)
    
    Z0 = Zbase*np.eye(Z.shape[1])
    S = [ (Zk - Z0) @ np.linalg.inv(Zk + Z0)  for Zk in Z]
    
    S = S[0,0,0] if L == 0 else S[:,0,0] if L == 1 else S[0,:,:] if L==2 else S 
    
    return S

#===============================================================================
#
# S _ f r o m _ Y 
#
def S_from_Y(Y, Zbase=50.):
    """
    returns the scattering matrix S defined on Zbase [Ohm] from an 
    admittance matrix Y
    
    always returns an numpy.ndarray unless the Z is a scalar
    """
    L, Y = _check_3D_shape_(Y)
    
    Y0 = np.eye(Y.shape[1])/Zbase
    S = [ (Y0 - Yk) @ np.linalg.inv(Y0 + Yk)  for Yk in Y]
    
    S = S[0,0,0] if L == 0 else S[:,0,0] if L == 1 else S[0,:,:] if L==2 else S 
    
    return S

#===============================================================================
#
#  _c h e c k _ 3 D _ s h a p e _
#
def _check_3D_shape_(M3):
    
    M3= np.array(M3)
    L = len(M3.shape)
    
    if L == 0 or L == 1 :
        M3 = M3.reshape(M3.shape[0] if L else 1, 1, 1)
        
    elif L == 2 and M3.shape[0] == M3.shape[1]:
        M3 = M3.reshape(1,M3.shape[0],M3.shape[1])
        
    elif L !=3 or M3.shape[1] != M3.shape[2]:
        #
        # expected:
        #    a scalar (float or complex) possibly as numpy.array(<scalar>)
        #    a 1D list or array which is reshaped to (len(list), 1, 1)
        #    a square 2D list or array which is reshaped to (1, N, N), N = len(list)
        #    a list or aray of square 2D lists / arrays of the shape (M, N, N)
        #
        raise ValueError('_check_3d_shape_: Unexpected shape: %r' % (M3.shape))

    return L, M3
    
#===============================================================================
#
#  c o n v e r t _ g e n e r a l
#
def convert_general(Z2, S1, Z1, type1='std', type2='std'):
    
    
    # check S1 ... a single scalar s         -> [ [[s]] ] 1 freq 1x1 S-matrix
    #              a list of 1 element [s]   -> [ [[s]] ] 1 freq 1x1 S-matrix
    #              a 2D list/array     [[S]] -> [ [[S]] ] 1 freq NxN S-matrix
    #              a 3D list/array    [ [[S_1]] ... [[S_n]] ] n freqs NxN S-matrices
    
    L, S1 = _check_3D_shape_(S1)
            
    nF, N = S1.shape[:2]
    
    O = np.ones((nF, N))                  # (nF,N)

    def coeffs(Z, typ):    
        
        # check Z ... a single scalar z   -> [ [z ... z] * nF ] nF freqs N ports
        #             a 1D list/array [Z] -> [ [z_1 ... z_N] * nF ] nF freqs N ports
        #             a 2D list/array [[Z]] -> nF frequencies N ports
        
        Zd = np.array(Z)
        
        if len(Zd.shape) == 0:
            Zd = np.array([[Zd] * N] * nF)
            
        elif len(Zd.shape) == 1 and Zd.shape[0] == N:
            Zd = np.array(Zd.tolist() * nF).reshape(nF,N)
            
        elif len(Zd.shape) != 2 or Zd.shape[0] != nF or Zd.shape[1] != N:
            raise ValueError(
                'convert_general: Z1 or Z2 shape not expected ... %r' % (
                    np.array(Z).shape))
                    
        iZd = 1/Zd                                           # (nF,N)
        iRd = 1/np.sqrt(np.real(Zd))                         # (nF,N)
        
        if typ == 'V':
            QaV, QaI = O/2,  Zd/2                            # (nF,N)
            QbV, QbI = O/2, -Zd/2                            # (nF,N)
            QVa, QVb = O, O                                  # (nF,N)
            QIa, QIb = iZd, -iZd                             # (nF,N)
            
        elif typ == 'G':
            QaV, QaI = iRd/2,  Zd * iRd / 2                  # (nF,N)
            QbV, QbI = iRd/2, -np.conj(Zd) * iRd / 2         # (nF,N)
            QVa, QVb = np.conj(Zd) * iRd, Zd * iRd           # (nF,N)
            QIa, QIb = iRd, -iRd                             # (nF,N)
        
        elif typ == 'P':
            iRd = 1/np.sqrt(Zd)                              # (nF,N)
            QaV, QaI = iRd/2,  Zd * iRd / 2                  # (nF,N)
            QbV, QbI = iRd/2, -Zd * iRd / 2                  # (nF,N) 
            QVa, QVb = Zd * iRd, Zd * iRd                    # (nF,N) 
            QIa, QIb = iRd, -iRd                             # (nF,N)
        
        else:
            raise ValueError('type must be "std" or "gsm" got %r' % typ)
                 
        return QaV, QaI, QbV, QbI, QVa, QVb, QIa, QIb
    
    *_, SVa, SVb, SIa, SIb = coeffs(Z1, type1)
    TaV, TaI, TbV, TbI, *_ = coeffs(Z2, type2)
    
    # there is no neat way to do this without looping over the frequencies ...
    S2 = []
    for S1k, SVak, SVbk, SIak, SIbk, TaVk, TbVk, TaIk, TbIk in zip(
        S1, SVa, SVb, SIa, SIb, TaV, TbV, TaI, TbI):
        
        V = np.diag(SVak) + np.diag(SVbk) @ S1k
        I = np.diag(SIak) + np.diag(SIbk) @ S1k
        A2 = np.diag(TaVk) @ V + np.diag(TaIk) @ I
        B2 = np.diag(TbVk) @ V + np.diag(TbIk) @ I
        S2.append(B2 @ np.linalg.inv(A2))
    
    S2 = np.array(S2)
    
    if L == 0:
        return S2[0,0,0]
    
    elif L == 1:
        return [S2[0,0,0]]
    
    elif L == 2:
        return S2[0,:,:]
             
    return S2

    
