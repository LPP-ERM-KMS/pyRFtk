__updated__ = "2023-09-12 10:02:08"

import numpy as np
import re
from .getlines import getlines 
from .config import tLogger, logit, ident
from .config import fscale

from .ConvertGeneral import ConvertGeneral
from .S_from_Z import S_from_Z
from .S_from_Y import S_from_Y

#===============================================================================
#
#  r e a d _ t s f 
#
def ReadTSF(src, **kwargs):
    """read_tsf
    
    read a touchstone file version 1 
    
    special comments:
        !PORTS      str, str, ... , str
        !PART       (TOP|BOTTOM)?(LEFT|RIGHT)?
        !SHAPE      tuple of int
        !NUMBERING  (UP|DOWN)?
        !REM
        !MARKERS    list of frequencies
    """
    debug = logit['DEBUG']
    debug and tLogger.debug(ident(
        f'> [CommonLib.ReadTSF] src= {src}, kwargs= {kwargs}',
        1
    ))
    
    #TODO: implement ports, funit
    
    TZbase = kwargs.pop('Zbase', 50.)
    Tfunit = kwargs.get('funit', 'GHz')
    Tports = kwargs.get('ports', 'port-%03d')
    Tcomments = kwargs.get('comments',[])
    
    if debug:
        tLogger.debug(ident(f'TZbase = {TZbase}'))
        tLogger.debug(ident(f'funit  = {Tfunit}'))
        tLogger.debug(ident(f'Tports = {Tports}'))
        
        for k, cmt in Tcomments:
            tLogger.debug(ident(f'Tcomments[{k:4d}] = "{cmt}"'))
    
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
    
    # analyse the format string: 
    #  for unnormalized data (R missing) Zbase is set to None
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
            if len(shape) == 1:
                shape.append(0)
            
        tlist = re.findall('!PART (.+)',aline)
        if len(tlist) > 0:
            part = tuple([x.lower() for x in tlist[0].split()[:2]])
            if len(part) == 1:
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
    debug and tLogger.debug(ident(f'source = {source}'))
    
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
                
                debug and tLogger.debug(ident(f'GENERAL_MIXED = {GENERAL_MIXED}'))
                
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
        reGline = re.compile('(?i)\s*(?:Gamma)\s+!?(.+)')
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
            Gline = re.findall('(?i)\s*(Gamma)\s+!?(.+)',rest)
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
            (Gms.shape[1] != N and                    # ports mismatch
             Gms.shape[1] != N**2)):                  # ports mismatch (new HFSS fmt ?
            print('coefss:', Ss.shape)
            print('Gms:', Gms.shape)
            print('Zcs:', Zcs.shape)
            raise IOError('Complex Port Impedance and Gamma data mismatch')
    
        # print(f'Gms.shape : {Gms.shape}')
    
        if Gms.shape[1] == N**2:
            # this appear to be the new format of the Gamma Z0 data
            # not clear what the reason for the change is ...
            # we assume that the data is on the diagonal of the square matrix
            # and we check that all other elements are zero ...
            Gms = Gms.reshape((Ss.shape[0],N,N))
            Zcs = Zcs.reshape((Ss.shape[0],N,N))
            tlist = []
            for kf, (fGms, fZcs) in enumerate(zip(Gms, Zcs)):
                for kr, (rGm, rZc) in enumerate(zip(fGms, fZcs)):
                    for kc, (Gm, Zc) in enumerate(zip(rGm, rZc)):
                        if kr != kc and (np.abs(Gm) > 0 or np.abs(Zc) > 0):
                            tlist.append((kf, kr, kc))
            if tlist:
                logit['WARNING'] and tLogger.warning(
                    f'{len(tlist)} non-zero off-diagnal elements for Gammas and'
                     ' Complex Port Impedances'
                )
                
            Gms = [np.diag(Gmk) for Gmk in Gms]
            Zcs = [np.diag(Zck) for Zck in Zcs]
        
    # get the frequencies and scale for the units
    fs = np.array(fs, dtype=float) * fscale(funit, Tfunit)
    
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
    
    else: # datatype == 'S':
    
        if Zbase is not None:
            if Zbase != TZbase:
                Ss = ConvertGeneral(TZbase, Ss, Zbase, 'V', 'V')
    
        else: # Zbase is None:
            if len(Zcs[0]) == N:
                Ss = ConvertGeneral(TZbase, Ss, Zcs, 'P', 'V')
            else:
                logit['ERROR'] and tLogger.error(f'len(Zcs)[{len(Zcs)}] != N[{N}]')
                logit['ERROR'] and tLogger.error(f'Zcs={Zcs}')
            
    # set the portnames
    
    if 'ports' in kwargs:
        # caller supplied portnames (or equivalent format string)
        if isinstance(Tports,str):
            ports = [Tports % k for k in range(1,N+1)]
            debug and tLogger.debug(ident(
                f'ports were supplied as a format string "{Tports}" -> {ports}'))
            
        elif isinstance(Tports, list) and len(Tports) == N:
            ports = Tports
            debug and tLogger.debug(ident(
                f'ports were supplied as a list -> {ports}'))
        
        else:
            logit['ERROR'] and tLogger.error(
                f'ports were supplied but could not be used {Tports}')
    
    elif ports and len(ports) == N: 
        # these are the ports that were found in the tsf
        debug and tLogger.debug(ident(f'found ports in tsf {ports}'))
        
    else:
        # supplied or found ports are not suitable
        debug and tLogger.debug(ident(f'  port in kwargs: {"ports" in kwargs}'))
        debug and tLogger.debug(ident(f'  ports         : {ports}'))
        if hasattr(ports, '__iter__'):
            debug and tLogger.debug(ident(
                f'  len(ports) == {N:4} : {len(ports) == N}'))
        
        ports = [Tports % k for k in range(1,N+1)]
        debug and tLogger.debug(ident(f'set ports to {ports}'))
    
    debug and tLogger.debug(ident(f'< [CommonLib.ReadTSF]',-1))
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
        "Zc"        : Zcs,
        "Gm"        : Gms,
        "variables" : None,
        "datafmt"   : datafmt,
    }

#===============================================================================
#
#  _ _ m a i n _ _ 
#
if __name__ == '__main__':
    # ReadTSF(src)
    pass
    
