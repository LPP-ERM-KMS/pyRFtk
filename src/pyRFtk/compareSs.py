__updated__ = '2021-10-14 10:02:38'

import numpy as np
import matplotlib.pyplot as pl

def compareSs(fs, Ss, cs=[], lbls=[], **figkwargs):
    
    if not isinstance(Ss, list):
        Ss = [Ss]
    
    if len(Ss) > 1:
#         for k, Sk in enumerate(Ss):
#             print(f'Ss[{k}].shape {Sk.shape}')
            
        if any([len(Sk.shape) != 3 or 
                Sk.shape[1] != Sk.shape[2] or
                Sk.shape[1:]!= Ss[0].shape[1:] for Sk in Ss]):
            raise ValueError('incompatible list of S\'')
    
    if not isinstance(fs, list):
        fs = [fs for k in range(len(Ss))]
    elif len(fs) != len(Ss):
        raise ValueError('incompatible number of fs and Ss')
    
    if any([len(fk) != len(Sk) for fk, Sk in zip(fs, Ss)]):
        raise ValueError('incompatible fs and Ss')
    
    if not len(cs):
        cs = ['r','b--','g--','m','c','k']
    
    if not len(lbls):
        lbls = [f'{k+1}' for k in range(len(Ss))]
    
    N = Ss[0].shape[1]
    
    fig, axs = pl.subplots(N, N+1, sharex=True, **figkwargs)
    if N == 1:
        axs = [axs]
    
    for kr in range(N):
        for kc in range(kr+1):
            pl.sca(axs[kr][kc])
            for fk, Sk, ck, lblk in zip(fs, Ss, cs, lbls):
                pl.plot(fk, np.abs(Sk[:,kr,kc]), ck, label=lblk)
            pl.grid()
            if kr == N-1:
                pl.xlabel('frequency [MHz]')
            pl.title(f'| S$_{{{kr+1}{kc+1}}}$ |')
            pl.legend(loc='best')
            pl.ylim(-0.05,1.05)
    
            pl.sca(axs[kc][kr+1])
            for fk, Sk, ck, lblk in zip(fs, Ss, cs, lbls):
                ph = np.angle(Sk[:,kr,kc])
                # ph += (2*np.pi) if ph[0] < 0 else 0                    
                pl.plot(fk, ph, ck, label=lblk)
            pl.grid()
            if kc == N-1:
                pl.xlabel('frequency [MHz]')
            pl.title(f'phase( S$_{{{kr+1}{kc+1}}}$ ) [rad]')
            pl.legend(loc='best')
            pl.ylim(-np.pi,np.pi)
            
    pl.tight_layout()
    
#===============================================================================
#
# _ _ m a i n _ _
#
if __name__ == '__main__':
    
    import os
    
    from Utilities import ReadDictData
    from pyRFtk2 import rfGTL, rfObject
    from pyRFtk import circuit_Class3a as _ct, TouchStoneClass3a as _ts
    
    from pprint import pprint
    
    fMHzs = np.linspace(35, 65, num=51)
    Zbase = 20

    if False:
        path2model = '/home/frederic/git/iter_d_wws896'                \
                     '/ITER_D_WWS896/src/CYCLE 2018/SS/WHu 20201021'   \
                     '/S_STSR_XN7VDD_v1p0.modss'
        
        SS = ReadDictData(path2model)
        pprint(SS['SS']['variables'])
        SSTS = rfObject(touchstone = path2model.replace('.modss','.s2p'))
        
        gtlCT = rfGTL(SS['SS'],  SS['variables'])
        
        SS2 = ReadDictData(path2model)
        SS2['SS']['variables']['LS'] = 0.6565
        pprint(SS2['SS']['variables'])
        gtlCT2 = rfGTL(SS2['SS'],  SS2['variables'])
                
        gtlSs = gtlCT.getS(fMHzs*1E6, Zbase=Zbase)
        gtlSs2 = gtlCT2.getS(fMHzs*1E6, Zbase=Zbase)
        tsfSs = SSTS.getS(fMHzs*1E6, Zbase=Zbase)
        
        compareSs(fMHzs, 
                  [gtlSs, gtlSs2, tsfSs], 
                  lbls=['LS=0.6265','LS=0.6565','tsf'])
    
    if False:
        path2model = '/home/frederic/git/iter_d_wws896'                \
                     '/ITER_D_WWS896/src/CYCLE 2018/FPJ/WHu20201021'   \
                     '/S_FPJR_XN7VDD_v1p0.mod4pj'
        
        FPJ = ReadDictData(path2model)
        FPJTS = rfObject(touchstone = path2model.replace('.mod4pj','.s4p'))
        pprint(FPJ['4PJ'])
        
        gtlCT = rfGTL(FPJ['4PJ'],  FPJ['variables'], Zbase=Zbase)
        
        ogtlSs = []
        FPJ = ReadDictData(path2model)
        for fMHz in fMHzs:
            CT, SZ = _ct.processGTL(fMHz, Zbase, FPJ['4PJ'], 
                                    variables = FPJ['variables'])
            ogtlSs.append(SZ.S)
        ogtlSs = np.array(ogtlSs)
        
        oFPJTS = _ts.TouchStone(filepath=path2model.replace('.mod4pj','.s4p'))
        
        gtlSs = gtlCT.getS(fMHzs*1E6, Zbase=Zbase)
        tsfSs = FPJTS.getS(fMHzs*1E6, Zbase=Zbase)
        otsfSs = oFPJTS.Datas(mtype='S',zref=Zbase)
        print(np.array(otsfSs).shape)
        compareSs(fMHzs, 
                  [gtlSs, tsfSs, ogtlSs, otsfSs], 
                  lbls=['gtl','tsf','pyRFtk','touch'])
        
        print(gtlCT)
        print(CT)
        print(SZ)
        
        
    if True:
        dirpath = '/home/frederic/git/pyrftk/pyRFtk/test pyRFtk_new/' 
        
        TWAnew = rfObject(Zbase= 8.59, touchstone = os.path.join(
            dirpath,'n','WEST_TWA_7s_4_No_FS_protruding_sweep.s7p'))
        Snew = TWAnew.getS(TWAnew.fs)
        
        TWAorig = rfObject(Zbase= 8.59, touchstone = os.path.join(
            dirpath,'WEST_TWA_7s_4_No_FS_protruding.s7p'))
        Sorig = TWAorig.getS(TWAorig.fs)
    
#         compareSs([TWAnew.fs,TWAorig.fs],
#                   [Snew, Sorig], 
#                   lbls=['new','orig'])
        
        fig, axs = pl.subplots(2,4,figsize=(16,8))
        for lbl, TWA in zip(['new','orig'],[TWAnew, TWAorig]):
            S = TWA.getS(TWA.fs)
            for k in range(len(TWA)): # shoud be 7
                pl.sca(axs[k // 4][k % 4])
                pl.plot(TWA.fs/1E6,np.abs(S[:,k,k]),label=lbl)
                pl.grid(True)
                pl.title(f'| S$_{{{k+1}{k+1}}}$ |')
                pl.legend(loc='best')
                if k // 4 == 1:
                    pl.xlabel('frequency [MHz]')
        pl.tight_layout()
        
    pl.show()

