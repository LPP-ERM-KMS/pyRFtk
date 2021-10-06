"""
Arnold's Laws of Documentation:
    (1) If it should exist, it doesn't.
    (2) If it does exist, it's out of date.
    (3) Only documentation for useless programs transcends the first two laws.

Created on 10 Feb 2021

@author: frederic

this implements a circuit class for manipulating RFbase objects

RFobject must implement following methods/attributes
    (attribute) Zbase     float
    (attribute) ports     list of strings
    (method)    __len__   integer = number of ports (#p)
    (method)    getS      input:
                            fs float or list/array of floats, frequencies in Hz 
                            Zbase float, 
                            params dict
                          output:
                          - list/array of f's -> array of Smatrices [#f,#p,#p]
                             single f -> single Smatrix [#p,#p]
                          - the resulting Smatrices are converted to Zbase if 
                            given else the Smatrix is given in the circuit's
                            Zbase (default 50 Ohm)
    (method)    set       *args, **kwargs
                     
    
circuit building methods
    addblock(name, RFobj, ports, params)
    connect(*ports)
    terminate(port, Z=complex | Y=complex | RC=complex)


unused ports automatically become external ports

TODO: rename and order external ports as requested
TODO: rethink on how to set parameters
TODO: lumped circuit elements
TODO: maxV 
TODO: check logic for the status of solved or not
TODO: use external sNp if available

"""

__updated__ = "2021-10-01 13:42:45"

if __name__ == '__main__':
    import sys
    sys.path.append('../pyRFtk2 test')
    print('importing test_circuit ...')
    import test_circuit                                      # @UnresolvedImport

import numpy as np
import matplotlib.pyplot as pl

from .CommonLib import ConvertGeneral
from .Utilities import strM

from pyRFtk2 import rfObject # the name may change ... e.g. rfTSF
 
#===============================================================================
#
# c i r c u i t
#
class circuit():
    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, **kwargs):
        
        self.Zbase = kwargs.pop('Zbase', 50.)
        sNp = kwargs.pop('sNp', None)
        if kwargs:
            print(f'unprocessed kwargs: {".".join([kw for kw in kwargs])}')
        self.M = np.array([],dtype=np.complex).reshape((0,0))
        self.ports = []
        self.waves = []
        self.nodes = {}
        self.blocks = {}
        self.eqns = []
        self.f = np.nan
        self.C = {}
        self.T = {}
        self.E = {}                         # E[port] -> eqn number
        self.idxEs = []
        self.invM = None
        self.S = None
        
        if sNp:
            self.sNp = rfObject(touchtone=sNp)
        
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, full=False):
        s = f'circuit at {id(self)} (pyRFtk2.circuit version {__updated__}) \n'
        s += f'| Zbase: {self.Zbase} Ohm\n'
        s += f'| ports: {self.ports} \n'
        s += '|\n'
        s += f'+ {self.M.shape[0]} equations, {self.M.shape[1]} unknowns \n'
        s += '| \n'
        l1 = max([len(eqn) for eqn in self.eqns])
        l2 = max(len(wave) for wave in self.waves)
        for k, (Mr, eqn) in enumerate(zip(self.M, self.eqns)):
            s += f'| {self.waves[k]:<{l2}s} [{k:3d}] {eqn:<{l1}s} '
            typ, port = tuple(eqn.split())
            name = port.split('.')[0]
            # print(typ,port,name)
            if typ[0] == 'S':
                blk = self.blocks[name]
                N = len(blk['ports'])
                i1, i2 = blk['loc']
                s += f'[{i2:3d}..{i2+N-1:3d}] '
                for sij in Mr[i2:i2+N]:
                    s += f'{sij.real:+7.4f}{sij.imag:+7.4f}j '
                sij = Mr[i2+N+k-i1]
                s += f'[{i2+N+k-i1:3d}] {sij.real:+7.4f}{sij.imag:+7.4f}j '
            elif typ[0] == 'C':
                idxA, idxBs = self.C[port]
                idxs = sorted([idxA] + idxBs)
                for idx in idxs:
                    sij = Mr[idx]
                    s += f'[{idx:3d}] {sij.real:+7.4f}{sij.imag:+7.4f}j '
            elif typ[0] == 'T':
                idxs = sorted(self.T[port])
                for idx in idxs:
                    sij = Mr[idx]
                    s += f'[{idx:3d}] {sij.real:+7.4f}{sij.imag:+7.4f}j '
            elif typ[0] == 'E':
                idx = self.waves.index('->'+port)
                sij = Mr[idx]
                s += f'[{idx:3d}] {sij.real:+7.4f}{sij.imag:+7.4f}j '
            s += '\n'
        
        for k, p in zip(range(self.M.shape[0], len(self.waves)), self.ports):
            s += f'| {self.waves[k]:<{l2}s} [{k:3d}] E? {p:<{l1}s}\n'
        
        alreadylisted = {}
        if self.blocks:
            s += f'|\n+ {len(self.blocks)} Blocks:\n|\n'
            
            for blkID, blk in self.blocks.items():
                thisid = id(blk['object'])
                if thisid in alreadylisted:
                    alreadylisted[thisid].append(blkID)
                else:
                    alreadylisted[thisid] = [blkID]
                    
                    if isinstance(blk['object'], (list,np.ndarray)):
                        #fixme: embellish this type of block
                        s += f'|   {blkID}: fixed numpy array like at {thisid}\n'
                        for k, sk in enumerate(str(blk['object']).split('\n')):
                            s += '|   | ' +  sk + '\n'
                        s += '|   ^\n'
                    else:
                        for k, sk in enumerate(blk["object"].__str__(full).split('\n')):
                            s += ('|   ' if k else f'|   {blkID}: ') + sk + '\n'
                        
            for thisid, objs in alreadylisted.items():
                if len(objs) > 1:
                    s += f'|   {",".join(objs[1:])}: see {objs[0]} at {thisid}\n'

        if self.S is not None:
            s += f'|\n+ last evaluation at {self.f:_.1f} Hz: \n'
            for sk in str(self.S).split('\n'): 
                s += '|   ' + sk + '\n'
                
        s += '^'
        
        return s
    
    #===========================================================================
    #
    # _ _ l e n _ _
    #
    def __len__(self):
        return len(self.ports)
    
    #===========================================================================
    #
    # a d d b l o c k
    #
    def addblock(self, name, RFobj, ports=None, params={}, **kwargs):
        """addblock
            inputs:
                name : str an ID of the added block
                RFobj : the added rf object:
                            this object must minimally implement __len__ returning
                                the number of ports
                ports : a port mapping : 
                            - can be a list of length the number of ports
                            - of a dict mapping the PF object's portnames to new
                                names.
                            - or ommitted to generate a generic set of names
                params : the parameters that will be supplied to the RFobject's
                            getS(...) method
                kwargs:
                    xpos: the (relative) position of the RFobject's ports
                            - can be a scalar 
                            - a list of lenght the number of ports
                            
        """
        
        if name in self.blocks:
            # we cannot redefine a block once created
            raise ValueError(f'circuit.addblock["{name}"]: name already in use')
        
        # RFobj must have len method (this is also true for an nd.array
        #
        N = len(RFobj)
        
        oports = []
        if hasattr(RFobj,'ports'):
            ports = {} if ports is None else ports
            if isinstance(ports, dict):
                # remap port names (note: empty dict pulls the object's portnames)
                oports = [ports.pop(p, p) for p in RFobj.ports]
                # possibly the user supplied none existing port names
                if ports:
                    raise ValueError(
                        f'circuit.addblock["{name}"]: ports {ports} not defined')
        
        if not oports:
            ports = "%d" if ports is None else ports
            
            if isinstance(ports, dict):
                # this can only happen when the object has no attribute ports
                # otherwise oports would have been filled
                raise TypeError(
                    f'circuit.addblock["{name}"]: the object has no ports '
                     'attribute. Therfore a dict port mapping cannot be used.')
                
            elif isinstance(ports, list):
                oports = ports
    
            elif isinstance(ports, str):
                oports = [ports % k for k in range(1,N+1)]
                 
            else:
                raise ValueError(
                    'circuit.addblock: expecting str, dict or list to remap port names')
        
        if len(oports) != N:
            raise ValueError(
                f'circuit.addblock["{name}"]: RFobj len(ports)={len(oports)} '
                f'and len(RFobj)={N} mismatch')
        
        # check the position parameter xpos
        xpos = kwargs.pop('xpos', np.zeros(len(oports)))
        if hasattr(xpos,'__iter__'):
            if len(xpos) != N:
                raise ValueError(
                    f'circuit.addblock["{name}"]: xpos length does not match'
                    ' the object\' number of ports')
        else:
            xpos += np.zeros(len(oports))
            
        self.blocks[name] = {
            'params' : params,
            'ports'  : oports,
            'loc'    : self.M.shape,
            'object' : RFobj,
            'xpos'   : xpos
        } 
   
        # update the waves
        self.waves += [f'->{name}.{p}' for p in oports] \
                    + [f'<-{name}.{p}' for p in oports]
        
        
        # x = kwargs.pop('x',0)
        # for p in oports:
            # self.nodes[f'{name}.{p}'] = {'x' : x}
            #
        # if hasattr(RFobj, 'L'):
            # for p in oports[1:]:
                # self.nodes[f'{name}.{p}']['x'] = x + RFobj.L

        # update the available ports
        self.ports += [f'{name}.{p}' for p in oports]
        
        # prepare the M matrix
        #
        #               M   |     0
        #    M ->  ---------+----------
        #               0   |   S   -I
        self.M = np.vstack((
            np.hstack((
                self.M,
                np.zeros((self.M.shape[0],2*N),dtype=np.complex)
            )),
            np.hstack((
                np.zeros((N, self.M.shape[1]), dtype=np.complex),
                np.nan * np.ones((N, N), dtype=np.complex),
                -np.eye(N, dtype=np.complex)
            ))
        ))
        
        # update the equation types
        self.eqns += [f'S: {name}.{p}' for p in oports]
        
           
    #===========================================================================
    #
    # c o n n e c t
    #
    def connect(self, *ports):
        """connect
            inputs are existing as well as not yet existing ports
        """
        
        # create possibly missing nodes
        newports = [p for p in ports if ('->'+p) not in self.waves]
        for p in newports:
            self.waves.append('->'+p)
            self.waves.append('<-'+p) 
            self.ports.append(p)
        
        # do we need to reverse '->' and '<-' for new ports
        idxAs, idxBs = [], []
        for p in ports:
            A, B = ('<-', '->') if p in newports else ('->', '<-')
            idxAs.append(self.waves.index(A+p))
            idxBs.append(self.waves.index(B+p))
        
        N = len(idxAs)
        
        # update the port list (except for the new ports they are consumed)
        for p in ports:
            if p not in newports:
                self.ports.remove(p)
        
        # now we need to expand M for the new variables
        self.M = np.hstack((
            self.M, np.zeros((self.M.shape[0],2*len(newports)))
        ))
                
        # ideal Kirchhoff N-port junction
        SJ = np.array([[2/N - (1 if kc == kr else 0)
                        for kc in range(N)] for kr in range(N)],
                      dtype=np.complex)
        
        # the equations are 
        #   [[SJ]] . [B_ports] - [A_ports] = [0]
        
        for SJr, idxA, port in zip(SJ,idxAs, ports):
            # add a row to self.M
            self.M = np.vstack((
                self.M, 
                np.zeros((1, self.M.shape[1]), dtype=np.complex)
            ))
            self.M[-1,idxA] = -1
            
            # update the equation types
            m1 = '[' if port == ports[0] else ']' if port == ports[-1] else '+'
            self.eqns.append(f'C{m1} {port}')
            
            self.C[port] = idxA, idxBs
            
            for SJrc, idxB in zip(SJr,idxBs):
                self.M[-1, idxB] = SJrc
                            
    #===========================================================================
    #
    # t e r m i n a t e
    #
    def terminate(self, port, **kwargs):
        
        try:        
            idxA = self.waves.index('->'+port)
            idxB = self.waves.index('<-'+port) 
        except ValueError as e:
            msg = e.message if hasattr(e,'message') else e
            print(f'circuit.terminate: port not found: {msg}')
       
        if len(kwargs) > 1:
            raise ValueError(
                'circuit.terminate: only one of "RC", "Y", "Z" kwargs allowed')
        
        # update the port list
        self.ports.remove(port)
        
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
            rho = (Z - self.Zbase) / (Z + self.Zbase)
        elif 'Y' in kwargs:
            Y = kwargs.pop('Y')
            rho = (1 - Y * self.Zbase) / (1 + Y * self.Zbase)
        else:
            rho = kwargs.pop('RC', 0.)
            
        # equation is:
        # rho . B_port - A_port = 0
        
        # add a row to self.M
        self.M = np.vstack((
            self.M, 
            np.zeros((1, self.M.shape[1]), dtype=np.complex)
        ))
        self.M[-1,idxA] = -1
        self.M[-1,idxB] = rho
        
        # update the equations
        self.eqns.append(f'T: {port}')
        
        self.T[port] = idxA, idxB
    
    
    #===========================================================================
    #
    # e x t S
    #
    def extS(self):
        """
        extS returns the S-matrix solution of the current state of the 
           circuit matrix for the circuit's base impedance (self.Zbase)
           
           it is called after e.g. getS(f, Zbase, params) has recursively
           filled all the circuit's block S-matrices in its matrix.
        """
        
        if self.M.shape[0] != self.M.shape[1]:
            # we have not yet made self.M square by adding equation lines for
            # the external ports
            self.idxEs = []
            for p in self.ports:
                self.M = np.vstack((
                    self.M,
                    np.zeros((1, self.M.shape[1]), dtype=np.complex)
                ))
                self.M[-1,self.waves.index('->'+p)] = 1
                self.eqns.append(f'E: {p}')
                self.E[p] = self.M.shape[0]-1
                self.idxEs.append(self.M.shape[0]-1)
        
        idxAs = [self.waves.index('->'+p) for p in self.ports]
        idxBs = [self.waves.index('<-'+p) for p in self.ports]

        self.invM = np.linalg.inv(self.M)
        
        QA = self.invM[np.r_[idxAs],:][:,np.r_[self.idxEs]] # N x E
        try:
            invQA = np.linalg.inv(QA)
            
        except:
            print(self)
            print('invM:\n', strM(self.invM))
            print('invM[np.r[idxAs]]: \n', strM(self.invM[np.r_[idxAs],:]))
            print('idxAs:', [(k, self.waves[k]) for k in idxAs])
            print('idxBs:', [(k, self.waves[k]) for k in idxBs])
            print(QA)
            raise
        
        QB = self.invM[np.r_[idxBs],:][:,np.r_[self.idxEs]] # N x E
        self.S = QB @ invQA
        
        return self.S
    
    #===========================================================================
    #
    # s e t 
    #
    def set(self, **kwargs):
        
        for blkID, blk in self.blocks.items():
            try:
                blk['object'].set(**kwargs.pop(blkID,{}))
            except AttributeError:
                raise ValueError()
            
        self.invM = None # mark the circuit as not solved
        
        if kwargs:
            print(f'circuit.set: unused kwargs:\n{kwargs}')
            
    #===========================================================================
    #
    # g e t S 
    #
    def getS(self, fs, Zbase=None, params={}):
        
        def get1S(f):
            self.f = f
            for blkID, blk in self.blocks.items():
                                
                i1, i2 = blk['loc']
                N = len(blk['ports'])
                
                if hasattr(blk['object'], 'getS'):
                    try:
                        blk['object'].set(**params.get(blkID,{}))
                    except AttributeError:
                        pass
                    S = blk['object'].getS(f, Zbase=self.Zbase, 
                                           params=blk['params'])
                else:
                    S = blk['object']
                
                self.M[i1:i1+N,i2:i2+N] = S
            
            M = self.extS()
            self.f, self.S = f, M
            
            if Zbase and Zbase != self.Zbase:
                M = ConvertGeneral(Zbase,M,self.Zbase)
                
            return M
        
        Ss = []
        if hasattr(fs,'__iter__'):
            for f in fs:
                Ss.append(get1S(f))
            Ss = np.array(Ss)
        else:
            Ss = get1S(fs)
            
        return Ss

    #===========================================================================
    #
    # S o l u t i o n
    #
    def Solution(self, f, E, Zbase=None, level=0):
        """circuit.Solution(f. E, Zbase)
        
            returns a dict of (node, (V, I, Vf, Vr)) pairs for a excitation E at
            the fequency f [Hz] where node is the node's name, V its voltage, I 
            the current flowing into the node, Vf the forward voltage wave into 
            the node and Vr the voltage wave reflected from the node.
            
            E and the returned voltage wave quantities are expressed for a
            reference impedance Zbase which defaults to the circuit's one.
        """
        
        # TODO:
        #   - add kwargs to solve only for a selection of nodes
        #      caveat: still need to solve internal blocks 
        #   - add kwargs for a selection of quantities
        #   - loop on freuencies 
        
        Zbase = self.Zbase if Zbase is None else Zbase
        
        # incident voltage waves @  Zbase to the circuits feed ports
        Ea = [E[p] for p in self.ports]
        
        # reflected voltage waves @ Zbase from the circuit's feed ports
        Eb = self.getS(f,Zbase) @ Ea  # (implicitly updates and solves the circuit)
        
        # incident voltage waves @ self.Zbase
        Ei = ((Ea + Eb) + self.Zbase/Zbase * (Ea - Eb)) / 2  
        
        # build the excitation vector @ self.Zbase
        Es = np.zeros(self.M.shape[0], dtype=np.complex)
        for p, Ek in zip(self.ports, Ei):
            Es[self.E[p]] = Ek
        
        # get all the wave quantities @ self.Zbase
        self.sol = self.invM @ Es
        
        # get all the voltages and currents
        tSol = {}
        for kf, w in enumerate(self.waves):
            # only process forward waves and look up reflected waves
            if w[:2] == '->':
                node = w[2:]
                kr = self.waves.index(f'<-{node}')
                Vk = self.sol[kf] + self.sol[kr]
                Ik = (self.sol[kf] - self.sol[kr]) / self.Zbase
                Vf = (Vk + Zbase * Ik) / 2
                Vr = (Vk - Zbase * Ik) / 2
                tSol[node] = Vk, Ik, Vf, Vr
                
        # recursively solve the internal nodes of the blocks in the circuit
        if level:
            for blkID, blk in self.blocks.items():
                k0, obj = blk['loc'][0], blk['object']
                if hasattr(obj, 'Solution'):
                    
                    # -> build the excitation to pass on to the Solution method.
                
                    # the trick is to find the wave solutions corresponding to the
                    # object's ports. The problem is compounded because the addblock
                    # call might have renamed the underlying object's portnames ...
                
                    # we need to scan the waves for the '->{blkID}.{circuit portname}'
                    # these will show in the order of the object's portnames
    
                    blkports = f'->{blkID}.'
                    # print(f'blkports = {blkports}')
                    # so the circuit's port's solutions:
                    cps = [ s for w, s in zip(self.waves, self.sol) 
                           if w.find(blkports) == 0]
                    # print(f'cps = {cps}')
                
                    #and map it to the object's ports
                    Ek = dict(zip(obj.ports, cps))
    
                    tSolr = obj.Solution(f, Ek, Zbase, level-1)
                    
                    # collect and add the result to tSol
                    for tnode, tval in tSolr.items():
                        tSol[f'{blkID}.{tnode}'] = tval
                    
        return tSol
    
    #===========================================================================
    #
    # m a x V
    #
    def maxV(self,f, E, Zbase=None, ID='<top>'):
        Zbase = self.Zbase if Zbase is None else Zbase
        
        Ea = [E[p] for p in self.ports]
        Eb = self.getS(f,Zbase) @ Ea   # implicitly updates and solves the circuit
        Ei = ((Ea + Eb) + self.Zbase/Zbase * (Ea - Eb)) / 2 
        
        Es = np.zeros(self.M.shape[0], dtype=np.complex)
        for p, Ek in zip(self.ports, Ei):
            Es[self.E[p]] = Ek

        self.sol = self.invM @ Es
        for w, s, e in zip(self.waves,self.sol, Es):
            print(f'{w:10} {s.real:+10.6f}{s.imag:+10.6f}j  '
                  f'{e.real:+10.6f}{e.imag:+10.6f}j')
        print()
            
        tmax, nmax = - np.inf, 'N/A'
        for k, w in enumerate(self.waves):
            if w[:2] == '->':
                k1 = self.waves.index('<-'+w[2:])
                Vk = np.abs(self.sol[k]  + self.sol[k1])
                if Vk > tmax:
                    tmax, nmax = Vk, w[2:]
        
        tfig = pl.figure(f'{ID} VSWs')
        l0 = max([len(_) for _ in self.blocks])+1
        for blkID, blk in self.blocks.items():
            k0, obj = blk['loc'][0], blk['object']
            print(f'{blkID:{l0}s} -> k0 = {k0}:')

            # -> build the excitation to pass on to the object's maxV method.
            
            # the trick is to find the wave solutions corresponding to the
            # object's ports. The problem is compounded because the addblock
            # call might have renamed the underlying object's portnames ...
            
            # we need to scan the waves for the '->{blkID}.{circuit portname}'
            # these will show in the order of the object's portnames

            blkports = f'->{blkID}.'
            # print(f'blkports = {blkports}')
            # so the circuit's port's solutions:
            cps = [ s for w, s in zip(self.waves, self.sol) 
                    if w.find(blkports) == 0]
            # print(f'cps = {cps}')
            
            #and map it to the object's ports
            Ek = dict(zip(obj.ports, cps))
            # print(f'Ek: {Ek}')
            
            if hasattr(obj, 'maxV'):
                t1, n1 = blk['object'].maxV(f, Ek, self.Zbase, blkID)
                if t1 > tmax:
                    tmax, nmax = t1, f'{blkID}.{n1}' 
                if hasattr(obj,'VISWs'):
                    xs, (Vsw, *_) = obj.VISWs(f, Ek, self.Zbase)
                    xpos = blk['xpos']
                    xst = xpos[0]+(xs-xs[0])*(xpos[1]-xpos[0])/(xs[-1]-xs[0])
                    
                    pl.figure(tfig.number)
                    pl.plot(xst, np.abs(Vsw), '.-', label=blkID)
        
        pl.figure(tfig.number)
        pl.legend(loc='best')
        pl.xlabel('x [m]')
        pl.ylabel('U [kV]')
        pl.title(f'{ID} Voltage standing waves')
        pl.grid(True)
                    
        return tmax, nmax
            