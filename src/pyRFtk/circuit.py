if __name__ == '__main__':
    import sys
    sys.path.append('../pyRFtk test')
    print('importing test_circuit ...')
    import test_circuit                        # @UnresolvedImport @UnusedImport
    sys.exit(0)

import numpy as np
import matplotlib.pyplot as pl
from copy import deepcopy

from .ConvertGeneral import ConvertGeneral
from .tictoc import tic, toc
from .printMatrices import strM
from .whoami import whoami

from . import rfObject # the name may change ... e.g. rfTSF
from .config import logit, tLogger, ident, _newID, logident

#===============================================================================
#
# c i r c u i t
#
class circuit():
    """
    .. deprecated:: 1.0.0
        'circuit' has been replaced by 'rfCircuit' but kept here for compatibility with legacy code

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
"""
#TODO: rename and order external ports as requested
#TODO: rethink on how to set parameters
#TODO: check logic for the status of solved or not
#TODO: use external sNp if available

    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, **kwargs):
        from warnings import warn
        from inspect import stack        
        print(f"CALLER FUNCTION: {stack()[1].function}")
        print(f"CALLER FUNCTION: {stack()[2]}")
        warn('[depreciated] please use the rfCircuit class')
        
        self.Id = kwargs.pop('Id', 'circuit_' + _newID())
        self.Zbase = kwargs.pop('Zbase', 50.)
        sNp = kwargs.pop('sNp', None)
        self.Portnames = kwargs.pop('Portnames', [])
        
        if kwargs:
            print(f'unprocessed kwargs: {".".join([kw for kw in kwargs])}')
            
        self.M = np.array([],dtype=np.complex).reshape((0,0))
        self.ports = []
        self.waves = []
        self.nodes = {}
        self.blocks = {
        #   'params' : params,
        #   'ports'  : oports,
        #   'loc'    : self.M.shape,
        #   'object' : RFobj,
        #   'xpos'   : xpos
        }
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
    # _ _ c o p y _ _
    #
    def __copy__(self):
        
        # we deepcopy everything because the main idea of copying the circuit 
        # would be to e.g. keep a snapshot of the object ...
       
        return self.__deepcopy__()
    
     
    #===========================================================================
    #
    # _ _ d e e p c o p y _ _
    #
    def __deepcopy__(self, memo={}):
        debug = logit['DEBUG']
        debug and tLogger.debug(ident(
            f'> circuit.__deepcopy__ {self.Id}', 1
        ))
        other = type(self)()
        for attr, val in self.__dict__.items():
            try:
                debug and tLogger.debug(ident(f'copy {attr}'))
                other.__dict__[attr] = deepcopy(val)
            except:
                print(f'{whoami(__package__)}: failed to deepcopy {attr}')
                raise
        debug and tLogger.debug(ident(f'< circuit.__deepcopy__ {self.Id}', -1))
        return other
    
    #===========================================================================
    #
    # __state__
    #
    def __state__(self, d=None):
        pass
    
    #===========================================================================
    #
    # c o p y
    #
    def copy(self):
        return self.__copy__()
    
        # other = circuit(Zbase = self.Zbase)
        # other.Portnames = self.Portnames[:]
        # other.ports = self.ports[:]
        # other.M = self.M.copy()
        # other.waves = self.waves[:]
        # for blk, val in self.blocks.items():
        #     obj = val['object']
        #     other.blocks[blk] = {
        #         'params' : val['params'],
        #         'ports'  : val['ports'].copy(),
        #         'loc'    : val['loc'][:],
        #         'object' : obj.copy() if hasattr(obj,'copy') else obj,
        #         'xpos'   : val['xpos'][:]
        #     }
        # other.eqns = self.eqns[:]
        # other.f = self.f
        # other.C = self.C.copy()
        # other.T = self.T.copy()
        # other.T = self.E.copy()
        # other.invM = None
        # other.S = None
        # other.idxEs = self.idxEs[:]
        # if hasattr(self, 'sNp'):
        #     other.sNp = self.sNp.copy()
        #
        # return other
        
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, full=0):
        s = f'{type(self).__name__} Id="{self.Id}" at {hex(id(self))} (pyRFtk.circuit version {__updated__}) \n'
        s += f'| Zbase: {self.Zbase} Ohm\n'
        s += f'| ports: {self.ports} \n'
        
        if full == 0:
            return s
        
        s += '|\n'
        s += f'+ {self.M.shape[0]} equations, {self.M.shape[1]} unknowns \n'

        if self.M.shape[0] == 0:
            s += '| \n'
            s += '| <empty>\n^\n'
            return s
        
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
                        s +=  f'|   {blkID}: \n'
                        #fixme: embellish this type of block
                        mx = max([len(_p) for _p in blk['ports']])
                        for _p, _x in zip(blk['ports'], blk['xpos']):
                            s += f'|   |   {_p:{mx+1}s} @ {_x:7.4f}m\n'
                        s += f'|   |\n'

                        s += f'+   fixed numpy array like at {thisid}\n'
                        for k, sk in enumerate(str(blk['object']).split('\n')):
                            s += '|   | ' +  sk + '\n'
                        s += '|   ^\n'
                    else:
                        #print(blkID)
                        #print(blk["object"])
                        try:
                            s +=  f'|   {blkID}: \n'
                            mx = max([len(_p) for _p in blk['ports']])
                            if hasattr(blk['object'],'ports'):
                                oports = blk['object'].ports
                            else:
                                oports = ["-"]*len(blk['ports'])
                            omx = max([len(_p) for _p in oports])
                            for _p, _po, _x in zip(blk['ports'],oports, blk['xpos']):
                                s += f'|   |   {_p:{mx+1}s} -> {_po:{omx+1}s} @ {_x:7.4f}m\n'
                            s += f'|   |\n'
                            if full:
                                for k, sk in enumerate(blk["object"].__str__(full-1).split('\n')):
                                    s += ('|   'if k else '|   + ') + sk + '\n'
                        except TypeError:
                            print(f'{whoami(__package__)}: {blkID}')
                            raise
            for thisid, objs in alreadylisted.items():
                lmax = max([len(objk) for objk in objs])
                if len(objs) > 1:
                    for objk in objs[1:]:
                        s += f'|   {objk:{lmax+1}s}: --> {objs[0]} at {thisid}\n'

        if self.S is not None:
            s += f'|\n+ last evaluation at {self.f:_.1f} Hz: \n'
            for sk in str(self.S).split('\n'): 
                s += '|   ' + sk + '\n'
                
        s += '^'
        
        return s
    
    #===========================================================================
    #
    # l i s t B l o c k s
    #
    def listBlocks(self):
        blks = []
        for blkID, blk in self.blocks.items():
            blks.append(blkID)
            try:
                for sblkID in blk['object'].listBlocks():
                    blks.append(blkID+'.'+sblkID)
            except AttributeError:
                pass
    
        return blks
    
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
        debug = logit['DEBUG'] 
        debug and tLogger.debug(ident(
            f'> [circuit.addblock] '
            f'name= {name}, RFobj= {RFobj.Id}, '
            f'ports= {ports}, params={params}, kwargs= {kwargs}',
            1
        ))
        
        self.S = None
        self.invM = None
        
        if '.' in name:
            debug and tLogger.debug(ident(f'---> replacing {name}'))
            name, subblock = name.split('.',1)
            if name not in self.blocks:
                raise ValueError(
                    f'{whoami(__package__)}: [update subblock {name+"."+subblock}]:'
                    f' {name} not present.')
                
            debug and tLogger.debug(ident(f'copying "{self.Id}".blocks[{name}]["object"]'))
            subblockobj = self.blocks[name]['object'].copy()
            debug and tLogger.debug(ident(f'adding {subblock} to "{self.Id}".blocks[{name}]'))
            subblockobj.addblock(subblock, RFobj, ports=ports, params=params, **kwargs)
            self.blocks[name]['object'] = subblockobj
            if '.' not in subblock:
                self.blocks[name]['params'] = params
                
            debug and tLogger.debug(ident(
                f'< [circuit.addblock] (replaced at a deeper level)', -1
            ))
            return
            
        elif name in self.blocks:
            if len(RFobj) != len(self.blocks[name]['object']):
                raise ValueError(
                    f'{whoami(__package__)}: [update {name}] number of ports mismatch: '
                    f'existing {len(self.blocks["object"])}, new {len(RFobj)}')
            
            #TODO: maybe some other checks are in order / necessary
            
            self.blocks[name]['object'] = RFobj
            
            debug and tLogger.debug(ident(
                f'< [circuit.addblock] (replaced at this level)', -1
            ))
            return
                
        
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
                        f'{whoami(__package__)}["{name}"]: ports {ports} not defined')
        
        if not oports:
            ports = "%d" if ports is None else ports
            
            if isinstance(ports, dict):
                # this can only happen when the object has no attribute ports
                # otherwise oports would have been filled
                raise TypeError(
                    f'{whoami(__package__)}["{name}"]: the object has no ports '
                     'attribute. Therfore a dict port mapping cannot be used.')
                
            elif isinstance(ports, list):
                oports = ports
    
            elif isinstance(ports, str):
                oports = [ports % k for k in range(1,N+1)]
                 
            else:
                raise ValueError(
                    f'{whoami(__package__)}: expecting str, dict or list to remap port names')
        
        if len(oports) != N:
            raise ValueError(
                f'{whoami(__package__)}["{name}"]: RFobj len(ports)={len(oports)} '
                f'and len(RFobj)={N} mismatch')
        
        # check the position parameter xpos
        
        xpos = kwargs.pop('xpos', None)
        if xpos == None:
            xpos = np.zeros(len(oports))
            
        if hasattr(xpos,'__iter__'):
            if len(xpos) != N:
                raise ValueError(
                    f'{whoami(__package__)}["{name}"]: xpos length does not match'
                    ' the object\'s number of ports')
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

        debug and tLogger.debug(ident(
            f'< [circuit.addblock] (inserted a new block)', -1
        ))
        
        return

        
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
                f'{whoami(__package__)}: only one of "RC", "Y", "Z" kwargs allowed')
        
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
        debug = logit['DEBUG']
        debug and tLogger.debug(ident(
            f'> [circuit.extS] ',
            1
        ))
        
        if self.M.shape[0] != self.M.shape[1]:
            debug and tLogger.debug(ident(f'M is not square {self.M.shape}'))
            self.S = None
            
            # we have not yet made self.M square by adding equation lines for
            # the external ports
            self.idxEs = []
            
            # if external port names are available we use that ordering
            if self.Portnames:
                if len(self.Portnames) != len(self.ports):
                    raise ValueError(
                        f'{whoami(__package__)}: incompatible number of anounced ports set '
                        f'via kwarg Portnames [{len(self.Portnames)}] and the resolved '
                        f'number of free external ports [{len(self.ports)}].'
                    )
                else:
                    for p in self.ports:
                        if p not in self.Portnames:
                            raise ValueError(
                                f'{whoami(__package__)}: resolved free port name {p} not found '
                                f'in list of announced port names'
                            )
                # use external port ordering
                self.ports = self.Portnames
                
            else:
                self.Portnames = self.ports                   
                            
            for p in self.ports:                
                self.M = np.vstack((
                    self.M,
                    np.zeros((1, self.M.shape[1]), dtype=np.complex)
                ))
                self.M[-1,self.waves.index('->'+p)] = 1
                self.eqns.append(f'E: {p}')
                self.E[p] = self.M.shape[0]-1
                self.idxEs.append(self.M.shape[0]-1)
               
        debug and tLogger.debug(ident(f'Portnames= {self.Portnames}'))
        debug and tLogger.debug(ident(f'ports=     {self.ports}'))
        
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
            print(f'{whoami(__package__)}')
            raise
        
        QB = self.invM[np.r_[idxBs],:][:,np.r_[self.idxEs]] # N x E
        self.S = QB @ invQA
        
        if debug:
            tLogger.debug(ident(f'S:'))
            ss = strM(self.S,
                      pfun=lambda z: (np.abs(z), np.angle(z, deg=1)),
                      pfmt='%8.5f %+7.2f'+u'\N{DEGREE SIGN}'+', '
                     )
            for sl in ss.split('\n'):
                if sl:
                    tLogger.debug(ident(f'  {sl}'))
                
        debug and tLogger.debug(ident(f'< [circuit.extS]',-1))
        return self.S
    
    #===========================================================================
    #
    # s e t 
    #
    def set(self, **kwargs):
        
        for blkID, blk in self.blocks.items():
            if hasattr(blk,'set'):
                blk['object'].set(**kwargs.pop(blkID,{}))
            elif blkID in kwargs:
                raise ValueError(
                    f'{whoami(__package__)}: tried to set params for {blkID} '
                    f'but the object has no \'set\' attribute')
            
        self.invM = None # mark the circuit as not solved
        
        if kwargs:
            print(f'circuit.set: unused kwargs:\n{kwargs}')
            
    #===========================================================================
    #
    # g e t S 
    #
    def getS(self, fs, Zbase=None, params={}, flags={}):
        
        debug = logit['DEBUG'] 
        debug and tLogger.debug(ident(
            f'> [circuit.getS] '
            f'fs= {fs}, Zbase= {Zbase}, params= {params}, flags= {flags}',
            1
        ))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        def get1S(f):
            
            # fixme: circuit object should not know about gtl object ...
            #        but it does know about sNp...
            
            if hasattr(self, 'sNp') and flags.get('sNp', True) :
                # note this bypasses the solving of the nested RF objects
                debug and tLogger.debug(ident(f'using sNp'))
                    
                M = self.sNp.getS(f, Zbase=self.Zbase)

            else:
                # note the port order is not controlled unless the user
                # has added code to ensure the port order !
                
                if debug:
                    msg = ('no sNp' if not hasattr(self, 'sNp') else 
                           ('flag sNp set to False' if 'sNp' in flags else '(why?)'))
                    tLogger.debug(ident(f'not using sNp: {msg}'))
                    
                if f != self.f or self.S is None or self.flags != flags:
                    
                    # we need to recompute the S-matrix
                    
                    for blkID, blk in self.blocks.items():
                        debug and tLogger.debug(ident(
                            f'getting S for {blkID}'
                        ))
                                        
                        i1, i2 = blk['loc']
                        N = len(blk['ports'])
                        
                        if hasattr(blk['object'], 'getS'):
                            try:
                                blk['object'].set(**params.get(blkID,{}))
                            except AttributeError:
                                pass
                            # recursively get the lower level S-matrices
                            S = blk['object'].getS(f, Zbase=self.Zbase, 
                                                   params=blk['params'], 
                                                   flags=flags)
                        else:
                            # this is when the object is a np.array
                            S = blk['object']
                        
                        self.M[i1:i1+N,i2:i2+N] = S
                                                        
                    M = self.extS()
                    self.f, self.S, self.flags = f, M, flags.copy()
                    
                else:
                    M = self.S

            if Zbase and Zbase != self.Zbase:
                M = ConvertGeneral(Zbase,M,self.Zbase)
                
            return M
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
        Ss = []
        if hasattr(fs,'__iter__'):
            for f in fs:
                Ss.append(get1S(f))
            Ss = np.array(Ss)
        else:
            Ss = get1S(fs)
        
        if debug:
            if hasattr(fs, '__iter__'):
                tLogger.debug(ident(f'Ss: [{len(Ss)} matrices]'))
                fss, Sss = fs, Ss
            else:
                tLogger.debug(ident(f'Ss: 1 matrix'))
                fss, Sss = [fs], [Ss]
                
            for fk, Sk in zip(fss,Sss):
                tLogger.debug(ident(f'f= {fk} Hz'))
                ss = strM(Sk,
                          pfun=lambda z: (np.abs(z), np.angle(z, deg=1)),
                          pfmt='%8.5f %+7.2f'+u'\N{DEGREE SIGN}'+', '
                         )
                for sl in ss.split('\n'):
                    if sl:
                        tLogger.debug(ident(f'  {sl}'))

        debug and tLogger.debug(ident('< [circuit.getS]',-1))
        return Ss

    #===========================================================================
    #
    #  G e t _ S m a t r i x
    #
    def Get_internalSmatrix(self, f, nodes, Zbase=None):
        """given a set of (internal) nodes try and find a corresponding Smatrix
        """
        
        Zbase = self.Zbase if Zbase is None else Zbase
        
        # as nodes can be inside deeper levels an "easy" solution as was possible
        # in pyRFtk.circuit_Class3a.Get_Smatrix is no longer possible
        
       
        # first we get all the solutions for exciting each external port successively
        
        tSols = []
        for p in self.ports:
            E = dict([(p_, 1. if p_ == p else 0.) for p_ in self.ports])
            tnodes = [node[1:] if node[0] == '<' else node for node in nodes]
            tSols.append(self.Solution(f, E, Zbase, tnodes))   
        
        # we construct the matrix MA_Es and MB_Es which will give the relation
        # between the nodes forward and reflected waves from the nodes (reversed
        # when flaged by the < sign preceeding the node name) and the external
        # excitation ports of the circuit
 
        MA_Es, MB_Es = [],[]
        for node in nodes:
            iA, iB = 2, 3 
            if node[0] == '<':
                node = node[1:]
                iA, iB = 3, 2
                
            MA_E, MB_E = [], []
            for tSolk in tSols:
                Solk = tSolk[node]
                MA_E.append(Solk[iA])
                MB_E.append(Solk[iB])
            MA_Es.append(MA_E)
            MB_Es.append(MB_E)
            
        S = np.array(MB_Es) @ np.linalg.inv(MA_Es)
            
        return S
    
    #===========================================================================
    #
    # s o l v e
    #
    def solve(self, f, E, Zbase=None, flags={}):
        """circuit.solve(f, E, Zbase)
        
            solves the circuit for the excitation given in Zbase (defaults to
            self.Zbase) and flags (defaults to no flags) at frequency f
            
            returns the solution at all nodes        
        """
        debug = logit['DEBUG']
        debug and tLogger.debug(ident(
            f'> [circuit.solve] '
            f'f= {f}, E= {E}, Zbase= {Zbase}, flags= {flags}',
            1
        ))
        Zbase = self.Zbase if Zbase is None else Zbase
        
        # incident voltage waves @  Zbase to the circuits feed ports
        Ea = [E[p] for p in self.ports]
        
        # reflected voltage waves @ Zbase from the circuit's feed ports
        Eb = self.getS(f,Zbase,flags=flags) @ Ea  # (implicitly updates and solves the circuit)
        
        # incident voltage waves @ self.Zbase
        Ei = ((Ea + Eb) + self.Zbase/Zbase * (Ea - Eb)) / 2  
        
        # build the excitation vector @ self.Zbase
        Es = np.zeros(self.M.shape[0], dtype=np.complex)
        for p, Ek in zip(self.ports, Ei):
            Es[self.E[p]] = Ek
        
        # if the returned S matrix came from a touchstone then invM was not 
        # computed
        if self.invM is None:
            S = self.extS()
            
        # get all the wave quantities @ self.Zbase
        self.sol = self.invM @ Es
        debug and tLogger.debug(ident('< [circuit.solve]', -1))
        
        
    #===========================================================================
    #
    # S o l u t i o n
    #
    def Solution(self, f, E, Zbase=None, nodes=None, flags={}):
        """circuit.Solution(f. E, Zbase=self.Zbase, nodes=None, flags={})
        
            returns a dict of (node, (V, I, Vf, Vr)) pairs for a excitation E at
            the fequency f [Hz] where node is the node's name, V its voltage, I 
            the current flowing into the node, Vf the forward voltage wave into 
            the node and Vr the voltage wave reflected from the node.
            
            E and the returned voltage wave quantities are expressed for a
            reference impedance Zbase which defaults to the circuit's one.
        """
        debug = logit['DEBUG']
        debug and tLogger.debug(ident(
            f'> [circuit.Solution] '
            f'f= {f}, E= {E}, Zbase= {Zbase}, nodes= {nodes}, flags= {flags}',
            1
        ))
        
        # TODO:
        #   - add kwargs for a selection of quantities
        #   - loop on frequencies 
        
        Zbase = self.Zbase if Zbase is None else Zbase
        
        # incident voltage waves @  Zbase to the circuits feed ports
        # Ea is a list of excitations in the order of definition of the 
        #    ports of the object
        Ea = [E[p] for p in self.ports]
        
        # reflected voltage waves @ Zbase from the circuit's feed ports
        # unless gtl is True the sNp based value for S is returned
        Eb = self.getS(f,Zbase,flags=flags) @ Ea  # (implicitly updates and solves the circuit)
        
        # incident voltage waves @ self.Zbase
        Ei = ((Ea + Eb) + self.Zbase/Zbase * (Ea - Eb)) / 2  
        
        if hasattr(self, 'sNp') and flags.get('sNp', True):
            # need only to solve for the exernal nodes
            tSol = {}
            
            for node, Eak, Ebk  in zip(self.ports, Ea, Eb):
                tSol[node] = Eak + Ebk, (Eak - Ebk)/Zbase, Eak, Ebk
        
        else:
            # need to solve also all internal nodes as gtl was enforced
            
            # build the excitation vector @ self.Zbase
            Es = np.zeros(self.M.shape[0], dtype=np.complex)
            for p, Ek in zip(self.ports, Ei):
                Es[self.E[p]] = Ek
            
            # if the returned S matrix came from a touchstone then invM was not 
            # computed
            debug and tLogger.debug(ident(f'invM is None ? {self.invM is None}'))
            if self.invM is None:
                debug and tLogger.debug(ident(f'force recomputation of invM with extS'))
                self.extS()
                
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
                
        if debug:
            tLogger.debug(ident(f'tSol:'))
            ls = max([len(_) for _ in tSol])
            for node, (Vn, In, An, Bn) in tSol.items():
                tLogger.debug(ident(
                    f'  {node:{ls}s} : {Vn:15.3f}V, {In:15.3f}A,'
                    f' {An:13.3f}V+, {Bn:13.3f}V-'
                ))
        
        if nodes is None:
            nodesleft = None
            
        elif nodes:
            nodesleft = [node for node in nodes if node not in tSol]
            
        else:
            nodesleft = []
        
        debug and tLogger.debug(ident(f'nodesleft= {nodesleft}'))
               
        # recursively solve the internal nodes of the blocks in the circuit
        if nodesleft is None or nodesleft:
            debug and tLogger.debug(ident(f'further looking for {nodesleft}'))
            # if nodesleft is not None:
            #     # strip the leading blockname
            #     snodesleft = [node.split('.')[-1] for node in nodesleft]
                
            for blkID, blk in self.blocks.items():
                debug and tLogger.debug(ident(f'analyzing {blkID}'))
                obj = blk['object']

                check = nodesleft is None or any(
                    [n.split('.',1)[0] == blkID for n in nodesleft])
                
                snodesleft = None
                if nodesleft is not None:
                    # strip the leading blockname
                    snodesleft = [node.split('.',1)[-1] for node in nodesleft 
                                  if node.split('.',1)[0] == blkID]
                
                debug and tLogger.debug(ident(f'snodesleft: {snodesleft}'))
                
                # if hasattr(obj, 'Solution') and check:
                if hasattr(obj, 'Solution') and snodesleft:   
                    # -> build the excitation to pass on to the Solution method.    
                    
                    Ek = {}
                    for pobj in obj.ports:
                        try:
                            Ek[pobj] = tSol[f'{blkID}.{pobj}'][2]
                            # print(f'[OK] {blkID}.{pobj} -> {Ek[pobj]:7.3f}')
                        except KeyError:
                            for k, s in tSol.items():
                                print(f'[EE] {k:30s} -> {s[2]:7.3f}')
                            print(f'{whoami(__package__)}')
                            raise
    
                    if debug:
                        tLogger.debug(ident(f'  port excitations:'))
                        ls = max([len(_) for _ in Ek])
                        for _p, _v in Ek.items():
                            tLogger.debug(ident(f'    {_p:{ls+1}s}: {_v:25.3f}V'))
                   
                    
                    tSolr = obj.Solution(f, Ek, Zbase, snodesleft, flags=flags)
                    
                    # collect and add the result to tSol
                    for tnode, tval in tSolr.items():
                        tSol[f'{blkID}.{tnode}'] = tval
                        
        debug and tLogger.debug(ident('< [cicuit.Solution]',-1))
        return tSol
    
    #===========================================================================
    #
    # m a x V
    #
    def maxV(self,f, E, Zbase=None, Id=None, flags={}, **kwargs):
        
        """
        kwargs : future extesnsion ?
        xpos: list-like position of ports
        """
        
        _debug_ = logit['DEBUG']
        _debug_ and logident(f'>',printargs=True)
        
        Id = Id if Id else self.Id
        
        # The circuit object itself does not know where it is located : if called
        # recursively the xpos position kwargs is passed from the level above
        
        xpos = kwargs.get('xpos', None)
        if xpos is None:
            # we did not receive any info on the position
            if hasattr(self,'xpos'):
                xpos = self.xpos
            else:
                xpos = [0.] * len(self.ports)
                
        elif hasattr(xpos,'__iter__') and len(xpos) == len(self):
            pass # xpos is the position of the object's ports
        
        elif isinstance(xpos, (int, float)):
            if hasattr(self,'xpos'):
                xpos += np.aray(self.xpos)
            else:
                xpos = np.zeros(len(self))
            xpos = list(xpos)
        
        else:
            msg = (f' could not understand supplied xpos: {type(xpos)} '
                   f'{("[%d]"%len(xpos)) if hasattr(xpos,"__iter__") else ""}')
            _debug_ and logident(msg)
            raise ValueError(f'{whoami(__package__)}: {msg}')
            
        ppos = dict(zip(self.ports, xpos))
        
        if _debug_:
            logident(f'{self.Id} port positions')
            for _p, _pos in ppos.items(): 
                logident(f'  {_p} : {_pos:7.3f}m')
        
        Zbase = self.Zbase if Zbase is None else Zbase
        
        try:
            Ea = [E[p] for p in self.ports]
        except KeyError:
            undef = [p for p in self.ports if p not in E]
            raise ValueError(
                f'{whoami(__package__)}: undeclared ports {undef}'
            ) from None
        
        # print('pyRFtk.circuit.maxV')
        Eb = self.getS(f, Zbase, flags=flags) @ Ea   # implicitly updates and solves the circuit
                                        # except if a sNp is present but in that case ...

        # Ei are the forward waves into to circuit's ports at self.Zbase
        Ei = ((Ea + Eb) + self.Zbase/Zbase * (Ea - Eb)) / 2 
        
        # Es is the excitation : M . Ewaves = Es
        Es = np.zeros(self.M.shape[0], dtype=np.complex)
        for p, Ek in zip(self.ports, Ei):
            Es[self.E[p]] = Ek

        # if the returned S matrix came from a touchstone then invM was not 
        # computed
        if self.invM is None or f != self.f:
            S = self.extS()

        self.sol = self.invM @ Es
        Esol = dict((w,s) for w,s in zip(self.waves, self.sol))
        
        if False:
            for w, s, e in zip(self.waves, self.sol, Es):
                print(f'{w:10} {s.real:+10.6f}{s.imag:+10.6f}j  '
                      f'{e.real:+10.6f}{e.imag:+10.6f}j')
            print()
            
        # find the maximum voltage on the nodes
        VSWs = {Id:{}}
        tmax, nmax = - np.inf, 'N/A'
        for k, w in enumerate(self.waves):
            if w[:2] == '->':
                k1 = self.waves.index('<-'+w[2:])
                Vk = np.abs(self.sol[k]  + self.sol[k1])
                # note the circuit object itself does not know where it is
                # positioned; it only keeps track of the blocks/objects inside it
                # there is some logic to this as the circuit object itself can be
                # used in several places in an encompassing circuit object
                if w[2:] not in ppos:
                    VSWs[Id][w[2:]]=([None],[Vk])
                else:
                    _debug_ and logident(f'{w[2:]} found in ppos: {ppos[w[2:]]:7.3f}m')
                    VSWs[Id][w[2:]]=([ppos[w[2:]]],[Vk])
                if Vk > tmax:
                    tmax, nmax = Vk, w[2:]
        
        # l0 = max([len(_) for _ in self.blocks])+1
        if _debug_:
            ls = max([len(_) for _ in VSWs[Id]])
            tLogger.debug(ident(f'VSWs:'))
            for nd,vl in VSWs[Id].items():
                _ = '  N/A  ' if vl[0][0] is None else f'{vl[0][0]:7.3f}'
                tLogger.debug(ident(
                    f'  {nd:{ls+1}s}: [{_}]m, [{vl[1][0]:7.3f}]V'
                ))
        
        for blkID, blk in self.blocks.items():
            _debug_ and tLogger.debug(ident(f'finding maxV for {blkID}'))
            obj = blk['object']
            if hasattr(obj, 'ports'):
                # if the object is a numpy array then it will have no maxV anyway
                 
                # print(f'{blkID:{l0}s} -> k0 = {k0}:')
    
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
                
                if _debug_:
                    tLogger.debug(ident(f'  port excitations:'))
                    ls = max([len(_) for _ in Ek])
                    for _p, _v in Ek.items():
                        tLogger.debug(ident(f'    {_p:{ls+1}s}: {_v:25.3f}V'))
                   
                if hasattr(obj, 'maxV'):
                    try:
                        t1, n1, VSW = obj.maxV(f, Ek, self.Zbase, blkID, 
                                               flags=flags,
                                               xpos = blk['xpos'])
                        if len(VSW) == 1 and blkID in VSW:
                            _debug_ and tLogger.debug(ident(
                                f'squeezing return maxV dict'
                            ))
                            VSW = VSW[blkID]
                    except TypeError:
                        print(blkID)
                        print(obj)
                        raise
                    if isinstance(VSW, tuple): # if underlying structure VSW is a dict
                        xs, xpos = np.array(VSW[0]), blk['xpos']
                        if xs[-1] != xs[0]:
                            xst = xpos[0]+(xs-xs[0])*(xpos[1]-xpos[0])/(xs[-1]-xs[0])
                        else:
                            xst = xpos[0] * np.ones(xs.shape)
                        VSW = (xst, VSW[1])
                        
                    VSWs[Id][blkID] = VSW
                    if t1 > tmax:
                        tmax, nmax = t1, f'{blkID}.{n1}' 
                else:
                    _debug_ and tLogger.debug(ident(f'{blkID} has no maxV attribute'))
        
        _debug_ and tLogger.debug(ident('< [circuit.maxV]',-1))            
        return tmax, nmax, VSWs
    
    #===========================================================================
    #
    # p l o t V S W s 
    #
    def _obsolete_plotVSWs(self, f, E, Zbase=None, Id='<top>'):
        
        Vmax, where, VSWs = self.maxV(f, E, Zbase=Zbase, Id=Id)
        
        tfig = pl.figure(f'{Id} VSWs')
        
        def plotit(BLK, VSW):
            if isinstance(VSW, dict):
                for blk, vsw in VSW.items():
                    plotit(blk, vsw)
            elif  isinstance(VSW, tuple):
                xs, absV = VSW
                if any([x != xs[0] for x in xs]) or any([v != absV[0] for v in absV]):
                    pl.plot(xs, absV, '.-', label=BLK)

        plotit(Id, VSWs)
                    
        pl.figure(tfig.number)
        pl.legend(loc='best')
        pl.xlabel('x [m]')
        pl.ylabel('U [kV]')
        pl.title(f'{Id} Voltage standing waves')
        pl.grid(True)
        
        return Vmax, where, VSWs

            
