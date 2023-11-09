__updated__ = "2023-11-09 10:00:05"

import numpy as np
import matplotlib.pyplot as plt
import warnings

from .ConvertGeneral import ConvertGeneral
from .printMatrices import strM
from .whoami import whoami

from . import rfBase, rfObject 
from .config import logit, tLogger, ident, logident

#===============================================================================
#
# r f C i r c u i t
#
class rfCircuit(rfBase):
    """
    this implements a circuit class for manipulating RFbase objects, it depends on the 
    :class: 'rfObject' class which must implement following methods/attributes
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
    TODO: check logic for the status of solved or not
    TODO: use external sNp if available

    """

    #===========================================================================
    #
    # _ _ i n i t _ _
    #
    def __init__(self, *args, **kwargs):
        
        _debug_ = logit['DEBUG']
        _debug_ and logident('>', printargs=False)

        type_rfCircuit = type(self).__name__ == 'rfCircuit'
        
        if not hasattr(self,'args'):
            self.args = args
        if not hasattr(self,'kwargs'):
            self.kwargs = kwargs.copy()
        
        super().__init__(**kwargs)
        
        # pop the kwargs consumed by rfBase (self.kwused is initialized in rfBase)
        for kw in self.kwused:
            kwargs.pop(kw, None)
        
        sNp = kwargs.pop('sNp', None)
        
        # why Portnames ? ports is processed already by rfBase
        self.Portnames = kwargs.pop('Portnames', [])
        
        if kwargs and type_rfCircuit:
            msg = f'unprocessed kwargs: {", ".join([kw for kw in kwargs])}'
            _debug_ and logident(msg)
            warnings.warn(msg)
            
        self.M = np.array([],dtype=complex).reshape((0,0))
        # self.ports = [] # FIXedME: resets ports set by rfBase from kwargs
        self.waves = []
        self.nodes = {}
        self.blocks = {
        #   'params' : params,
        #   'ports'  : oports,
        #   'loc'    : self.M.shape,
        #   'object' : RFobj,
        #   'xpos'   : xpos  <== is now also an object propery (caveat the
                            #    same opject could be in different positions
                            #    in a circuit)
        }
        self.eqns = []
        self.f = np.nan # set in rfBase to None
        self.C = {}
        self.T = {}
        self.E = {}                         # E[port] -> eqn number
        self.idxEs = []
        self.invM = None
        self.S = None # set in rfBase to None
        
        if sNp:
            self.sNp = rfObject(touchtone=sNp)
        
        _debug_ and logident('<')
    
    #===========================================================================
    #
    # __state__
    #
    def __state__(self, d=None):
        pass
            
    #===========================================================================
    #
    # _ _ s t r _ _
    #
    def __str__(self, full=0):
        s = super().__str__(full=0)
        plines = '\n|  '.join(self.ports)
        s += f'\n| ports: [\n|  {plines}\n| ]\n'
        s += '|\n'
        s += f'+ {self.M.shape[0]} equations, {self.M.shape[1]} unknowns \n'
            
        if not full:
            return s
        
        # s += f'\n| ports: {self.ports} \n'
        # s += '|\n'
        # s += f'+ {self.M.shape[0]} equations, {self.M.shape[1]} unknowns \n'
        s += '| \n'
            
        if self.M.shape[0] == 0:
            s += '| <empty>\n^\n'
            return s
        
        l1 = max([len(eqn) for eqn in self.eqns])
        l2 = max(len(wave) for wave in self.waves)
        for k, (Mr, eqn) in enumerate(zip(self.M, self.eqns)):
            s += f'| {k:3d} {self.waves[k]:<{l2}s} [{k:3d}] {eqn:<{l1}s} '
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
            elif typ[0] == 'D':
                # brute force: we are just looking for non-zeros
                for idx, sij in enumerate(Mr):
                    if -1E-6 < (np.abs(sij) - 1) < 1E-6:
                        s += f'[{idx:3d}] {sij.real:+7.4f}{sij.imag:+7.4f}j '
            elif typ[0] == 'T':
                idxs = sorted(self.T[port][:-1])
                for idx in idxs:
                    sij = Mr[idx]
                    s += f'[{idx:3d}] {sij.real:+7.4f}{sij.imag:+7.4f}j '
            elif typ[0] == 'E':
                idx = self.waves.index('->'+port)
                sij = Mr[idx]
                s += f'[{idx:3d}] {sij.real:+7.4f}{sij.imag:+7.4f}j '
            s += '\n'
        
        for k, p in zip(range(self.M.shape[0], len(self.waves)), self.ports):
            s += f'| {k:3d} {self.waves[k]:<{l2}s} [{k:3d}] E? {p:<{l1}s}\n'
        
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
                            for _p, _po, _x in zip(blk['ports'], oports, blk['object'].xpos):
                                _x += blk['xpos']
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
    # f i n d O b j
    #
    def findObj(self, location):
        
        debug = logit['DEBUG']       
        debug and logident(f'>', printargs=True)
        
        obj = self
        try:
            while location:
                blk, location = (location.split('.', 1) + [''])[:2] 
                debug and logident(f'loc: {blk}, arc: {location}')
                obj = obj.blocks[blk]['object']
        except KeyError:
            if debug:
                for s in obj.asstr(1).split('\n'):
                    logident(s)
                if hasattr(obj,'blocks'):
                    for itm in obj.blocks:
                        logident(itm)
            logident(f'could not find {blk} in obj.blocks')
            obj = None
            
        debug and logident(f'<')
        
        return obj
    
    #===========================================================================
    #
    # _ s u b s t p o r t s
    #
    def _substports(self, name, RFobj, ports):
        
        _debug_ = logit['DEBUG'] 
        _debug_ and tLogger.debug(ident(
            f'> [circuit._substports] '
            f'name= {name}, RFobj= {RFobj.Id}, '
            f'ports= {ports}' ,
            1
        ))
        
        # RFobj must have len method (this is also true for an nd.array
        #
        N = len(RFobj)
        
        oports = []
        if hasattr(RFobj,'ports'):
            ports = {} if ports is None else ports
            if isinstance(ports, dict):
                # remap port names (note: empty dict pulls the object's portnames)
                oports = [ports.pop(_p, _p) for _p in RFobj.ports]
                # possibly the user supplied non-existing port names
                if ports:
                    raise ValueError(
                        f'{whoami(__package__)}["{name}"]: ports {ports} not defined')
              
        if not oports:  # thus ports supplied was not a dict 
                        # [or there were no RFobj.ports e.g. an array obj]
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
                raise ValueError( f'{whoami(__package__)}: expecting str, dict '
                                  'or list to remap port names')  
        
        if len(oports) != N:
            raise ValueError(
                f'{whoami(__package__)}["{name}"]: RFobj len(ports)={len(oports)} '  
                f'and len(RFobj)={N} mismatch')
        
        _debug_ and tLogger.debug(ident(
            f'< [circuit._substports], N={N}, oports={oports}', -1
        ))
        
        return N, oports

    #===========================================================================
    #
    # d e e m b e d
    #
    def deembed(self, IntPorts={}, ExtPorts={}):
        """deembeds ports
        State of the rfCircuit object: it solves for (Sinternal is not
        explicitly known)
        
        

        :param IntPorts: list of size 2 tuples or dict
                [ (rfCircuit.port, rfCircuit.port_new), ... ] or
                { rfCircuit.port:rfCircuit.port_new, ... }
                
                these are the "internal" ports of the circuit i.e. these are
                connected through the RFobj
                
        :param ExtPorts: list of size 2 tuples or dict
                [ (rfCircuit.port, rfCircuit.port), ... ] or
                { rfCircuit.port:rfCircuit.port, ... }
                
                these are the "external" ports of the circuit i.e. these are
                connected through the RFobj
                
                    """
        
        _debug_ = logit['DEBUG'] 
        _debug_ and tLogger.debug(ident(
            f'> [circuit.deembed] IntPorts= {IntPorts}, ExtPorts= {ExtPorts}',
            1
        ))

        self.S = None 
        self.invM = None
        
        if isinstance(IntPorts, dict):
            IntPorts = [( dp, p) for dp, p in IntPorts.items()] 
        
        if isinstance(ExtPorts, dict):
            ExtPorts = [( dp, p) for dp, p in ExtPorts.items()] 
        
        # validate partially input IntPorts and ExtPorts
        
        for Ports, typ in zip([IntPorts, ExtPorts],['Int','Ext']):
            
            if not (
                isinstance(Ports, list) 
                and all( [ isinstance(t, tuple) and len(t)==2 and 
                           all(isinstance(q, str) for q in t) for t in Ports
                           
                         ]
                       )
               ):
                
                raise ValueError(
                    f'{whoami(__package__)}: {typ}Ports: expecting dict or '
                    'list of tuples to remap port names'
                )
            
        
        # add the columns for the internal ports
        
        self.M = np.hstack((
                    self.M,
                    np.zeros((self.M.shape[0],2*len(IntPorts)),dtype=complex)
                ))
        
        # add the new internal ports and find the remaining ports of the
        # deembed object
          
        for dp, ip in IntPorts:
            
            if ip in self.ports:
                raise ValueError(
                    f'{whoami(__package__)}: (internal) port {ip} already exists '
                )
            
            # create the new internal port
            self.waves += [f'->{ip}', f'<-{ip}']
            self.ports.append(ip)
                        
            if dp not in self.ports:
                raise ValueError(
                    f'{whoami(__package__)}: (external) port {dp} does not exist'
                )
                
            idxA = self.waves.index(f'->{dp}')
            idxB = self.waves.index(f'<-{dp}')
            
            # add two rows to self.M
            self.M = np.vstack((
                self.M, 
                np.zeros((2, self.M.shape[1]), dtype=complex)
            ))
        
            self.M[-2, idxA] = 1
            self.M[-2, len(self.waves)-1] = -1
            self.M[-1, idxB] = 1
            self.M[-1, len(self.waves)-2] = -1
            
            # remove the deembeded port
            self.ports.pop(self.ports.index(dp))
            
            # update the equation types
            self.eqns.append(f'Di {dp}')
            self.eqns.append(f'Di {ip}')
            
        
        for dp, ep in ExtPorts:
            
            if dp not in self.ports:
                raise ValueError(
                    f'{whoami(__package__)}: (external) port {dp} does not exist'
                )
                
            if ep not in self.ports:
                raise ValueError(
                    f'{whoami(__package__)}: (external) port {ep} does not exist'
                )
                
            # add two rows to self.M
            self.M = np.vstack((
                self.M, 
                np.zeros((2, self.M.shape[1]), dtype=complex)
            ))
        
            idxAd = self.waves.index(f'->{dp}')
            idxBd = self.waves.index(f'<-{dp}')

            idxAe = self.waves.index(f'->{ep}')
            idxBe = self.waves.index(f'<-{ep}')

            self.M[-2, idxAd] = 1
            self.M[-2, idxAe] = -1
            self.M[-1, idxBd] = 1
            self.M[-1, idxBe] = -1
            
            # update the equation types
            self.eqns.append(f'De {dp}')
            self.eqns.append(f'De {ep}')
            
            # now we need to remove the remaining ports of the port list
            self.ports.pop(self.ports.index(dp))
            self.ports.pop(self.ports.index(ep))
                        
        _debug_ and tLogger.debug(ident(
            f'< [circuit.deembed]', -1
        ))
        
        return

    #===========================================================================
    #
    # a d d b l o c k
    #
    def addblock(self, name, RFobj, ports=None, params={}, **kwargs):

        """adds a previously defined circuit block to the circuit

        :param name: str an ID of the added block
        :param RFobj: the added rf object:
                            this object must minimally implement __len__ returning
                                the number of ports
        :param ports: a port mapping : 
                            - can be a list of length the number of ports
                            - of a dict mapping the PF object's portnames to new
                                names.
                            - or ommitted to generate a generic set of names

        :param params: the parameters that will be supplied to the RFobject's
                            getS(...) method
        :param kwargs: relpos: the (relative) position of the RFobject
                            
        """

        _debug_ = logit['DEBUG'] 
        _debug_ and tLogger.debug(ident(
            f'> [circuit.addblock] '
            f'name= {name}, RFobj= {RFobj.Id}, '
            f'ports= {ports}, params={params}, kwargs= {kwargs}',
            1
        ))
        
        self.S = None
        self.invM = None
        
        if '.' in name:
            _debug_ and logident(f'---> replacing {name}')
            name, subblock = name.split('.',1)
            if name not in self.blocks:
                raise ValueError(
                    f'{whoami(__package__)}: [update subblock {name+"."+subblock}]:'  
                    f' {name} not present.')
                
            _debug_ and logident(f'copying "{self.Id}".blocks[{name}]["object"]')
            
            # we need to copy the subblock as otherwise we may also change the
            # properties of other instances of the subblock used else where in
            # the circuit
            
            try:
                subblockobj = self.blocks[name]['object'].copy()
            except AttributeError:
                print(type(self).__name__)
                print(self.Id)
                raise
            
            _debug_ and logident(f'adding {subblock} to "{self.Id}".blocks[{name}]')
            
            subblockobj.addblock(subblock, RFobj, ports=ports, params=params, **kwargs)
            self.blocks[name]['object'] = subblockobj
            if '.' not in subblock:
                self.blocks[name]['params'] = params
                
            _debug_ and logident(
                f'< [circuit.addblock] (replaced at a deeper level)', -1
            )
            return
            
        elif name in self.blocks:
            if len(RFobj) != len(self.blocks[name]['object']):
                raise ValueError(
                    f'{whoami(__package__)}: [update {name}] number of ports mismatch: ' 
                    f'existing {len(self.blocks["object"])}, new {len(RFobj)}')
            
            #TODO: maybe some other checks are in order / necessary
            
            self.blocks[name]['object'] = RFobj
            
            _debug_ and tLogger.debug(ident(
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
                oports = [ports.pop(_p, _p) for _p in RFobj.ports]
                # possibly the user supplied non-existing port names
                if ports:
                    raise ValueError(
                        f'{whoami(__package__)}["{name}"]: ports {ports} not defined')
        
        if not oports: # thus ports supplied was not a dict [or there were no RFobj.ports e.g. an array obj]
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
        
        if 'xpos' in kwargs:
            warnings.warn(f'{whoami(__package__)}: xpos depreciated; use relpos')  
            relpos = kwargs.pop('xpos', 0.)
        else:
            relpos = kwargs.pop('relpos', 0.)
            
        # if hasattr(xpos,'__iter__'):
        #     if len(xpos) != N:
        #         raise ValueError(
        #             f'{whoami(__package__)}["{name}"]: xpos length does not match'
        #             ' the object\'s number of ports')
        # else:
        #     xpos += np.zeros(len(oports))
            
        self.blocks[name] = {
            'params' : params,
            'ports'  : oports,
            'loc'    : self.M.shape,
            'object' : RFobj,
            'xpos'   : relpos # but this will not work for an np.array
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
                np.zeros((self.M.shape[0],2*N),dtype=complex)
            )),
            np.hstack((
                np.zeros((N, self.M.shape[1]), dtype=complex),
                np.nan * np.ones((N, N), dtype=complex),
                -np.eye(N, dtype=complex)
            ))
        ))
        
        # update the equation types
        self.eqns += [f'S{"[" if k==0 else "]" if k == len(oports)-1 else "+"}' 
                      f' {name}.{p}' for k, p in enumerate(oports)]
        
        # update the self.xpos
        # self.xpos = []
        for p, x in zip(oports, RFobj.xpos):
            self.xpos.append(relpos + x)
                
        _debug_ and tLogger.debug(ident(
            f'< [circuit.addblock] (inserted a new block)', -1
        ))
        
        return

        
    #===========================================================================
    #
    # g e t p o s
    #
    def getpos(self, node):
        """return the position of the node

        :param node: The node
        """
        relpos, obj = 0., self
        while '.' in node:
            try:
                onode = node # belts and braces :)
                blk, node = (node.split('.',1)+[None])[:2]
                block = obj.blocks[blk]
                relpos += block['xpos']
                obj = block['object']
            except AttributeError: # because obj is not a circuit (= no .blocks)
                node = onode
                break
            except KeyError:
                # because the node's toplevel block is this self self ?
                return self.getpos(node)
        try:  
            kp =block['ports'].index(node)
        except:
            kp = obj.ports.index(node)
            
        return relpos + obj.xpos[kp]
                
    #===========================================================================
    #
    # c o n n e c t
    #
    def connect(self, *ports):

        """connects the specified ports in the circuit. 

        :params ports: existing as well as not yet existing ports

        """
        
        _debug_ = logit['DEBUG'] 
        _debug_ and tLogger.debug(ident(
            f'> [circuit.connect] '
            f'ports= {ports}',
            1
        ))

        # create possibly missing nodes
        newports = [p for p in ports if ('->'+p) not in self.waves]
        _debug_ and tLogger.debug(ident(f'new ports: {", ".join(newports)}', 0))
        
        if len(newports) > 1:
            raise ValueError(
                f'{whoami(__package__)}: cannot have more than one new port'  
            )
        elif len(newports) and '.' in newports[0]:
            raise ValueError(
                f"{whoami(__package__)}: a new port name can't contain a '.'"  
            )
            
        oldports = [p for p in ports if ('->'+p) in self.waves]
        _debug_ and tLogger.debug(ident(f'old ports: {", ".join(oldports)}', 0))
        
        for p in newports:
            self.waves.append('->'+p)
            self.waves.append('<-'+p) 
            self.ports.append(p)
            # here we need to make assumptions on the position of the new port
            # -> we take the position of the first port in the list
            self.xpos.append(self.getpos(oldports[0]))
            
        _debug_ and tLogger.debug(ident(f'self.ports = {", ".join(self.ports)}', 0))
        
        
        # do we need to reverse '->' and '<-' for new ports
        idxAs, idxBs = [], []
        for p in ports:
            A, B = ('<-', '->') if p in newports else ('->', '<-')
            idxAs.append(self.waves.index(A+p))
            idxBs.append(self.waves.index(B+p))
        
        N = len(idxAs)
        
        # update the port list (except for the new ports they are consumed)
        for _p in ports:
            if _p not in newports:
                _kp = self.ports.index(_p)
                self.ports.pop(_kp)
                self.xpos.pop(_kp)
                _debug_ and tLogger.debug(ident(f'deleted port {_p}', 0))
                
        _debug_ and tLogger.debug(ident(f'self.ports -> {", ".join(self.ports)}', 0))
                
        # now we need to expand M for the new variables
        self.M = np.hstack((
            self.M, np.zeros((self.M.shape[0],2*len(newports)))
        ))
                
        # ideal Kirchhoff N-port junction
        SJ = np.array([[2/N - (1 if kc == kr else 0)
                        for kc in range(N)] for kr in range(N)],
                      dtype=complex)
        
        # the equations are 
        #   [[SJ]] . [B_ports] - [A_ports] = [0]
        
        for SJr, idxA, port in zip(SJ,idxAs, ports):
            # add a row to self.M
            self.M = np.vstack((
                self.M, 
                np.zeros((1, self.M.shape[1]), dtype=complex)
            ))
            self.M[-1,idxA] = -1
            
            # update the equation types
            m1 = '[' if port == ports[0] else ']' if port == ports[-1] else '+'
            self.eqns.append(f'C{m1} {port}')
            
            self.C[port] = idxA, idxBs
            
            for SJrc, idxB in zip(SJr,idxBs):
                self.M[-1, idxB] = SJrc
                
        self.invM, self.S = None, None

        _debug_ and tLogger.debug(ident(
            f'< [circuit.connect]', -1
        ))
                            
    #===========================================================================
    #
    # t e r m i n a t e
    #
    def terminate(self, port, **kwargs):

        """terminates the given port, can be either to ground by specifying some Z value
        or leave it open by specifying Y=0

        :param port: The port to terminate
        :param kwargs: With what to terminate the port (e.g Z=5)

        """
        
        _debug_ = logit['DEBUG'] 
        _log_ = lambda *args: _debug_ and tLogger.debug(ident(*args))
        
        _log_(f'> [circuit.terminate] port= "{port}", kwargs= {kwargs}', 1)

        
        try:        
            idxA = self.waves.index('->'+port)
            idxB = self.waves.index('<-'+port) 
        except ValueError as e:
            msg = e.message if hasattr(e,'message') else e
            print(f'circuit.terminate: port not found: {msg}')
       
        if len(kwargs) > 1:
            raise ValueError(
                f'{whoami(__package__)}: only one of "RC", "Y", "Z" kwargs allowed')  
                
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
            rho = (Z - self.Zbase) / (Z + self.Zbase)
        elif 'Y' in kwargs:
            Y = kwargs.pop('Y')
            rho = (1 - Y * self.Zbase) / (1 + Y * self.Zbase)
        else:
            rho = kwargs.pop('RC', 0.)
            
        if kwargs:
            raise ValueError(
                f'rfCircuit.terminate: unknown kwargs: {", ".join([_ for _ in kwargs])}.')
            
        # equation is:
        # rho . B_port - A_port = 0
        
        if port in self.T:
            # this port was already terminated
            _debug_ and tLogger.debug(ident(f'port already terminated'))
        
            idxA, idxB, eqn = self.T[port]
            self.M[eqn, idxB] = rho
            self.invM = None
        
        elif port in self.E:
            # this port was and external port
            if _debug_:
                tLogger.debug(ident(f'port was an external port'))
                _log_(f'"{port}" -> idxA: {idxA}, idxB: {idxB}')
                tLogger.debug(ident(f'self.E["{port}"] = {self.E[port]}'))
                for k, mp in enumerate(self.M[self.E[port],:]):
                    if np.abs(mp) > 0.:
                        tLogger.debug(ident(f'self.M[{k}] -> {mp}'))
                        
            eqn = self.E[port]
            self.M[eqn, idxA] = -1.
            self.M[eqn, idxB] = rho
            self.eqns[eqn] = f'T: {port}'
            self.T[port] = idxA, idxB, eqn
            self.E.pop(port)
            _kp = self.ports.index(port)
            self.ports.pop(_kp)
            self.xpos.pop(_kp)
            self.invM = None
            self.S = None
                        
        elif False and (port in self.C):
            
            # 2023/11/08 : (FDe) disabled this branch with "False and ..."
            # trying to terminate a port newly created would erroneously cause
            # a port already connected error
                                            
            _debug_ and tLogger.debug(ident(f'port already connected !'))
            raise ValueError(
                f'{whoami(__package__)}: port {port} already conneccted' 
            )
            
        else:
            # this port was not terminated yet
            _debug_ and tLogger.debug(ident(f'port not yet used'))
            # update the port list
            try:
                _kp = self.ports.index(port)
                self.ports.pop(_kp)
                self.xpos.pop(_kp)
                
            except IndexError:
                raise ValueError(
                    f'{whoami(__package__)}: port {port} already in use'  
                )
        
            # add a row to self.M
            self.M = np.vstack((
                self.M, 
                np.zeros((1, self.M.shape[1]), dtype=complex)
            ))
            self.M[-1,idxA] = -1
            self.M[-1,idxB] = rho
            
            # update the equations
            self.eqns.append(f'T: {port}')
            
            self.T[port] = idxA, idxB, self.M.shape[0] - 1
            
            self.invM, self.S = None, None

        _debug_ and tLogger.debug(ident(
            f'< [circuit.terminate]', -1
        ))

    #===========================================================================
    #
    # e x t S
    #
    def extS(self):
        """returns the S-matrix solution of the current state of the 
        circuit matrix for the circuit's base impedance (self.Zbase)
        it is called after e.g. getS(f, Zbase, params) has recursively
        filled all the circuit's block S-matrices in its matrix.
        """
        _debug_ = logit['DEBUG']
        _debug_ and tLogger.debug(ident(
            f'> [circuit.extS] ',
            1
        ))
        
        if self.M.shape[0] != self.M.shape[1]:
            _debug_ and tLogger.debug(ident(f'M is not square {self.M.shape}'))
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
                    np.zeros((1, self.M.shape[1]), dtype=complex)
                ))
                self.M[-1,self.waves.index('->'+p)] = 1
                self.eqns.append(f'E: {p}')
                self.E[p] = self.M.shape[0]-1
                # self.idxEs.append(self.M.shape[0]-1) # <===== also need to change for different port order
               
        _debug_ and tLogger.debug(ident(f'Portnames= {self.Portnames}'))
        _debug_ and tLogger.debug(ident(f'ports=     {self.ports}'))
        
        ports = self.Portnames
        self.idxEs = [ self.E[p] for p in ports]
        idxAs = [self.waves.index('->'+p) for p in ports] # <===== also need to change for different port order
        idxBs = [self.waves.index('<-'+p) for p in ports] # <===== also need to change for different port order

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
        
        if _debug_:
            tLogger.debug(ident(f'S:'))
            ss = strM(self.S,
                      pfun=lambda z: (np.abs(z), np.angle(z, deg=1)),
                      pfmt='%8.5f %+7.2f'+u'\N{DEGREE SIGN}'+', '
                     )
            for sl in ss.split('\n'):
                if sl:
                    tLogger.debug(ident(f'  {sl}'))
                
        _debug_ and tLogger.debug(ident(f'< [circuit.extS]',-1))
        return self.S
    
    #===========================================================================
    #
    # s e t 
    #
    def set(self, *args, **kwargs):
        """set
            kwargs: dict of 
                        ("blkID", sdict) 
                            where sdict is a similar dict that goes further into
                            the chain of nested rf objects
                        (">attr", value)
                            where attr is an rf object's attribute that gets the
                            value
            
            e.g.
               rfCircuitA
               |
               +- rfRLC1
               |  |
               |  +- Ls
               |
               +- rfCircuitB
                  |
                  +- rfTRL1
                     |
                     +- L
            
            rfCircuitA.set(rfRLC1= {"Ls":20e-9},
                           rfCircuitB= {"rfTRL1": {"L":1.0}}
            or
            
            rfCircuitA.set({'rfRLC1.Ls':20e-9, 
                            'rfCircuitB.rfTRL1.L':1.0'})
                            
            or
            
            rfCircuitA.set('rfRLC1.Ls', 20e-9)
            rfCircuitA.set('rfCircuitB.rfTRL1', {'L:1.0})
            
            
            The purpose of the different formalisms:
            
                .set('kw', {attr1: val1, attr2: val2}, ...)
            or  .set(('kw', {attr1: val1, attr2: val2}), ...)
            or  .set(kw = {attr1: val1, attr2: val2}, ...)
            
            is to be able to set multiple attributes at once.
            
            in case a single attribute needs to be set:
            
                 .set('kw.attr', val, ...)
            or   .set(('kw.attr', val), ... )
            but  .set(kw.attr = val, ... )               will not work !!
            thus .set(kw = {'attr': val}, ... }
            
            The kwarg formulation is more restrictive if one wants to loop
            over multiple objects where one wants to set the attributes.
            
            Possibly the kwarg formalism need to be cancelled ?
            
        """
        #TODO: set should be made recursive as well
        
        debug = logit['DEBUG']
        debug and logident(f'>', printargs=True)
        
        modified = False

        # unpack args and add them to kwargs
        if args:
            
            debug and logident('if args -> True')
            debug and logident(str(args))
            
            largs = list(args)
            while largs:
                ak = largs.pop(0)
                debug and logident(f'  ak = {ak}')
                
                if isinstance(ak, tuple) and len(ak) == 2:
                    kwargs = dict([ak], **kwargs)
                    
                elif isinstance(ak,str) and len(largs)>0:
                    vk = largs.pop(0)
                    kwargs = dict([(ak,vk)], **kwargs)
                
                else:
                    raise ValueError(
                        f'{whoami(__package__)}: {self.Id} '
                        'did not find a tuple of length 2 or a \'kw\' value pair'
                        ' in supplied args'
                    )
                    
            
            debug and logident(f'-> kwargs = {kwargs}')
            
        # progess the kwargs
        debug and logident('starting kwargs loop ...')

        processed = []
        for kw, val in kwargs.items():
            
            debug and logident(f'  processing kw={kw},val={val}')  
            processed.append(kw)
            
            if not isinstance(val, dict):
                # need to check if type val makes sense: expect int, float or complex
                
                # in this case the kw ends with the attribute to be modified
                # so we change the kw = obj.attr, val  to obj, {attr, val}
                try:
                    kw, attr = kw.rsplit('.',1)
                except:
                    raise ValueError(
                        f'{whoami(__package__)}: {self.Id} '
                        f'as the value supplied is not a dict the object locator'
                        f' \'{kw}\' should terminate with .attr of the object'
                        ' whose attribute is to be set'
                    )
                val = {attr: val}
                 
            # kw now points to an object
            block, rest = (kw.split('.',1) + [''])[:2]
            obj = self.blocks[block]['object']
            # recursively call the block's set method
            try:
                if rest:
                    # block is a rfCircuit itself
                    modified |= obj.set(rest, val)
                
                else:
                    # block is an rf object with a set method: e.g. rfRLC, ...
                    modified |= obj.set(**val)
                    
            except AttributeError:
                raise ValueError(
                    f'{whoami(__package__)}: {obj.Id} has no \'set\' attribute'
                )
            
            except:
                raise
            
            attr = [kw for kw in val][0]
            debug and logident(
                f'{obj.Id}.{attr} -> {getattr(obj,attr)}'
            )
                       
        if modified:
            self.invM = None # mark the circuit as not solved
            self.f = None
            self.S = None
            
        # removes the processed kwargs 
        for kw in processed:
            kwargs.pop(kw)
               
        if kwargs:
            print(f'{whoami(__package__)}: {self.Id} : unused kwargs:\n{kwargs}')
        
        debug and logident(f'<')
        
        return modified
    
    #===========================================================================
    #
    # g e t S 
    #
    def getS(self, fs, Zbase=None, params={}, flags={}):
        
        _debug_ = logit['DEBUG'] 
        _debug_ and tLogger.debug(ident(
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
                _debug_ and tLogger.debug(ident(f'using sNp'))
                    
                M = self.sNp.getS(f, Zbase=self.Zbase)

            else:
                # note the port order is not controlled unless the user
                # has added code to ensure the port order !
                
                if _debug_:
                    msg = ('no sNp' if not hasattr(self, 'sNp') else 
                           ('flag sNp set to False' if 'sNp' in flags else '(why?)'))
                    tLogger.debug(ident(f'not using sNp: {msg}'))
                    
                if f != self.f or self.S is None or self.flags != flags:
                    
                    # we need to recompute the S-matrix
                    
                    for blkID, blk in self.blocks.items():
                        _debug_ and tLogger.debug(ident(
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
        
        if _debug_:
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

        _debug_ and tLogger.debug(ident('< [circuit.getS]',-1))
        return Ss

    #===========================================================================
    #
    #  G e t _ S m a t r i x
    #
    def Get_internalSmatrix(self, f, nodes, Zbase=None):
        """given a set of (internal) nodes try and find a corresponding Smatrix
        """
                
        debug = logit['DEBUG']
        debug and logident(f'>', printargs=True)
        
        Zbase = self.Zbase if Zbase is None else Zbase
        
        # as nodes can be inside deeper levels an "easy" solution as was possible
        # in pyRFtk.circuit_Class3a.Get_Smatrix is no longer possible
        
        self.getS(f)    # if the circuit was not yet completed for external ports 
                        # this will force the definition of the ports
       
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
        
        dims = np.array(MA_Es).shape
        if dims[0] == dims[1]:
            S = np.array(MB_Es) @ np.linalg.inv(MA_Es)
        elif dims[0] < dims[1]:
            S = np.array(MB_Es) @ np.linalg.pinv(MA_Es)
        else:
            msg = 'cannot get S-matrix for more nodes than the number of exernal ports'
            debug and logident(msg)
            raise ValueError(msg)
        
        debug and logident(f'<')
        return S
    
    #===========================================================================
    #
    # s o l v e
    #
    def solve(self, f, E, Zbase=None, flags={}):
        """circuit.solve(f, E, Zbase)
        
            solves the circuit for the excitation given in Zbase (defaults to
            self.Zbase) and flags (defaults to no flags) at frequency f
            
            sets the solution at all waves in self.sol        
        """
        _debug_ = logit['DEBUG']
        _debug_ and tLogger.debug(ident(
            f'> [circuit.solve] '
            f'f= {f}, E= {E}, Zbase= {Zbase}, flags= {flags}',
            1
        ))
        Zbase = self.Zbase if Zbase is None else Zbase
        
        # incident voltage waves @  Zbase to the circuits feed ports
        Ea = [E[p] for p in self.ports]
        if _debug_ :
            tLogger.debug(ident(f'self.ports = {self.ports}'))
            tLogger.debug(ident(f'Ea = {Ea}'))
        
        # reflected voltage waves @ Zbase from the circuit's feed ports
        Eb = self.getS(f,Zbase,flags=flags) @ Ea  # (implicitly updates and solves the circuit)
        
        # incident voltage waves @ self.Zbase
        Ei = ((Ea + Eb) + self.Zbase/Zbase * (Ea - Eb)) / 2  
        
        # build the excitation vector @ self.Zbase
        Es = np.zeros(self.M.shape[0], dtype=complex)
        for p, Ek in zip(self.ports, Ei):
            Es[self.E[p]] = Ek
        
        # if the returned S matrix came from a touchstone then invM was not 
        # computed
        if self.invM is None:
            self.extS()
                        
        # get all the wave quantities @ self.Zbase
        self.sol = self.invM @ Es
        _debug_ and tLogger.debug(ident('< [circuit.solve]', -1))
        
        
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
            
            nodes: None, [ nodes ]
            
                If the requested nodes input is None all nodes are evaluated 
                recursively in the whole circuit. 
                
                If nodes is [] then anyway the nodes at the level if the object 
                are evaluated but the nodes in subblocks are not evaluated.
                
                So there may be more nodes evaluated than requested. 
        """
        _debug_ = logit['DEBUG']
        _debug_ and tLogger.debug(ident(
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
            self.tSol = {}
            
            for node, Eak, Ebk  in zip(self.ports, Ea, Eb):
                self.tSol[node] = Eak + Ebk, (Eak - Ebk)/Zbase, Eak, Ebk
        
        else:
            # need to solve also all internal nodes as gtl was enforced
            
            # build the excitation vector @ self.Zbase
            Es = np.zeros(self.M.shape[0], dtype=complex)
            for p, Ek in zip(self.ports, Ei):
                Es[self.E[p]] = Ek
            
            # if the returned S matrix came from a touchstone then invM was not 
            # computed
            _debug_ and tLogger.debug(ident(f'invM is None ? {self.invM is None}'))
            if self.invM is None:
                _debug_ and tLogger.debug(ident(f'force recomputation of invM with extS'))
                self.extS()
                
            # get all the wave quantities @ self.Zbase
            self.sol = self.invM @ Es
            
            # get all the voltages and currents
            self.tSol = {}
            for kf, w in enumerate(self.waves):
                # only process forward waves and look up reflected waves
                if w[:2] == '->':
                    node = w[2:]
                    kr = self.waves.index(f'<-{node}')
                    Vk = self.sol[kf] + self.sol[kr]
                    Ik = (self.sol[kf] - self.sol[kr]) / self.Zbase
                    Vf = (Vk + Zbase * Ik) / 2
                    Vr = (Vk - Zbase * Ik) / 2
                    self.tSol[node] = Vk, Ik, Vf, Vr
                
        if _debug_:
            tLogger.debug(ident(f'tSol:'))
            ls = max([len(_) for _ in self.tSol])
            for node, (Vn, In, An, Bn) in self.tSol.items():
                tLogger.debug(ident(
                    f'  {node:{ls}s} : {Vn:15.3f}V, {In:15.3f}A,'
                    f' {An:13.3f}V+, {Bn:13.3f}V-'
                ))
        
        if nodes is None:
            nodesleft = None
            
        elif nodes:
            nodesleft = [node for node in nodes if node not in self.tSol]
            
        else:
            nodesleft = []
        
        _debug_ and tLogger.debug(ident(f'nodesleft= {nodesleft}'))
               
        # recursively solve the internal nodes of the blocks in the circuit
        if nodesleft != []:
            _debug_ and tLogger.debug(ident(f'further looking for {nodesleft}'))
            # if nodesleft is not None:
            #     # strip the leading blockname
            #     snodesleft = [node.split('.')[-1] for node in nodesleft]
                
            for blkID, blk in self.blocks.items():
                _debug_ and tLogger.debug(ident(f'analyzing {blkID}'))
                obj = blk['object']

                check = nodesleft is None or any(
                    [n.split('.',1)[0] == blkID for n in nodesleft])
                
                snodesleft = None
                if nodesleft is not None:
                    # strip the leading blockname
                    snodesleft = [node.split('.',1)[-1] for node in nodesleft 
                                  if node.split('.',1)[0] == blkID]
                
                _debug_ and tLogger.debug(ident(f'snodesleft: {snodesleft}'))
                
                # if hasattr(obj, 'Solution') and check:
                if hasattr(obj, 'Solution') and (snodesleft != []):   
                    # -> build the excitation to pass on to the Solution method.    
                    
                    Ek = {}
                    for pobj in obj.ports:
                        try:
                            Ek[pobj] = self.tSol[f'{blkID}.{pobj}'][2]
                            # print(f'[OK] {blkID}.{pobj} -> {Ek[pobj]:7.3f}')
                        except KeyError:
                            for k, s in self.tSol.items():
                                print(f'[EE] {k:30s} -> {s[2]:7.3f}')
                            print(f'{whoami(__package__)}')  
                            raise
    
                    if _debug_:
                        tLogger.debug(ident(f'  port excitations:'))
                        ls = max([len(_) for _ in Ek])
                        for _p, _v in Ek.items():
                            tLogger.debug(ident(f'    {_p:{ls+1}s}: {_v:25.3f}V'))
                   
                    
                    tSolr = obj.Solution(f, Ek, Zbase, snodesleft, flags=flags)
                    
                    # collect and add the result to tSol
                    for tnode, tval in tSolr.items():
                        self.tSol[f'{blkID}.{tnode}'] = tval
                        
        _debug_ and tLogger.debug(ident('< [cicuit.Solution]',-1))
        return self.tSol
    
    #===========================================================================
    #
    # r e s o l v e _ x p o s
    #
    def resolve_xpos(self):
        """tries to build and xpos based on the underlying blocks' information
        """
        rXpos = [None] * len(self.ports)
        
        for _kp, _p in enumerate(self.ports):
            _x = rXpos[_kp]
            if _x is not None:
                continue
            blkID, blkport = (_p.split('.',1) + [None])[:2]
            if blkport is not None:
                obj = self.blocks[blkID]['object']
                try:
                    rXpos[_kp] = self.blocks[blkID]['xpos'][obj.ports.index(blkport)]
                except ValueError:
                    raise ValueError(
                        f'rfCircuit.resolve_xpos: port {blkport} is not in block {blkID}'
                    )
                except Exception as e:
                    raise RuntimeError(
                        f'rfCircuit.resolve_xpos: while checking for {_p} {e} occured'
                    )
                
            if rXpos[_kp] is None:
                rXpos[_kp] = 0.
            
        self.xpos = rXpos
        
    #===========================================================================
    #
    # m a x V
    #
    def maxV(self,f, E, Zbase=None, Id=None,flags={}, **kwargs):
        
        """
        kwargs : future extension ?
        """
        
        _debug_ = logit['DEBUG']
        _debug_ and logident(f'>',printargs=True)
        
        Id = Id if Id else self.Id
        
        # The circuit object itself does not know where it is located : if called
        # recursively the xpos position kwargs is passed from the level above
        
        relpos = kwargs.get('xpos', 0.)
        # if xpos is None:
        #     # we did not receive any info on the position
        #     if hasattr(self,'xpos'):
        #         xpos = self.xpos
        #     else:
        #         xpos = [0.] * len(self)
        #
        # elif hasattr(xpos,'__iter__') and len(xpos) == len(self):
        #     pass # xpos is the position of the object's ports
        #
        # elif isinstance(xpos, (int, float)):
        #     if hasattr(self,'xpos'):
        #         xpos = list(np.aray(self.xpos) + xpos)
        #     else:
        #         xpos = [xpos] * len(self)
        #
        # else:
        #     msg = (f' could not understand supplied xpos: {type(xpos)} '
        #            f'{("[%d]"%len(xpos)) if hasattr(xpos,"__iter__") else ""}')
        #     _debug_ and logident(msg)
        #     raise ValueError(f'{whoami(__package__)}: {msg}')
            
        ppos = dict([(f'{Id}.{_p}', 
                      (self.xpos[_kp] if isinstance(self.xpos[_kp], (float,int)) else 0.) + relpos)
                      for _kp, _p in enumerate(self.ports)])
        if _debug_:
            logident('ppos = {')
            for _key, _val in ppos.items():
                logident(f'  {_key} :{_val},')
            logident('}')
            
        Zbase = self.Zbase if Zbase is None else Zbase
        
        try:
            Ea = [E[p] for p in self.ports]
        except KeyError:
            undef = [p for p in self.ports if p not in E]
            raise ValueError(
                f'{whoami(__package__)}: undeclared ports {undef}'  
            ) from None
        
        # print('pyRFtk2.circuit.maxV')
        Eb = self.getS(f, Zbase, flags=flags) @ Ea   # implicitly updates and solves the circuit
                                        # except if a sNp is present but in that case ...

        # Ei are the forward waves into the circuit's ports at self.Zbase
        Ei = ((Ea + Eb) + self.Zbase/Zbase * (Ea - Eb)) / 2 
        
        # Es is the excitation : M . Ewaves = Es
        Es = np.zeros(self.M.shape[0], dtype=complex)
        for p, Ek in zip(self.ports, Ei):
            Es[self.E[p]] = Ek

        # if the returned S matrix came from a touchstone then invM was not 
        # computed
        if self.invM is None or f != self.f:
            self.extS()

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
                
                x = relpos + self.getpos(w[2:])
                VSWs[Id][w[2:]] = ([x], [Vk]) 
                    
                if Vk > tmax:
                    tmax, nmax = Vk, Id+'.'+w[2:]
                        
                    
        
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
                
                #and map it to the object's ports [is it clear that ports and cps are in order ?]
                Ek = dict(zip(obj.ports, cps))
                # print(f'Ek: {Ek}')
                
                if _debug_:
                    tLogger.debug(ident(f'  port excitations:'))
                    ls = max([len(_) for _ in Ek])
                    for _p, _v in Ek.items():
                        tLogger.debug(ident(f'    {_p:{ls+1}s}: {_v:25.3f}V'))
                   
                if hasattr(obj, 'maxV'):
                    try:
                        _debug_ and logident(f'{blkID}: xpos = {blk["xpos"]}')
                        t1, n1, VSW = obj.maxV(f, Ek, self.Zbase, blkID, 
                                               flags=flags,
                                               xpos = relpos + blk['xpos'])
                        
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
                        xs, xpos = np.array(VSW[0]), relpos + blk['xpos']
                        xpos = [xpos + obj.xpos[_kp] for _kp in range(len(obj))]
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
        
        tfig = plt.figure(f'{Id} VSWs')
        
        def plotit(BLK, VSW):
            if isinstance(VSW, dict):
                for blk, vsw in VSW.items():
                    plotit(blk, vsw)
            elif  isinstance(VSW, tuple):
                xs, absV = VSW
                if any([x != xs[0] for x in xs]) or any([v != absV[0] for v in absV]):
                    plt.plot(xs, absV, '.-', label=BLK)

        plotit(Id, VSWs)
                    
        plt.figure(tfig.number)
        plt.legend(loc='best')
        plt.xlabel('x [m]')
        plt.ylabel('U [kV]')
        plt.title(f'{Id} Voltage standing waves')
        plt.grid(True)
        
        return Vmax, where, VSWs

