
S-matrix and combining circuits
-------------------------------
The program is capable of combining circuits and calculating the S matrix for
given frequencies, an example is given below:

.. code-block:: python

        from pyRFtk import rfCircuit, rfTRL
        from pyRFtk.printMatrices import printMA 

        A = rfCircuit()
        A.addblock('TL1', rfTRL(L=0))
        A.addblock('TL2', rfTRL(L=0))
        A.connect('TL1.1','TL2.1','TA')

        B = rfCircuit()
        B.addblock('TL1', rfTRL(L=0))
        B.addblock('TL2', rfTRL(L=0))
        B.connect('TL1.1','TL2.1','TB')

        C = rfCircuit()
        C.addblock('A', A)
        C.addblock('B', B)
        C.addblock('TL3',rfTRL(L=0))
        C.connect('A.TA','B.TB','TL3.1')

        printMA(C.getS(1E6))

Here we create a circuit C consisting of a transmission line of length 0, which has as an input
(if no port names are given, 1 is the left input and 2 is the right output) of A and B, each of
which are two 0 lenght coax cables of which the inputs are wired together.

We calculate the S matrix at 1MHz and print it using printMA, this funciton makes it posible
to print complex matrices in easy to read format, e.g it will give an output *0.6 +180°* impying

.. math::

   \rho e^{i\theta} = 0.6 e^{i\pi} = -0.6

Multiple ports and Logging
--------------------------

.. code-block:: python
        
        from pyRFtk import rfCircuit, rfTRL, plotVSWs
        from pyRFtk.printMatrices import printMA 
        from pyRFtk.config import setLogLevel
        import matplotlib.pyplot as plt

        TRL1 = rfTRL(L=1.1, ports=['1a','1b'])
        TRL2 = rfTRL(L=2.1, ports=['2a','2b'])

        CT1 = rfCircuit()
        CT1.addblock('TRL1', TRL1, relpos= 0. )
        CT1.addblock('TRL2', TRL2, relpos= 1.1 )
        CT1.connect('TRL1.1b','TRL2.2a')

        TRL3 = rfTRL(L=1.3, ports=['3a','3b'])
        TRL4 = rfTRL(L=1.4, ports=['4a','4b'])
        CT2 = rfCircuit()
        CT2.addblock('TRL3', TRL3, relpos= 0. )
        CT2.addblock('TRL4', TRL4, relpos= 1.3)
        CT2.connect('TRL3.3b','TRL4.4a')
        CT2.terminate('TRL4.4b', RC=0.5j)

        CT3 = rfCircuit()
        CT3.addblock('CT1', CT1, relpos= 0. )
        CT3.addblock('CT2', CT2, relpos= 0. )
        CT3.connect('CT1.TRL1.1a','CT2.TRL3.3a','ct1')

        CT4 = rfCircuit(Id='Duh')
        CT4.addblock('TRL5', rfTRL(L=2.5, ports=['5a','5b']), relpos= 0. )
        CT4.addblock('CT3', CT3, relpos= 2.5 )
        CT4.connect('TRL5.5b','CT3.ct1')

        setLogLevel('DEBUG')
        maxV, where, VSWs = CT4.maxV(f=45e6, E={'TRL5.5a':1, 'CT3.CT1.TRL2.2b':0.5})
        setLogLevel('CRITICAL')
        plotVSWs(VSWs)

        plt.show()

Here we excite two ports: TRL5.5a with a 1V wave and CT3.CT1.TRL2.2b with a 0.5V wave. As can be seen in the code
