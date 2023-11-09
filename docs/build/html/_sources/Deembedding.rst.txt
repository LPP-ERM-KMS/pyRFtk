Deembedding
===========
**De-embedding**
    "De-embedding is a post-measurement process to minimize errors and reveal information about the device under test. When there is a composite measurement of a device under test or fixture combination, you can use de-embedding to isolate the performance of the fixture and extract (de-embed) the fixture from the measurements."

    "De-embedding is mathematically removing the measurements affected by the fixture leaving only the behavior of the device under test. This is commonly used when there are non-coaxial connection from the VNA to the DUT, and it is used on circuit board traces, backplane channels, semiconductor packages, connectors or other discrete components."

    -Andreas Henkel, Product Management for Network Analysis - Rohde & Schwarz

What this means in the context of our software is the following: 

Say we have a circuit structure as follows:

.. code-block:: 
   
    +-------------------------------------------------+
    |                 Sexternal                       |
    |                                                 |
    |    +-------------+                              |
    |    |  Sinternal  |                              |
    |    |            ( )----------------------------( )
    |    |             |               :              |
    |    |             |               :              |
    |    |             |               :              |
    |    |            ( )----------------------------( )
    |    |             |                              |
    |    |             |         +---------+          |
    |    |             |         | Deembed |          |
    |    |             |         |         |          |
    |    |            ( )-------( )       ( )--------( )
    |    |             |    :    |         |     :    |
    |    |       ip    |    :    | dpi dpe |     :    | ep
    |    |             |    :    |         |     :    |
    |    |            ( )-------( )       ( )--------( )
    |    |             |         |         |          |
    |    +-------------+         +---------+          |
    |                                                 |
    +-------------------------------------------------+

Here we know the scatter matrices Sexternal (SE) and Deembed (SD), and we want to find the scatter matrix of the internal
component: Sinternal (SI).

IntPorts = {dpi:ip, ...}
ExtPorts = {dpe:ep, ...}

We may use the 'deembed' function in the rfCircuit object to solve for

.. code-block::

    +-------------+ 
    |  Sinternal  | 
    |            ( )
    |             | 
    |             | 
    |             | 
    |            ( )
    |             | 
    |             | 
    |             | 
    |             | 
    |            ( )
    |             | 
    |       ip    | 
    |             | 
    |            ( )
    |             | 
    +-------------+ 


As an example we'll use the TOMAS ICRH system again, 
the strap of the antenna has been measured together
with two pieces of TL, we can deembed it as follows:

.. code-block:: 

    A2Ca1_L = 0.1715
    A2Ca1_Z = 50
    T2Cs1_L = 0.121
    T2Cs1_Z = 50

    tsStrap = 'Antenna/Tomas-Ref_geo-R=200-Diel_eps=0500.s2p'


    strap = rfObject(touchstone=tsStrap, ports=['cap','t'])

    if strap.fs[0] == 0.:
        strap.fs[0] = 1e-3 # avoid divisions by 0 at 0 Hz in deembed

    strap.deembed({'cap':(A2Ca1_L, A2Ca1_Z), 
                        't'  :(T2Cs1_L, T2Cs1_Z)})
