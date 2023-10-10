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
