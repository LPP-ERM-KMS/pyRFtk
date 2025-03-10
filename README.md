# pyRFtk
[![DOI](https://zenodo.org/badge/702980029.svg)](https://zenodo.org/doi/10.5281/zenodo.10391750)
![logo](https://github.com/LPP-ERM-KMS/pyRFtk/blob/master/logo/logo.svg)
## About

pyRFtk is a Python library able to build and analyse RF circuits, originally designed for ICRH antennae but now widely deployable.
you can:
- Incorporate touchstone files to create your own elements
- See the currents throughout your circuit
- Compute S-matrices
- Detect Arcs
- Deembed systems
- ...
  
## Installation
pyRFtk is available as a pip package:

```sh
pip install pyRFtk
```

Or using your favorite distribution platform

## Authors
The code was developed by RF-engineer Frederic Durodie and later
on refactored, packaged and documented by Arthur Adriaens

## Example usage
This code was extensively used for the paper [An automatic matching system for the ICRF antenna at TOMAS: Development and experimental proof](https://doi.org/10.1016/j.fusengdes.2025.114840), the code for which may be found in the examples folder.
