---
title: 'pyRFtk'
tags:
  - Python
  - RF
  - Electricity
  - Nuclear Fusion
  - ICRH
authors:
  - name: Arthur Adriaens
    orcid: 0009-0002-0200-8375
    equal-contrib: true
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Frederic Durodie
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: Royal Military Academy
   index: 1
 - name: Ghent University
   index: 2
date: 9 Oktober 2023
bibliography: paper.bib

---

# Summary

In the field of nuclear fusion, the most prominent sources of both heating and
wall conditioning are radio frequency heating. More particularly Ion cyclotron
resonance heating and electron cyclotron heating.  As such the understanding of
the circuitry involved and how to maximize the antenna-plasma coupling needs to be
thoroughly understood.

# Statement of need

'pyRFtk' is a python package for simulating RF circuitry. PyRFtk is built on the principle
of connecting various blocks such as transmission lines and touchstone files into a circuit, 
making it easy to design most circuits imaginable and examine them.

The program was designed to be used by experienced RF engineers needing to analyse an RF circuit
but also made straightforward enough to be usable by people new to the field.
Even though the software is new to the open-source landscape, it is under active development
since 19?? and has appeared in numerous papers [@??]

# Mathematics

The program solves the telegrapher's equations, i.e assuming that the conductors are composed of an infinite series of 
two-port elementary components. 

The telegrapher's equations in the time domain are :

$$\frac{\partial}{\partial x} V(x,t) = -L \frac{\partial}{\partial t} I(x,t) - R I(x,t)$$
$$\frac{\partial}{\partial x} I(x,t) = -C \frac{\partial}{\partial t} V(x,t) - G V(x,t)$$

Which can be combined

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
