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
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Frederic Durodie
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: Royal Military Academy
   index: 1
 - name: Ghent University
   index: 2
date: 09/10/2023
bibliography: paper.bib

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

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

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

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
