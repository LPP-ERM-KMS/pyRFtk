Installation
==================================
Pip
---
The program has a pre-build version which can be installed using pip, as this program is not yet made available in the python database,
to install the program you'll have to first clone the repository and navigate to the folder "dist" where the
install file is located, running

.. code-block::

        (.venv) $pip install pyrftk2-2.0.0-py3-none-any.whl

installs the program

Source
------
If you're planning to modify the source code to contribute to the project (or for your own benefit) you'll have to clone the project
to an easily accessible folder and add the path of the folder with the relative adress pyrftk/src/pyRFtk to you PYTHONPATH, e.g by adding
the following to your .zshrc:

.. code-block:: 

        export PYTHONPATH="${PYTHONPATH}:/path/to/pyrftk/src/pyRFtk"

Dependencies
------------
The software is dependent on numpy, scipy and matplotlib
