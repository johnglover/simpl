Sinusoidal Modelling - A Python Library (SiMPL)
===============================================

Version 0.3 (alpha)

http://simplsound.sourceforge.net  


Introduction
------------

Simpl is an open source library for sinusoidal modelling written in C/C++ and Python,
and making use of Scientific Python (SciPy). The aim of this
project is to tie together many of the existing sinusoidal modelling implementations
into a single unified system with a consistent API, as well as providing implementations
of some recently published sinusoidal modelling algorithms, many of which have yet
to be released in software. Simpl is primarily intended as a tool for other researchers
in the field, allowing them to easily combine, compare and contrast many of the published
analysis/synthesis algorithms.


C++ Library Dependencies
------------------------

* CMake_
* fftw3_
* GNU Scientific Library (for libsms)

.. _CMake: http://www.cmake.org
.. _fftw3: http://www.fftw.org


Additional Python Module Dependencies
-------------------------------------

* Python (>= 3.5)
* Cython_
* NumPy
* SciPy

.. _Cython: http://cython.org


Additional Test Dependencies
----------------------------

* sndobj_
* libsms_

.. _sndobj: http://sndobj.sourceforge.net
.. _libsms: http://mtg.upf.edu/static/libsms


Installation
------------

To build and install the C++ module, from the simpl root folder run:

::

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ sudo make install

To build and install the Python module, from the simpl root folder run:

::

    $ python setup.py build
    $ python setup.py install


Usage
-----

See the scripts in the examples folder.


Credits
-------

The SndObj library is by Dr. Victor Lazzarini (National University of Ireland, Maynooth) and others. 
See the main project page at http://sndobj.sourceforge.net/ for more information.

Libsms is an implementation of SMS by Rich Eakin, based on code by Dr. Xavier Serra (MTG,
Universitat Pompeu Fabra, Barcelona, Spain)
See the main project page at http://mtg.upf.edu/static/libsms for more information.

The MQ algorithm is based on the following paper:
R. McAulay, T. Quatieri, "Speech Analysis/Synthesis Based on a Sinusoidal Representation", 
IEEE Transaction on Acoustics, Speech and Signal Processing, vol. 34, no. 4, pp. 744-754, 1986.

Everything else: Copyright (c) 2012 John Glover, National University of Ireland, Maynooth  

john dot c dot glover @ nuim dot net
