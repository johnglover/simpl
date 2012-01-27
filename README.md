Sinusoidal Modelling - A Python Library (SiMPL)
===============================================

Version 0.3

Copyright (c) 2012 John Glover, National University of Ireland, Maynooth
http://simplsound.sourceforge.net
j@johnglover.net


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


Dependencies
------------

* C/C++ compiler
* Python (>= 2.6.*)
* NumPy
* SciPy
* GNU Scientific Library (for libsms)
* Developers who wish to run the unit tests also need the original open source libraries:
    * sndobj: http://sndobj.sourceforge.net/
    * libsms: http://mtg.upf.edu/static/libsms/


Installation
------------

First build the extension module (so that the SWIG wrapper files are created) by running
the following command in the root folder:

    $ python setup.py build

Then to install the module in your Python site-packages directory:

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


To Do
-----

general:

* include new RT Audio code
* tidy up code for HMM/LP partial tracking and Loris integration
* include binaries for Mac OS X and Windows so compilation from source is not needed
* performance issues: MQ, LP and HMM algorithms need to be coded in C/C++ really,
  Python is just too slow, particularly for real-time use. The pure Python implementations
  are useful for testing though.

sndobj:

* create exception objects
* add a set_synthesis_type property to SndObjSynthesis
* create properties for threshold and num_bins in SndObjPartialTracking class
* make sndobjs use self.sampling_rate
* make peak detection use the new window_size property

sms:

* move sms_scaleDet to the harmonic analysis phase
