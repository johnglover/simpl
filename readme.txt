Sinusoidal Modelling - A Python Library (SiMPL)
Version 0.2 (first released in December 2010)

Copyright (c) 2009 John Glover, National University of Ireland, Maynooth
http://simplsound.sourceforge.net
john.c.glover@nuim.ie

-----------------------------------------------------------------------------------------
 
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

-----------------------------------------------------------------------------------------

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

- C/C++ compiler
- Python (>= 2.6.*)
- SCons (>= 1.2.0)
- NumPy
- SciPy
- Developers who wish to run the unit tests also need the original open source libraries:
    - sndobj: http://sndobj.sourceforge.net/
    - libsms: http://mtg.upf.edu/static/libsms/


Installation
------------

To compile, in the root directory, run:
> scons

To install, run:
> sudo scons install

For a full list of options:
> scons --help


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
- include new RT Audio code
- tidy up code for HMM/LP partial tracking and Loris integration
- include binaries for Mac OS X and Windows so compilation from source is not needed
- performance issues: MQ, LP and HMM algorithms need to be coded in C/C++ really,
  Python is just too slow, particularly for real-time use. The pure Python implementations
  are useful for testing though.

sndobj:
- fix inaccuracy in the simplsndobj algorithms
- create exception objects
- add a set_synthesis_type property to SndObjSynthesis
- create properties for threshold and num_bins in SndObjPartialTracking class
- make sndobjs use self.sampling_rate
- make peak detection use the new window_size property

sms:
- can sms_scaleDet be moved to the harmonic analysis phase?
- include stochastic residual synthesis in SMSResidual

