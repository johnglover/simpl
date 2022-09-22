Sinusoidal Modelling - A Python Library (SiMPL)
===============================================

Version 0.3 (alpha) : http://simplsound.sourceforge.net  

### Introduction


Simpl is an open source library for sinusoidal modelling written in C/C++ and Python,
and making use of Scientific Python (SciPy). The aim of this
project is to tie together many of the existing sinusoidal modelling implementations
into a single unified system with a consistent API, as well as providing implementations
of some recently published sinusoidal modelling algorithms, many of which have yet
to be released in software. Simpl is primarily intended as a tool for other researchers
in the field, allowing them to easily combine, compare and contrast many of the published
analysis/synthesis algorithms.


### C++ Library Dependencies
------------------------

* [CMake](http://www.cmake.org) 
* [fftw3](http://www.fftw.org) 
* GNU Scientific Library (for libsms)

#### Additional Python Module Dependencies
-------------------------------------

* Python (>= 3.5) || Last builded in Python 3.10
* Cython - http://cython.org
* NumPy
* SciPy

#### Additional Test Dependencies
----------------------------

* [sndobj] (http://sndobj.sourceforge.net)
* [libsms] (http://mtg.upf.edu/static/libsms)


### Build Dynamic Library
------------

To build and install the C++ module, from the simpl root folder run:
* On Windows in Mingw64, see step 1 and 2 of Build Python Module for Windows.

```
    mkdir build
    cd build
    cmake ..
    cmake --build .
```

### Build Python Module
To build and install the Python module, from the simpl root folder run:

#### On Windows 

1. Install Msys2 for Windows using winget: `winget install msys2.msys2`.
2. Install `pacman -S mingw-w64-x86_64-fftw mingw-w64-x86_64-gsl mingw-w64-x86_64-cmake`.
3. Add Mingw64 to the Path enviroment of Windows. It should be `C:\msys64\mingw64\bin`.
4. Build a new enviroment using miniconda and add packages: 
```
winget install Anaconda.Miniconda3
conda create -n simpl python3.10
conda activate simpl
pip install numpy cython scipy
```
5. Inside your enviroment install `C:\Users\<Username>\miniconda3\envs\simpl\Lib\distutils` add the file `distutils.cfg` with this text:
```
[build]
compiler = mingw32

[build_ext]
compiler = mingw32

```
6. Then run `python setup.py build_ext -DMS_WIN64`
7. Inside the builded folder you need to copy this dll: `libfftw3-3 libgcc_s_seh-1 libgsl-27 libstdc++-6 libgslcblas-0 libwinpthread-1`.

#### On Linux

1. Tested on Miniconda Enviroment;
2. Run `sudo apt install -y build-essential cmake libfftw3-dev libgsl-dev`
3. Inside Conda Enviroment: `pip install scipy numpy cython`.
4. Then `pip setup.py build` and `pip setup.py install`.

#### On MacOS
?

### Usage

See the scripts in the examples folder.

### Credits

* The SndObj library is by Dr. Victor Lazzarini (National University of Ireland, Maynooth) and others. 
See the main project page at http://sndobj.sourceforge.net/ for more information.

* Libsms is an implementation of SMS by Rich Eakin, based on code by Dr. Xavier Serra (MTG,
Universitat Pompeu Fabra, Barcelona, Spain)
See the main project page at http://mtg.upf.edu/static/libsms for more information.

* The MQ algorithm is based on the following paper:
R. McAulay, T. Quatieri, "Speech Analysis/Synthesis Based on a Sinusoidal Representation", 
IEEE Transaction on Acoustics, Speech and Signal Processing, vol. 34, no. 4, pp. 744-754, 1986.

Everything else: Copyright (c) 2012 John Glover, National University of Ireland, Maynooth  

john dot c dot glover @ nuim dot net
