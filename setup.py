'''
Simpl is an open source library for sinusoidal modelling written in C/C++ and
Python, and making use of Scientific Python (SciPy). The aim of this project
is to tie together many of the existing sinusoidal modelling implementations
into a single unified system with a consistent API, as well as providing
implementations of some recently published sinusoidal modelling algorithms,
many of which have yet to be released in software. Simpl is primarily intended
as a tool for other researchers in the field, allowing them to easily combine,
compare and contrast many of the published analysis/synthesis algorithms.
'''
import os
import glob
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# -----------------------------------------------------------------------------
# Global
# -----------------------------------------------------------------------------

# detect platform
platform = os.uname()[0] if hasattr(os, 'uname') else 'Windows'

# get numpy include directory
try:
    import numpy
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
except ImportError:
    print('Error: Numpy was not found.')
    exit(1)

macros = []
link_args = []
include_dirs = ['simpl', 'src/simpl', 'src/sms', 'src/sndobj',
                'src/loris', 'src/mq', numpy_include, '/usr/local/include', '.']

# add FFTW3 include directory
if platform == 'Linux':
    include_dirs.append('/usr/include/fftw3')
elif platform == 'Darwin':
    include_dirs.append('/opt/local/include')
elif platform == 'Windows':
    user_input = input('Please enter the path to the FFTW3 include directory: ')
    # change backslashes to forward slashes
    user_input = user_input.replace('\\', '/')
    # remove trailing slash 
    if user_input[-1] == '/':
        user_input = user_input[:-1]
    include_dirs.append(user_input)
    include_dirs.append('include')


libs = ['m', 'fftw3', 'gsl', 'gslcblas']
compile_args = ['-DMERSENNE_TWISTER', '-DHAVE_FFTW3_H']
sources = []

# -----------------------------------------------------------------------------
# SndObj Library
# -----------------------------------------------------------------------------
sndobj_sources = '''
    SndObj.cpp SndIO.cpp FFT.cpp PVA.cpp IFGram.cpp SinAnal.cpp
    SinSyn.cpp AdSyn.cpp ReSyn.cpp HarmTable.cpp HammingTable.cpp
    '''.split()
sndobj_sources = map(lambda x: 'src/sndobj/' + x, sndobj_sources)
sources.extend(sndobj_sources)

# -----------------------------------------------------------------------------
# SMS
# -----------------------------------------------------------------------------
sms_sources = '''
    OOURA.c cepstrum.c peakContinuation.c soundIO.c tables.c
    fileIO.c peakDetection.c spectralApprox.c transforms.c
    filters.c residual.c spectrum.c windows.c SFMT.c fixTracks.c
    sineSynth.c stocAnalysis.c harmDetection.c sms.c synthesis.c
    analysis.c modify.c
    '''.split()
sms_sources = map(lambda x: 'src/sms/' + x, sms_sources)
sources.extend(sms_sources)

# -----------------------------------------------------------------------------
# Loris
# -----------------------------------------------------------------------------
compile_loris = False

loris_sources = glob.glob(os.path.join('src', 'loris', '*.c'))
sources.extend(loris_sources)

# -----------------------------------------------------------------------------
# MQ
# -----------------------------------------------------------------------------
mq_sources = glob.glob(os.path.join('src', 'mq', '*.cpp'))
sources.extend(mq_sources)

# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------
base = Extension(
    'simpl.base',
    sources=['simpl/base.pyx',
             'src/simpl/base.cpp',
             'src/simpl/exceptions.cpp'],
    include_dirs=include_dirs,
    language='c++'
)

# -----------------------------------------------------------------------------
# Peak Detection
# -----------------------------------------------------------------------------
peak_detection = Extension(
    'simpl.peak_detection',
    sources=sources + [
                    #'simpl/peak_detection.pyx',
                    #    'src/simpl/peak_detection.cpp',
                       'src/simpl/base.cpp',
                       'src/simpl/exceptions.cpp'],
    include_dirs=include_dirs,
    libraries=libs,
    extra_compile_args=compile_args,
    language='c++'
)

# -----------------------------------------------------------------------------
# Partial Tracking
# -----------------------------------------------------------------------------
partial_tracking = Extension(
    'simpl.partial_tracking',
    sources=sources + ['simpl/partial_tracking.pyx',
                       'src/simpl/partial_tracking.cpp',
                       'src/simpl/base.cpp',
                       'src/simpl/exceptions.cpp'],
    libraries=libs,
    extra_compile_args=compile_args,
    include_dirs=include_dirs,
    language='c++'
)

# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------
synthesis = Extension(
    'simpl.synthesis',
    sources=sources + ['simpl/synthesis.pyx',
                       'src/simpl/synthesis.cpp',
                       'src/simpl/base.cpp',
                       'src/simpl/exceptions.cpp'],
    libraries=libs,
    extra_compile_args=compile_args,
    include_dirs=include_dirs,
    language='c++'
)

# -----------------------------------------------------------------------------
# Residual
# -----------------------------------------------------------------------------
residual = Extension(
    'simpl.residual',
    sources=sources + ['simpl/residual.pyx',
                       'src/simpl/peak_detection.cpp',
                       'src/simpl/partial_tracking.cpp',
                       'src/simpl/synthesis.cpp',
                       'src/simpl/residual.cpp',
                       'src/simpl/base.cpp',
                       'src/simpl/exceptions.cpp'],
    libraries=libs,
    extra_compile_args=compile_args,
    include_dirs=include_dirs,
    language='c++'
)

# -----------------------------------------------------------------------------
# Package
# -----------------------------------------------------------------------------
doc_lines = __doc__.split('\n')

setup(
    name='simpl',
    description=doc_lines[0],
    long_description='\n'.join(doc_lines[2:]),
    url='http://simplsound.sourceforge.net',
    download_url='http://simplsound.sourceforge.net',
    license='GPL',
    author='John Glover',
    author_email='j@johnglover.net',
    platforms=['Linux', 'Mac OS-X', 'Unix'],
    version='0.3',
    ext_modules=[base, peak_detection, partial_tracking, synthesis, residual],
    cmdclass={'build_ext': build_ext},
    packages=['simpl', 'simpl.plot']
)
