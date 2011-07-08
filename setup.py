"""
Simpl is an open source library for sinusoidal modelling written in C/C++ and Python,
and making use of Scientific Python (SciPy). The aim of this
project is to tie together many of the existing sinusoidal modelling implementations
into a single unified system with a consistent API, as well as providing implementations
of some recently published sinusoidal modelling algorithms, many of which have yet
to be released in software. Simpl is primarily intended as a tool for other researchers
in the field, allowing them to easily combine, compare and contrast many of the published
analysis/synthesis algorithms.
"""
from distutils.core import setup, Extension
import os
from glob import glob

# ------------------------------------------------------------------------------
# Global
# ------------------------------------------------------------------------------

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
    print "Error: Numpy was not found."
    exit(1)

macros = []
link_args = [] 
swig_opts = ['-c++']
include_dirs = [numpy_include, '/usr/local/include'] 

# ------------------------------------------------------------------------------
# SndObj Library
# ------------------------------------------------------------------------------

sndobj_sources = """
	SndObj.cpp SndIO.cpp FFT.cpp IFFT.cpp PVA.cpp PVS.cpp IFGram.cpp 
    SinAnal.cpp SinSyn.cpp AdSyn.cpp ReSyn.cpp HarmTable.cpp HammingTable.cpp
    """.split()

fftw_sources = """
    config.c  fcr_9.c fhf_6.c fn_8.c  frc_1.c  ftw_16.c ftwi_7.c
    executor.c fftwnd.c fhf_7.c fn_9.c  frc_10.c ftw_2.c  ftwi_8.c 
    fcr_1.c fhb_10.c fhf_8.c fni_1.c frc_11.c ftw_3.c  ftwi_9.c 
    fcr_10.c  fhb_16.c fhf_9.c fni_10.c frc_12.c ftw_32.c generic.c 
    fcr_11.c  fhb_2.c fn_1.c fni_11.c frc_128.c ftw_4.c  malloc.c 
    fcr_12.c  fhb_3.c fn_10.c fni_12.c frc_13.c ftw_5.c  planner.c 
    fcr_128.c fhb_32.c fn_11.c fni_13.c frc_14.c ftw_6.c  putils.c 
    fcr_13.c  fhb_4.c fn_12.c fni_14.c frc_15.c ftw_64.c rader.c 
    fcr_14.c  fhb_5.c fn_13.c fni_15.c frc_16.c ftw_7.c  rconfig.c 
    fcr_15.c  fhb_6.c fn_14.c fni_16.c frc_2.c  ftw_8.c  rexec.c 
    fcr_16.c  fhb_7.c fn_15.c fni_2.c frc_3.c  ftw_9.c  rexec2.c 
    fcr_2.c fhb_8.c fn_16.c fni_3.c frc_32.c ftwi_10.c rfftwf77.c 
    fcr_3.c fhb_9.c fn_2.c fni_32.c frc_4.c  ftwi_16.c rfftwnd.c 
    fcr_32.c  fhf_10.c fn_3.c fni_4.c frc_5.c  ftwi_2.c rgeneric.c 
    fcr_4.c fhf_16.c fn_32.c fni_5.c frc_6.c  ftwi_3.c rplanner.c 
    fcr_5.c fhf_2.c fn_4.c fni_6.c frc_64.c ftwi_32.c timer.c 
    fcr_6.c fhf_3.c fn_5.c fni_64.c frc_7.c  ftwi_4.c twiddle.c 
    fcr_64.c  fhf_32.c fn_6.c fni_7.c frc_8.c  ftwi_5.c wisdom.c 
    fcr_7.c fhf_4.c fn_64.c fni_8.c frc_9.c  ftwi_6.c wisdomio.c 
    fcr_8.c fhf_5.c fn_7.c fni_9.c ftw_10.c ftwi_64.c cfft.c
    """.split()

sndobj_sources = map(lambda x: 'src/sndobj/' + x, sndobj_sources) 
sndobj_sources.extend(map(lambda x: 'src/sndobj/rfftw/' + x, fftw_sources))
sndobj_sources.append("simpl/sndobj.i")

sndobj_macros = [('PYTHON_WRAP', None)]
sndobj_macros.extend(macros)
sndobj_swig_opts = ['-c++', '-DNUMPY']
sndobj_swig_opts.extend(swig_opts)
sndobj_include_dirs = ['src/sndobj/rfftw', 'src/sndobj']
sndobj_include_dirs.extend(include_dirs)

sndobj = Extension("simpl/_simplsndobj", 
                   sources=sndobj_sources,
                   include_dirs=sndobj_include_dirs,
                   define_macros=sndobj_macros,
                   swig_opts=sndobj_swig_opts)

# ------------------------------------------------------------------------------
# SMS
# ------------------------------------------------------------------------------

sms_sources = """
    OOURA.c cepstrum.c peakContinuation.c soundIO.c tables.c
    fileIO.c peakDetection.c spectralApprox.c transforms.c
    filters.c residual.c spectrum.c windows.c SFMT.c fixTracks.c
    sineSynth.c stocAnalysis.c harmDetection.c sms.c synthesis.c
    analysis.c modify.c
    """.split()

sms_sources = map(lambda x: 'src/sms/' + x, sms_sources) 
sms_sources.append("simpl/sms.i")
sms_include_dirs = ['src/sms']
sms_include_dirs.extend(include_dirs)

sms = Extension("simpl/_simplsms", 
                sources=sms_sources,
                include_dirs=sms_include_dirs,
                libraries=['m', 'fftw3', 'gsl', 'gslcblas'],
                extra_compile_args=['-DMERSENNE_TWISTER'])

# ------------------------------------------------------------------------------
# Loris
# ------------------------------------------------------------------------------

loris_sources = glob('src/loris/*.C')
loris_sources.extend(glob('src/loris/*.c'))
loris_sources.append('simpl/loris.i')
loris_include_dirs = ['src/loris'] 
loris_include_dirs.extend(include_dirs)

loris = Extension("simpl/_simplloris", 
                  sources=loris_sources,
                  include_dirs=loris_include_dirs,
                  swig_opts=swig_opts)

# ------------------------------------------------------------------------------
# Package
# ------------------------------------------------------------------------------

doc_lines = __doc__.split("\n")

setup(name='simpl',
      description=doc_lines[0],
      long_description="\n".join(doc_lines[2:]),
      url='http://simplsound.sourceforge.net',
      download_url='http://simplsound.sourceforge.net',
      license='GPL',
      author='John Glover',
      author_email='j@johnglover.net',
      platforms=["Linux", "Mac OS-X", "Unix", "Windows"],
      version='0.3',
      ext_modules=[sndobj, sms, loris],
      packages=['simpl', 'simpl.plot'])
