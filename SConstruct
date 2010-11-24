# Copyright (c) 2009 John Glover, National University of Ireland, Maynooth
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import os, sys
import distutils.sysconfig

# location of msys (windows only)
# by default, it installs to C:/msys/1.0
msys_path = "C:/msys/1.0"

def get_platform():
    if sys.platform[:5] == "linux":
        return "linux"
    elif sys.platform[:3] == "win":
        return "win32"
    elif sys.platform[:6] == "darwin":
        return "darwin"
    else:
        return "unsupported"

def get_version():
    return sys.version[:3]

# check that the current platform is supported
if get_platform() == "unsupported":
    print "Error: Cannot build on this platform. "
    print "       Only Linux, Mac OS X and Windows are currently supported."
    exit(1)

# environment
env = Environment(ENV=os.environ)

# set default installation directories
default_install_dir = ""
if get_platform() == "win32":
    default_install_dir = "C:/msys/1.0/local"
    man_prefix = "C:/msys/1.0/local/man/man1"
else:
    default_install_dir = "/usr/local"
    man_prefix = "/usr/share/man/man1"

# command-line options
vars = Variables(["variables.cache"])
vars.AddVariables(
    ("prefix", "Installation directory", default_install_dir),
    ("libpath", "Additional directory to search for libraries", ""),
    ("cpath", "Additional directory to search for C header files", ""),
    BoolVariable('debug', 'Compile extension modules with debug symbols', False),
    BoolVariable('sndobj', 'Build and install the SndObj module', True),
    BoolVariable('sms', 'Build and install the SMS module', True),
    BoolVariable('loris', 'Build and install the loris module', True),
    BoolVariable('mq', 'Build and install the McAulay-Quatieri module', True),
    BoolVariable('hmm', 'Build and install the HMM partial tracking module', True),
    BoolVariable('lp', 'Build and install the LP partial tracking module', True)
)
vars.Update(env)
vars.Save("variables.cache", env)
Help(vars.GenerateHelpText(env))

# set library and header directories
if get_platform() == "linux":
    env.Append(LIBPATH=["/usr/local/lib", "/usr/lib"])
    env.Append(CPPPATH=["/usr/local/include", "/usr/include"])
elif get_platform() == "darwin":
    env.Append(LIBPATH=["/opt/local/lib", "/usr/local/lib", "/usr/lib"])
    env.Append(CPPPATH=["/opt/local/include", "/usr/local/include", "/usr/include"])
elif get_platform() == "win32":
    env.Append(LIBPATH=["/usr/local/lib", "/usr/lib", "C:/msys/1.0/local/lib", 
                        "C:/msys/1.0/lib", "C:/Python26/libs"])    
    env.Append(CPPPATH=["/usr/local/include", "/usr/include", "C:/msys/1.0/local/include",
                        "C:/msys/1.0/include", "C:/Python26/include"])

# add paths specified at the command line    
env.Append(LIBPATH = env["libpath"])
env.Append(CPPPATH = env["cpath"])

conf = Configure(env)

# set python library and include directories
python_lib_path = []
python_inc_path = []
# linux
if get_platform() == "linux":
    python_inc_path = ["/usr/include/python" + get_version()]
# os x
elif get_platform() == "darwin":
    python_inc_path = ["/Library/Frameworks/Python.framework/Headers", 
                       "/System/Library/Frameworks/Python.framework/Headers"]
# windows
elif get_platform() == "win32":
    python_lib = "python%c%c"% (get_version()[0], get_version()[2])
    python_inc_path = ["c:\\Python%c%c\include" % (get_version()[0], get_version()[2])]
    python_lib_path.append("c:\\Python%c%c\libs" % (get_version()[0], get_version()[2]))

# check for python
if not conf.CheckHeader("Python.h", language="C"):
    for i in python_inc_path:
        pythonh = conf.CheckHeader("%s/Python.h" % i, language="C")
        if pythonh:
            break
if not pythonh:
    print "Python headers are missing. Cannot build simpl."
    exit(1)
    
# check for swig
if not "swig" in env["TOOLS"]:
    print "Error: Swig was not found."
    exit(1)
    
# check for numpy
try:
    import numpy
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
except ImportError:
    print "Numpy was not found. Cannot build simpl.\n"
    exit(1)
env.Append(CPPPATH = numpy_include)

# check if we need debug symbols
if env['debug']:
    env.Append(CCFLAGS = "-g")
    
env = conf.Finish()

# get python installation directory
python_install_dir = os.path.join(distutils.sysconfig.get_python_lib(), "simpl")
env.Alias('install', python_install_dir)

# sndobj module
if env["sndobj"]:
    sndobj_env = env.Clone()
    sndobj_env.Append(SWIGFLAGS = ["-python", "-c++"])
    for lib_path in python_lib_path:
        sndobj_env.Append(LIBPATH = lib_path) 
    for inc_path in python_inc_path:
        sndobj_env.Append(CPPPATH = inc_path)
    sndobj_env.Append(CPPPATH = "sndobj")
    sndobj_env.Append(CPPPATH = "sndobj/rfftw")
    
    # get sources
    sndobj_sources = Glob("sndobj/*.cpp", strings=True)
    sndobj_sources.append(Glob("sndobj/rfftw/*.c", strings=True))
    # remove wrapper file from source list if it exists, otherwise it will be added twice
    if "sndobj/sndobj_wrap.cpp" in sndobj_sources:
        sndobj_sources.remove("sndobj/sndobj_wrap.cpp")

    # create the python wrapper using SWIG
    python_wrapper = sndobj_env.SharedObject("sndobj/sndobj.i")
    sndobj_sources.append(python_wrapper)
    
    # copy the generated .py file to the root directory
    Command("simplsndobj.py", "sndobj/simplsndobj.py", Copy("$TARGET", "$SOURCE"))

    # build the module
    if get_platform() == "win32":
        sndobj_env.Append(LIBS = [python_lib])
        sndobj_env.SharedLibrary("simplsndobj", sndobj_sources, SHLIBPREFIX="_", SHLIBSUFFIX=".pyd")
    elif get_platform() == "darwin":
        sndobj_env.Append(LIBS = ["python" + get_version()])
        sndobj_env.Prepend(LINKFLAGS=["-framework", "python"])
        sndobj_env.LoadableModule("_simplsndobj.so", sndobj_sources)
    else: # linux
        sndobj_env.Append(LIBS = ["python" + get_version()])
        sndobj_env.SharedLibrary("simplsndobj", sndobj_sources, SHLIBPREFIX="_")       

# sms module
if env["sms"]:
    sms_env = env.Clone()
    
    # look for additional libraries
    sms_conf = Configure(sms_env)
    
    # check for libmath
    if not sms_conf.CheckLibWithHeader('m','math.h','c'):
        print "libmath could not be found. Cannot build the SMS module."
        exit(1)
    
    # if using windows, assume default gsl paths
    # this is because env.ParseConfig calls gsl-config using the 
    # windows shell rather than the msys shell, and gsl-config
    # is a shell script so it will not run using the windows shell
    # TODO: is there a way to get env.ParseConfig to call the msys
    # shell instead? Might be useful, although would introduce 
    # another dependency, msys.
    if get_platform() == 'win32':
        # check for libgsl
        if not sms_conf.CheckLibWithHeader('gsl', 'gsl_sys.h', 'c'):
            print "libgsl (GNU Scientific Library) could not be found. Cannot build the SMS module."
            exit(1)
        if not sms_conf.CheckLibWithHeader('gslcblas', 'gsl_cblas.h', 'c'):
            print "libgsl (GNU Scientific Library) could not be found. Cannot build the SMS module."
            exit(1)
    # if not using windows, call gsl-config
    else:
        sms_env.ParseConfig("gsl-config --cflags --libs")
        
    sms_env = sms_conf.Finish()
    
    sms_env.Append(SWIGFLAGS = ["-python"])
    for lib_path in python_lib_path:
        sms_env.Append(LIBPATH = lib_path) 
    for inc_path in python_inc_path:
        sms_env.Append(CPPPATH = inc_path)
    sms_env.Append(CPPPATH = "sms")
    sms_env.Append(CPPPATH = "sms/SFMT")
    if not env['debug']:
        sms_env.Append(CCFLAGS = "-O2 -funroll-loops -fomit-frame-pointer -Wall -W")
        sms_env.Append(CCFLAGS = "-Wno-unused -Wno-parentheses -Wno-switch -fno-strict-aliasing")
    sms_env.Append(CCFLAGS = "-DMERSENNE_TWISTER")
    
    # get sources
    sms_sources = Glob("sms/*.c", strings=True)
    # remove wrapper file from source list if it exists, otherwise it may be added twice
    if "sms/sms_wrap.c" in sms_sources:
        sms_sources.remove("sms/sms_wrap.c")
    
    # create the python wrapper using SWIG
    python_wrapper = sms_env.SharedObject("sms/sms.i")
    sms_sources.append(python_wrapper)
    
    # copy the generated .py file to the simpl directory
    Command("simplsms.py", "sms/simplsms.py", Copy("$TARGET", "$SOURCE")) 
      
    # build the module
    if get_platform() == "win32":
        sms_env.Append(LIBS = [python_lib])
        sms_env.SharedLibrary("simplsms", sms_sources, SHLIBPREFIX="_", SHLIBSUFFIX=".pyd")
    elif get_platform() == "darwin":
        sms_env.Append(LIBS = ["python" + get_version()])
        sms_env.Prepend(LINKFLAGS=["-framework", "python"])
        sms_env.LoadableModule("_simplsms.so", sms_sources)         
    else: # linux
        sms_env.Append(LIBS = ["python" + get_version()])
        sms_env.SharedLibrary("simplsms", sms_sources, SHLIBPREFIX="_") 
        
# install the python modules
python_modules = Glob("*.py", strings=True)
if get_platform() == "win32":
    modules = Glob("*.pyd", strings=True)
else:
    modules = Glob("*.so", strings=True)
modules.extend(python_modules)

for module in modules:
    env.InstallAs(os.path.join(python_install_dir, module), module)

