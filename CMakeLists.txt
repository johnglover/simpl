cmake_minimum_required(VERSION 2.6)
project(simpl)



file(GLOB simpl_src src/simpl/*.cpp)
file(GLOB sms_src src/sms/*.c)
file(GLOB sndobj_src src/sndobj/*.cpp)
file(GLOB loris_src src/loris/*.c)
file(GLOB mq_src src/mq/*.cpp)
list(APPEND source_files ${simpl_src}
                         ${sndobj_src}
                         ${sms_src}
                         ${loris_src}
                         ${mq_src})

# In

file(GLOB simpl_include src/simpl/*.h)
set(sms_include src/sms/sms.h)
file(GLOB sndobj_include src/sndobj/*.h)
file(GLOB loris_include src/loris/*.h)
file(GLOB mq_include src/mq/*.h)
list(APPEND include_files ${simpl_include}
                          ${sms_include}
                          ${sndobj_include}
                          ${loris_include}
                          ${mq_include})

# if ninja is used, the following line is needed
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add required dinamic libraries
find_library(SNDFILE sndfile)
find_library(FFTW3 fftw3)
find_library(GSL gsl)

# undefined reference to `fftw_free' error
#
# if you get the above error, you need to add the following line

# Set library directory
# add dinamic libraries


# Add include directories







add_library(simpl SHARED ${source_files} ${include_files})
target_link_libraries(simpl ${libs})


# for debug
message(STATUS "FFTW_LIBRARIES: ${FFTW_LIBRARIES}") 
message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW_INCLUDE_DIRS}")
message(STATUS "SNDFILE_LIBRARIES: ${SNDFILE_LIBRARIES}")
message(STATUS "SNDFILE_INCLUDE_DIRS: ${SNDFILE_INCLUDE_DIRS}")
message(STATUS "SNDOBJ_LIBRARIES: ${SNDOBJ_LIBRARIES}")
message(STATUS "SNDOBJ_INCLUDE_DIRS: ${SNDOBJ_INCLUDE_DIRS}")
message(STATUS "GSL_LIBRARIES: ${GSL_LIBRARIES}")
message(STATUS "GSL_INCLUDE_DIRS: ${GSL_INCLUDE_DIRS}")




# # ----------------------------------------------------------------------------
# # Tests
# # ----------------------------------------------------------------------------
# if(BUILD_TESTS)
#     add_library(simpl_static STATIC ${source_files})
#     list(APPEND libs simpl_static cppunit sndfile)

#     include_directories(tests)

#     set(test_src tests/tests.cpp
#                  tests/test_base.cpp
#                  tests/test_peak_detection.cpp
#                  tests/test_partial_tracking.cpp
#                  tests/test_synthesis.cpp
#                  tests/test_residual.cpp)

#     add_executable(tests ${test_src})
#     target_link_libraries(tests ${libs})
# else()
#     message("Not building tests. To change run CMake with -D BUILD_TESTS=yes")
# endif()
