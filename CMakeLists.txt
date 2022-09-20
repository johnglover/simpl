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

# change all /c to C:  (windows only)
if(WIN32)
  foreach(file ${source_files})
    string(REGEX REPLACE "/c" "C:" file ${file})
  endforeach()
  foreach(file ${include_files})
    string(REGEX REPLACE "/c" "C:" file ${file})
  endforeach()
endif()


                
# include mingw64 FFTW3
set(FFTW3_INCLUDE_DIR "C:/msys64/mingw64/include")
set(FFTW3_LIBRARY "C:/msys64/mingw64/lib/libfftw3-3.dll.a")


add_library(simpl SHARED ${source_files} ${include_files})
target_link_libraries(simpl ${libs})

install(TARGETS simpl LIBRARY DESTINATION lib)
install(FILES ${include_files} DESTINATION include/simpl)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(BUILD_TESTS)
    add_library(simpl_static STATIC ${source_files})
    list(APPEND libs simpl_static cppunit sndfile)

    include_directories(tests)

    set(test_src tests/tests.cpp
                 tests/test_base.cpp
                 tests/test_peak_detection.cpp
                 tests/test_partial_tracking.cpp
                 tests/test_synthesis.cpp
                 tests/test_residual.cpp)

    add_executable(tests ${test_src})
    target_link_libraries(tests ${libs})
else()
    message("Not building tests. To change run CMake with -D BUILD_TESTS=yes")
endif()
