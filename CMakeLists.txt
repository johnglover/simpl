cmake_minimum_required(VERSION 2.6)
project(simpl)

FILE(GLOB simpl_src src/simpl/*.cpp)
FILE(GLOB sms_src src/sms/*.c)
FILE(GLOB sndobj_src src/sndobj/*.cpp)
FILE(GLOB loris_src src/loris/*.C)
LIST(APPEND source_files ${simpl_src}
                         ${sndobj_src}
                         ${sms_src}
                         ${loris_src})

FILE(GLOB simpl_include src/simpl/*.h)
set(sms_include src/sms/sms.h)
FILE(GLOB sndobj_include src/sndobj/*.h)
FILE(GLOB loris_include src/loris/*.h)
LIST(APPEND include_files ${simpl_include}
                          ${sms_include}
                          ${sndobj_include}
                          ${loris_include})

add_definitions(-DHAVE_FFTW3_H)
set(libs m fftw3 gsl gslcblas)

include_directories(src/simpl src/sms src/sndobj src/sndobj/rfftw src/loris)

add_library(simpl SHARED ${source_files})
target_link_libraries(simpl ${libs})

install(TARGETS simpl LIBRARY DESTINATION lib)
install(FILES ${include_files} DESTINATION include/simpl)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(BUILD_TESTS)
    set(test_base_src ${source_files} tests/test_base.cpp)
    set(test_peak_detection_src ${source_files} tests/test_peak_detection.cpp)
    set(test_partial_tracking_src ${source_files} tests/test_partial_tracking.cpp)
    set(test_synthesis_src ${source_files} tests/test_synthesis.cpp)

    LIST(APPEND libs cppunit sndfile)

    add_executable(test_base ${test_base_src})
    add_executable(test_peak_detection ${test_peak_detection_src})
    add_executable(test_partial_tracking ${test_partial_tracking_src})
    add_executable(test_synthesis ${test_synthesis_src})

    target_link_libraries(test_base ${libs})
    target_link_libraries(test_peak_detection ${libs})
    target_link_libraries(test_partial_tracking ${libs})
    target_link_libraries(test_synthesis ${libs})
else()
    message("-- Not building tests. To change run CMake with -D BUILD_TESTS=yes")
endif()
