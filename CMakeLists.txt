cmake_minimum_required(VERSION 2.6)
project(simpl)

set(source_files src/simpl/base.cpp
                 src/simpl/peak_detection.cpp
                 src/simpl/partial_tracking.cpp
                 src/simpl/synthesis.cpp
                 src/simpl/residual.cpp
                 src/sms/OOURA.c
                 src/sms/OOURA.h
                 src/sms/SFMT
                 src/sms/SFMT.c
                 src/sms/SFMT.h
                 src/sms/analysis.c
                 src/sms/cepstrum.c
                 src/sms/fileIO.c
                 src/sms/filters.c
                 src/sms/fixTracks.c
                 src/sms/harmDetection.c
                 src/sms/modify.c
                 src/sms/peakContinuation.c
                 src/sms/peakDetection.c
                 src/sms/residual.c
                 src/sms/sineSynth.c
                 src/sms/sms.c
                 src/sms/sms.h
                 src/sms/soundIO.c
                 src/sms/spectralApprox.c
                 src/sms/spectrum.c
                 src/sms/stocAnalysis.c
                 src/sms/synthesis.c
                 src/sms/tables.c
                 src/sms/transforms.c
                 src/sms/windows.c
)

FILE(GLOB sndobj_src src/sndobj/*.cpp)
FILE(GLOB rfftw_src src/sndobj/rfftw/*.c)
LIST(APPEND source_files ${source_files} ${sndobj_src} ${rfftw_src})

set(include_files src/simpl/simpl.h
                  src/simpl/base.h
                  src/simpl/peak_detection.h
                  src/simpl/partial_tracking.h
                  src/simpl/synthesis.h
                  src/simpl/residual.h
                  src/sms/sms.h
                  src/sndobj/SndObj.h
)

include_directories(src/simpl src/sms src/sndobj src/sndobj/rfftw)
add_library(simpl SHARED ${source_files})
target_link_libraries(simpl m fftw3 gsl gslcblas)

install(TARGETS simpl LIBRARY DESTINATION lib)
install(FILES ${include_files} DESTINATION include/simpl)
