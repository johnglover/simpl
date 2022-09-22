
// in Windows we have the error: 'ULONG' does not name a type then we need to
// include windows.h before the boost includes


#ifdef _WIN32
#include <windows.h>
typedef unsigned long ULONG, *PULONG;
#endif

// on Windows we have the error: 'ULONG' does not name a type then we need to
// include windows.h before the boost includes


#ifndef _SIMPL_MQ_H
#define _SIMPL_MQ_H

#include <cstdlib>


#include <fftw3.h>


#include <math.h>
#include <string.h>

#include "base.h"

namespace simpl
{


// ---------------------------------------------------------------------------
// MQPeak
// ---------------------------------------------------------------------------
class MQPeak {
    public:
        float amplitude;
        float frequency;
        float phase;
        int bin;
        MQPeak* next;
        MQPeak* prev;

        MQPeak() {
            amplitude = 0.f;
            frequency = 0.f;
            phase = 0.f;
            bin = 0;
            next = NULL;
            prev = NULL;
        }
};


// ---------------------------------------------------------------------------
// MQPeakList
// ---------------------------------------------------------------------------
class MQPeakList {
    public:
        MQPeakList* next;
        MQPeakList* prev;
        MQPeak* peak;

        MQPeakList() {
            next = NULL;
            prev = NULL;
            peak = NULL;
        }
};


// ---------------------------------------------------------------------------
// MQParameters
// ---------------------------------------------------------------------------
class MQParameters {
    public:
        int frame_size;
        int max_peaks;
        int num_bins;
        simpl_sample peak_threshold;
        simpl_sample fundamental;
        simpl_sample matching_interval;
        simpl_sample* window;
        simpl_sample* fft_in;
        fftw_complex* fft_out;
        fftw_plan fft_plan;
        MQPeakList* prev_peaks;

        MQParameters() {
            frame_size = 0;
            max_peaks = 0;
            num_bins = 0;
            peak_threshold = 0.f;
            fundamental = 0.f;
            matching_interval = 0.f;
            window = NULL;
            fft_in = NULL;
            fft_out = NULL;
            prev_peaks = NULL;
        }
};


// ---------------------------------------------------------------------------
// MQ functions
// ---------------------------------------------------------------------------
int init_mq(MQParameters* params);
void reset_mq(MQParameters* params);
int destroy_mq(MQParameters* params);
void mq_add_peak(MQPeak* new_peak, MQPeakList* peak_list);
void delete_peak_list(MQPeakList* peak_list);

MQPeakList* mq_sort_peaks_by_frequency(MQPeakList* peak_list, int num_peaks);
MQPeakList* mq_find_peaks(int signal_size, simpl_sample* signal,
                          MQParameters* params);
MQPeakList* mq_track_peaks(MQPeakList* peak_list, MQParameters* params);

} // end of namespace simpl


#endif
