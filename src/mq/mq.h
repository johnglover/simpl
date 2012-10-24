#ifndef _SIMPL_MQ_H
#define _SIMPL_MQ_H

#include <cstdlib>

#include <fftw3.h>
#include <math.h>
#include <string.h>

namespace simpl
{


typedef double sample;

typedef struct MQPeak {
    float amplitude;
    float frequency;
    float phase;
    int bin;
    struct MQPeak* next;
    struct MQPeak* prev;
} MQPeak;

typedef struct MQPeakList {
    struct MQPeakList* next;
    struct MQPeakList* prev;
    struct MQPeak* peak;
} MQPeakList;

typedef struct MQParameters {
    int frame_size;
    int max_peaks;
    int num_bins;
    sample peak_threshold;
    sample fundamental;
    sample matching_interval;
    sample* window;
    sample* fft_in;
	fftw_complex* fft_out;
	fftw_plan fft_plan;
    MQPeakList* prev_peaks;
} MQParameters;

int init_mq(MQParameters* params);
void reset_mq(MQParameters* params);
int destroy_mq(MQParameters* params);
void delete_peak_list(MQPeakList* peak_list);

MQPeakList* mq_sort_peaks_by_frequency(MQPeakList* peak_list, int num_peaks);
MQPeakList* mq_find_peaks(int signal_size, sample* signal,
                          MQParameters* params);
MQPeakList* mq_track_peaks(MQPeakList* peak_list, MQParameters* params);

} // end of namespace simpl


#endif
