#ifndef BASE_H
#define BASE_H

#include "string.h"
#include "stdio.h"

#include <vector>
#include <string>

#include "exceptions.h"

namespace simpl
{

typedef double simpl_sample;

// sample is ambiguous in this context, how about simpl_sample?

// typedef double sample;


// ---------------------------------------------------------------------------
// Peak
//
// A spectral Peak
// ---------------------------------------------------------------------------
class Peak {
    public:
        simpl_sample amplitude;
        simpl_sample frequency;
        simpl_sample phase;
        simpl_sample bandwidth;

        Peak();
        Peak(simpl_sample new_amplitude, simpl_sample new_frequency,
             simpl_sample new_phase, simpl_sample new_bandwidth);
        ~Peak();
        void reset();
};

typedef std::vector<Peak*> Peaks;


// ---------------------------------------------------------------------------
// Frame
//
// Represents a frame of audio information.
// This can be: - raw audio samples
//              - an unordered list of sinusoidal peaks
//              - an ordered list of partials
//              - synthesised audio samples
//              - residual samples
//              - synthesised residual samples
// ---------------------------------------------------------------------------
class Frame {
    private:
        int _size;
        int _synth_size;
        int _max_peaks;
        int _num_peaks;
        int _max_partials;
        int _num_partials;
        Peaks _peaks;
        Peaks _partials;
        simpl_sample* _audio;
        simpl_sample* _synth;
        simpl_sample* _residual;
        simpl_sample* _synth_residual;
        void init();
        bool _alloc_memory;
        void create_arrays();
        void destroy_arrays();
        void create_synth_arrays();
        void destroy_synth_arrays();
        void resize_peaks(int new_num_peaks);
        void resize_partials(int new_num_partials);

    public:
        Frame();
        Frame(int frame_size, bool alloc_memory=false);
        ~Frame();
        void clear();
        void clear_peaks();
        void clear_partials();
        void clear_synth();

        // peaks
        int num_peaks();
        void num_peaks(int new_num_peaks);
        int max_peaks();
        void max_peaks(int new_max_peaks);
        void add_peak(simpl_sample amplitude, simpl_sample frequency,
                      simpl_sample phase, simpl_sample bandwidth);
        Peak* peak(int peak_number);
        void peak(int peak_number, simpl_sample amplitude, simpl_sample frequency,
                  simpl_sample phase, simpl_sample bandwidth);

        // partials
        int num_partials();
        void num_partials(int new_num_partials);
        int max_partials();
        void max_partials(int new_max_partials);
        void add_partial(simpl_sample amplitude, simpl_sample frequency,
                         simpl_sample phase, simpl_sample bandwidth);
        Peak* partial(int partial_number);
        void partial(int partial_number, simpl_sample amplitude, simpl_sample frequency,
                     simpl_sample phase, simpl_sample bandwidth);

        // audio buffers
        int size();
        void size(int new_size);
        int synth_size();
        void synth_size(int new_size);
        void audio(simpl_sample* new_audio);
        void audio(simpl_sample* new_audio, int size);
        simpl_sample* audio();
        void synth(simpl_sample* new_synth);
        void synth(simpl_sample* new_synth, int size);
        simpl_sample* synth();
        void residual(simpl_sample* new_residual);
        void residual(simpl_sample* new_residual, int size);
        simpl_sample* residual();
        void synth_residual(simpl_sample* new_synth_residual);
        void synth_residual(simpl_sample* new_synth_residual, int size);
        simpl_sample* synth_residual();
};

typedef std::vector<Frame*> Frames;

} // end of namespace simpl

#endif
