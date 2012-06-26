#ifndef BASE_H
#define BASE_H

#include <vector>
#include <string>
#include "exceptions.h"

using namespace std;

namespace simpl 
{

typedef double sample;


// ---------------------------------------------------------------------------
// Peak
//
// A spectral Peak
// ---------------------------------------------------------------------------
class Peak {
    public:
        sample amplitude;
        sample frequency;
        sample phase;
        Peak* next_peak;
        Peak* previous_peak;
        int partial_id;
        int partial_position;
        int frame_number;

        Peak();
        ~Peak();

        bool is_start_of_partial() {
            return previous_peak == NULL;
        };
        bool is_free(const string direction = string("forwards"));
};

typedef std::vector<Peak*> Peaks;


// ---------------------------------------------------------------------------
// Partial
// ---------------------------------------------------------------------------
class Partial {};

typedef std::vector<Partial*> Partials;


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
        int _max_peaks;
        int _max_partials;
        Peaks _peaks;
        Partials _partials;
        sample* _audio;
        sample* _synth;
        sample* _residual;
        sample* _synth_residual;
        void init();

    public:
        Frame();
        Frame(int frame_size);
        ~Frame();

        // peaks
        int num_peaks();
        int max_peaks();
        void max_peaks(int new_max_peaks);
        void add_peak(Peak* peak);
        void add_peaks(Peaks* peaks);
        Peak* peak(int peak_number);
        void clear();
        Peaks::iterator peaks_begin();
        Peaks::iterator peaks_end();

        // partials
        int num_partials();
        int max_partials();
        void max_partials(int new_max_partials);
        void add_partial(Partial partial);
        Partials::iterator partials();

        // audio buffers
        int size();
        void size(int new_size);
        void audio(sample* new_audio);
        sample* audio();
        void synth(sample* new_synth);
        sample* synth();
        void residual(sample* new_residual);
        sample* residual();
        void synth_residual(sample* new_synth_residual);
        sample* synth_residual();
};

typedef std::vector<Frame*> Frames;


// ---------------------------------------------------------------------------
// PeakDetection
// 
// Detect spectral peaks
// ---------------------------------------------------------------------------

class PeakDetection {
    private:
        int _sampling_rate;
        int _frame_size;
        bool _static_frame_size;
        int _hop_size;
        int _max_peaks;
        std::string _window_type;
        int _window_size;
        sample _min_peak_separation;
        Frames _frames;

    public:
        PeakDetection();
        virtual ~PeakDetection();
        void clear();

        int sampling_rate();
        void sampling_rate(int new_sampling_rate);
        int frame_size();
        void frame_size(int new_frame_size);
        bool static_frame_size();
        void static_frame_size(bool new_static_frame_size);
        virtual int next_frame_size();
        int hop_size();
        void hop_size(int new_hop_size);
        int max_peaks();
        void max_peaks(int new_max_peaks);
        std::string window_type();
        void window_type(std::string new_window_type);
        int window_size();
        void window_size(int new_window_size);
        sample min_peak_separation();
        void min_peak_separation(sample new_min_peak_separation);
        int num_frames();
        Frame* frame(int frame_number);
        Frames frames();

        // Find and return all spectral peaks in a given frame of audio
        virtual Peaks find_peaks_in_frame(Frame* frame);

        // Find and return all spectral peaks in a given audio signal.
        // If the signal contains more than 1 frame worth of audio, it will be broken
        // up into separate frames, with an array of peaks returned for each frame.
        virtual Frames find_peaks(int audio_size, sample* audio);
};

} // end of namespace Simpl

#endif
