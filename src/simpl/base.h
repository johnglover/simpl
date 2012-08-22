#ifndef BASE_H
#define BASE_H

#include <vector>
#include <string>

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
//
// Represents a sinuoidal partial or track, an ordered sequence of Peaks
// ---------------------------------------------------------------------------
class Partial {
    private:
        int _starting_frame;
        long _partial_number;
        Peaks _peaks;

    public:
        Partial();
        ~Partial();

        void add_peak(Peak* peak);
        int length();
        int first_frame_number();
        int last_frame_number();
        Peak* peak(int peak_number);
};

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
        int _synth_size;
        int _max_peaks;
        int _num_peaks;
        int _max_partials;
        int _num_partials;
        Peaks _peaks;
        Peaks _partials;
        sample* _audio;
        sample* _synth;
        sample* _residual;
        sample* _synth_residual;
        void init();
        bool _alloc_memory;
        void create_arrays();
        void destroy_arrays();

    public:
        Frame();
        Frame(int frame_size, bool alloc_memory=false);
        ~Frame();

        // peaks
        int num_peaks();
        void num_peaks(int new_num_peaks);
        int max_peaks();
        void max_peaks(int new_max_peaks);
        void add_peak(Peak* peak);
        Peak* peak(int peak_number);
        void peak(int peak_number, Peak* peak);
        void clear();

        // partials
        int num_partials();
        void num_partials(int new_num_partials);
        int max_partials();
        void max_partials(int new_max_partials);
        void add_partial(Peak* peak);
        Peak* partial(int partial_number);
        void partial(int partial_number, Peak* peak);

        // audio buffers
        int size();
        void size(int new_size);
        int synth_size();
        void synth_size(int new_size);
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

} // end of namespace simpl

#endif
