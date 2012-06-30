#ifndef SYNTHESIS_H
#define SYNTHESIS_H

#include "base.h"

using namespace std;

namespace simpl
{


// ---------------------------------------------------------------------------
// Synthesis
//
// Synthesise audio from spectral analysis data
// ---------------------------------------------------------------------------

class Synthesis {
    private:
        int _frame_size;
        int _hop_size;
        int _max_partials;
        int _sampling_rate;

    public:
        Synthesis();
        int frame_size();
        void frame_size(int new_frame_size);
        int hop_size();
        void hop_size(int new_hop_size);
        int max_partials();
        void max_partials(int new_max_partials);
        int sampling_rate();
        void sampling_rate(int new_sampling_rate);

        virtual void synth_frame(Frame* frame);
        virtual Frames synth(Frames frames);
};


} // end of namespace Simpl

#endif
