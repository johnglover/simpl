#ifndef PARTIAL_TRACKING_H
#define PARTIAL_TRACING_H

#include "base.h"

using namespace std;

namespace simpl
{


// ---------------------------------------------------------------------------
// PartialTracking
//
// Link spectral peaks from consecutive frames to form partials
// ---------------------------------------------------------------------------

class PartialTracking {
    private:
        int _sampling_rate;
        int _max_partials;
        int _min_partial_length;
        int _max_gap;
        Frames _frames;

    public:
        PartialTracking();
        ~PartialTracking();

        void clear();

        int sampling_rate();
        void sampling_rate(int new_sampling_rate);
        int max_partials();
        void max_partials(int new_max_partials);
        int min_partial_length();
        void min_partial_length(int new_min_partial_length);
        int max_gap();
        void max_gap(int new_max_gap);

        virtual Peaks update_partials(Frame* frame);
        virtual Frames find_partials(Frames frames);
};


} // end of namespace Simpl

#endif
