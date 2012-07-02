#ifndef PARTIAL_TRACKING_H
#define PARTIAL_TRACKING_H

#include "base.h"

extern "C" {
    #include "sms.h"
}

using namespace std;

namespace simpl
{


// ---------------------------------------------------------------------------
// PartialTracking
//
// Link spectral peaks from consecutive frames to form partials
// ---------------------------------------------------------------------------

class PartialTracking {
    protected:
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
        virtual void sampling_rate(int new_sampling_rate);
        int max_partials();
        virtual void max_partials(int new_max_partials);
        int min_partial_length();
        virtual void min_partial_length(int new_min_partial_length);
        int max_gap();
        virtual void max_gap(int new_max_gap);

        virtual Peaks update_partials(Frame* frame);
        virtual Frames find_partials(Frames frames);
};


// ---------------------------------------------------------------------------
// SMSPartialTracking
// ---------------------------------------------------------------------------
class SMSPartialTracking : public PartialTracking {
    private:
        SMSAnalysisParams _analysis_params;
        SMSHeader _header;
        SMSData _data;
        int _num_peaks;
        sample* _peak_amplitude;
        sample* _peak_frequency;
        sample* _peak_phase;
        void init_peaks();

    public:
        SMSPartialTracking();
        ~SMSPartialTracking();
        void max_partials(int new_max_partials);
        Peaks update_partials(Frame* frame);
};


} // end of namespace Simpl

#endif
