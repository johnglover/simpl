#ifndef SYNTHESIS_H
#define SYNTHESIS_H

#include <math.h>

#include "base.h"

extern "C" {
    #include "sms.h"
}

#include "SndObj.h"
#include "HarmTable.h"
#include "SinAnal.h"
#include "AdSyn.h"

#include "Breakpoint.h"
#include "Oscillator.h"

using namespace std;

namespace simpl
{


// ---------------------------------------------------------------------------
// Synthesis
//
// Synthesise audio from spectral analysis data
// ---------------------------------------------------------------------------

class Synthesis {
    protected:
        int _frame_size;
        int _hop_size;
        int _max_partials;
        int _sampling_rate;

    public:
        Synthesis();
        virtual void reset();
        int frame_size();
        virtual void frame_size(int new_frame_size);
        int hop_size();
        virtual void hop_size(int new_hop_size);
        int max_partials();
        virtual void max_partials(int new_max_partials);
        int sampling_rate();
        void sampling_rate(int new_sampling_rate);

        virtual void synth_frame(Frame* frame);
        virtual Frames synth(Frames frames);
};


// ---------------------------------------------------------------------------
// MQSynthesis
// ---------------------------------------------------------------------------
class MQSynthesis : public Synthesis {
    private:
        simpl_sample* _prev_amps;
        simpl_sample* _prev_freqs;
        simpl_sample* _prev_phases;
        simpl_sample hz_to_radians(simpl_sample f);

    public:
        MQSynthesis();
        ~MQSynthesis();
        void reset();
        using Synthesis::max_partials;
        void max_partials(int new_max_partials);
        void synth_frame(Frame* frame);
};


// ---------------------------------------------------------------------------
// SMSSynthesis
// ---------------------------------------------------------------------------
class SMSSynthesis : public Synthesis {
    private:
        SMSSynthParams _synth_params;
        SMSData _data;

    public:
        SMSSynthesis();
        ~SMSSynthesis();
        using Synthesis::hop_size;
        void hop_size(int new_hop_size);
        using Synthesis::max_partials;
        void max_partials(int new_max_partials);
        int num_stochastic_coeffs();
        int stochastic_type();
        int det_synthesis_type();
        void det_synthesis_type(int new_det_synthesis_type);
        void synth_frame(Frame* frame);
};


// ---------------------------------------------------------------------------
// SndObjSynthesis
// ---------------------------------------------------------------------------
class SimplSndObjAnalysisWrapper : public SinAnal {
    public:
        SimplSndObjAnalysisWrapper(int max_partials);
        ~SimplSndObjAnalysisWrapper();
        Peaks partials;
        int GetTrackID(int track);
        int GetTracks();
        double Output(int pos);
};


class SndObjSynthesis : public Synthesis {
    private:
        SimplSndObjAnalysisWrapper* _analysis;
        HarmTable* _table;
        SimplAdSyn* _synth;

    public:
        SndObjSynthesis();
        ~SndObjSynthesis();
        void reset();
        using Synthesis::frame_size;
        void frame_size(int new_frame_size);
        using Synthesis::hop_size;
        void hop_size(int new_hop_size);
        using Synthesis::max_partials;
        void max_partials(int new_max_partials);
        void synth_frame(Frame* frame);
};


// ---------------------------------------------------------------------------
// LorisSynthesis
// ---------------------------------------------------------------------------
class LorisSynthesis : public Synthesis {
    private:
        std::vector<Loris::Oscillator> _oscs;
        simpl_sample _bandwidth;

    public:
        LorisSynthesis();
        ~LorisSynthesis();
        void reset();
        using Synthesis::max_partials;
        void max_partials(int new_max_partials);
        simpl_sample bandwidth();
        void bandwidth(simpl_sample new_bandwidth);
        void synth_frame(Frame* frame);
};

} // end of namespace Simpl

#endif
