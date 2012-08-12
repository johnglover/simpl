#ifndef SYNTHESIS_H
#define SYNTHESIS_H

#include "base.h"

extern "C" {
    #include "sms.h"
}

#include "SndObj.h"
#include "HarmTable.h"
#include "SinAnal.h"
#include "AdSyn.h"

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
        int frame_size();
        void frame_size(int new_frame_size);
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
// SMSSynthesis
// ---------------------------------------------------------------------------
class SMSSynthesis : public Synthesis {
    private:
        SMSSynthParams _synth_params;
        SMSData _data;

    public:
        SMSSynthesis();
        ~SMSSynthesis();
        void hop_size(int new_hop_size);
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
        AdSyn* _synth;
        void reset();

    public:
        SndObjSynthesis();
        ~SndObjSynthesis();
        void hop_size(int new_hop_size);
        void max_partials(int new_max_partials);
        void synth_frame(Frame* frame);
};

} // end of namespace Simpl

#endif
