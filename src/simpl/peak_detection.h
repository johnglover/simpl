#ifndef PEAK_DETECTION_H
#define PEAK_DETECTION_H

#include "base.h"

extern "C" {
    #include "sms.h"
}

using namespace std;


namespace simpl
{


// ---------------------------------------------------------------------------
// PeakDetection
//
// Detect spectral peaks
// ---------------------------------------------------------------------------

class PeakDetection {
    protected:
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

        virtual int sampling_rate();
        virtual void sampling_rate(int new_sampling_rate);
        virtual int frame_size();
        virtual void frame_size(int new_frame_size);
        virtual bool static_frame_size();
        virtual void static_frame_size(bool new_static_frame_size);
        virtual int next_frame_size();
        virtual int hop_size();
        virtual void hop_size(int new_hop_size);
        virtual int max_peaks();
        virtual void max_peaks(int new_max_peaks);
        virtual std::string window_type();
        virtual void window_type(std::string new_window_type);
        virtual int window_size();
        virtual void window_size(int new_window_size);
        virtual sample min_peak_separation();
        virtual void min_peak_separation(sample new_min_peak_separation);
        int num_frames();
        Frame* frame(int frame_number);
        Frames frames();
        void frames(Frames new_frames);

        // Find and return all spectral peaks in a given frame of audio
        virtual Peaks find_peaks_in_frame(Frame* frame);

        // Find and return all spectral peaks in a given audio signal.
        // If the signal contains more than 1 frame worth of audio, it will be broken
        // up into separate frames, with an array of peaks returned for each frame.
        virtual Frames find_peaks(int audio_size, sample* audio);
};


// ---------------------------------------------------------------------------
// SMSPeakDetection
// ---------------------------------------------------------------------------
class SMSPeakDetection : public PeakDetection {
    private:
        SMSAnalysisParams _analysis_params;
        SMSSpectralPeaks _peaks;

    public:
        SMSPeakDetection();
        ~SMSPeakDetection();
        int next_frame_size();
        void hop_size(int new_hop_size);
        void max_peaks(int new_max_peaks);
        Peaks find_peaks_in_frame(Frame* frame);
        Frames find_peaks(int audio_size, sample* audio);
};


} // end of namespace Simpl

#endif