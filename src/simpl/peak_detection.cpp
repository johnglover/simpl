#include "peak_detection.h"

using namespace std;
using namespace simpl;


// ---------------------------------------------------------------------------
// PeakDetection
// ---------------------------------------------------------------------------

PeakDetection::PeakDetection() {
    _sampling_rate = 44100;
    _frame_size = 2048;
    _static_frame_size = true;
    _hop_size = 512;
    _max_peaks = 100;
    _window_type = "hamming";
    _window_size = 2048;
    _min_peak_separation = 1.0; // in Hz
}

PeakDetection::~PeakDetection() {
    clear();
}

void PeakDetection::clear() {
    for(int i = 0; i < _frames.size(); i++) {
        if(_frames[i]) {
            delete _frames[i];
        }
    }

    _frames.clear();
}

int PeakDetection::sampling_rate() {
    return _sampling_rate;
}

void PeakDetection::sampling_rate(int new_sampling_rate) {
    _sampling_rate = new_sampling_rate;
}

int PeakDetection::frame_size() {
    return _frame_size;
}

void PeakDetection::frame_size(int new_frame_size) {
    _frame_size = new_frame_size;
}

bool PeakDetection::static_frame_size() {
    return _static_frame_size;
}

void PeakDetection::static_frame_size(bool new_static_frame_size) {
    _static_frame_size = new_static_frame_size;
}

int PeakDetection::next_frame_size() {
    return _frame_size;
}

int PeakDetection::hop_size() {
    return _hop_size;
}

void PeakDetection::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
}

int PeakDetection::max_peaks() {
    return _max_peaks;
}

void PeakDetection::max_peaks(int new_max_peaks) {
    _max_peaks = new_max_peaks;
}

std::string PeakDetection::window_type() {
    return _window_type;
}

void PeakDetection::window_type(std::string new_window_type) {
    _window_type = new_window_type;
}

int PeakDetection::window_size() {
    return _window_size;
}

void PeakDetection::window_size(int new_window_size) {
    _window_size = new_window_size;
}

sample PeakDetection::min_peak_separation() {
    return _min_peak_separation;
}

void PeakDetection::min_peak_separation(sample new_min_peak_separation) {
    _min_peak_separation = new_min_peak_separation;
}

int PeakDetection::num_frames() {
    return _frames.size();
}

Frame* PeakDetection::frame(int frame_number) {
    return _frames[frame_number];
}

Frames PeakDetection::frames() {
    return _frames;
}

void PeakDetection::frames(Frames new_frames) {
    _frames = new_frames;
}

// Find and return all spectral peaks in a given frame of audio
Peaks PeakDetection::find_peaks_in_frame(Frame* frame) {
    Peaks peaks;

    for(int i = 0; i < peaks.size(); i++) {
        frame->add_peak(peaks[i]);
    }

    return peaks;
}

// Find and return all spectral peaks in a given audio signal.
// If the signal contains more than 1 frame worth of audio, it will be broken
// up into separate frames, each containing a std::vector of peaks.
// Frames* PeakDetection::find_peaks(const samples& audio)
Frames PeakDetection::find_peaks(int audio_size, sample* audio) {
    clear();
    unsigned int pos = 0;

    while(pos <= audio_size - _hop_size) {
        // get the next frame size
        if(!_static_frame_size) {
            _frame_size = next_frame_size();
        }

        // get the next frame
        Frame* f = new Frame(_frame_size);
        f->audio(&audio[pos]);
        f->max_peaks(_max_peaks);

        // find peaks
        find_peaks_in_frame(f);

        _frames.push_back(f);
        pos += _hop_size;
    }

    return _frames;
}


// ---------------------------------------------------------------------------
// SMSPeakDetection
// ---------------------------------------------------------------------------
SMSPeakDetection::SMSPeakDetection() {
    sms_init();

    sms_initAnalParams(&_analysis_params);
    _analysis_params.iSamplingRate = _sampling_rate;
    _analysis_params.iFrameRate = _sampling_rate / _hop_size;
    _analysis_params.iWindowType = SMS_WIN_HAMMING;
    _analysis_params.fHighestFreq = 20000;
    _analysis_params.iMaxDelayFrames = 4;
    _analysis_params.analDelay = 0;
    _analysis_params.minGoodFrames = 1;
    _analysis_params.iCleanTracks = 0;
    _analysis_params.iFormat = SMS_FORMAT_HP;
    _analysis_params.nTracks = _max_peaks;
    _analysis_params.maxPeaks = _max_peaks;
    _analysis_params.nGuides = _max_peaks;
    _analysis_params.preEmphasis = 0;
    sms_initAnalysis(&_analysis_params);
    _analysis_params.iSizeSound = _hop_size;

    sms_initSpectralPeaks(&_peaks, _max_peaks);

    // By default, SMS will change the size of the frames being read
    // depending on the detected fundamental frequency (if any) of the
    // input sound. To prevent this behaviour (useful when comparing
    // different analysis algorithms), set the
    // _static_frame_size variable to True
    _static_frame_size = false;
}

SMSPeakDetection::~SMSPeakDetection() {
    sms_freeAnalysis(&_analysis_params);
    sms_freeSpectralPeaks(&_peaks);
    sms_free();
}

int SMSPeakDetection::next_frame_size() {
    return _analysis_params.sizeNextRead;
}

void SMSPeakDetection::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
    sms_freeAnalysis(&_analysis_params);
    _analysis_params.iFrameRate = _sampling_rate / _hop_size;
    sms_initAnalysis(&_analysis_params);
    _analysis_params.iSizeSound = _hop_size;
}

void SMSPeakDetection::max_peaks(int new_max_peaks) {
    _max_peaks = new_max_peaks;
    if(_max_peaks > SMS_MAX_NPEAKS) {
        _max_peaks = SMS_MAX_NPEAKS;
    }

    sms_freeAnalysis(&_analysis_params);
    sms_freeSpectralPeaks(&_peaks);

    _analysis_params.nTracks = _max_peaks;
    _analysis_params.maxPeaks = _max_peaks;
    _analysis_params.nGuides = _max_peaks;

    sms_initAnalysis(&_analysis_params);
    sms_initSpectralPeaks(&_peaks, _max_peaks);
}

// Find and return all spectral peaks in a given frame of audio
Peaks SMSPeakDetection::find_peaks_in_frame(Frame* frame) {
    Peaks peaks;

    int num_peaks = sms_findPeaks(frame->size(), frame->audio(), 
                                  &_analysis_params, &_peaks);

    for(int i = 0; i < num_peaks; i++) {
        Peak* p = new Peak();
        p->amplitude = _peaks.pSpectralPeaks[i].fMag;
        p->frequency = _peaks.pSpectralPeaks[i].fFreq;
        p->phase = _peaks.pSpectralPeaks[i].fPhase;
        peaks.push_back(p);

        frame->add_peak(p);
    }
    return peaks;
}

// Find and return all spectral peaks in a given audio signal.
// If the signal contains more than 1 frame worth of audio,
// it will be broken up into separate frames, with a list of
// peaks returned for each frame.
Frames SMSPeakDetection::find_peaks(int audio_size, sample* audio) {
    clear();

    _analysis_params.iSizeSound = audio_size;
    unsigned int pos = 0;

    while(pos <= audio_size - _hop_size) {
        // get the next frame size
        // if(!_static_frame_size) {
        _frame_size = next_frame_size();
        // }

        // get the next frame
        Frame* f = new Frame(_frame_size);
        f->audio(&audio[pos]);
        f->max_peaks(_max_peaks);

        // find peaks
        find_peaks_in_frame(f);

        _frames.push_back(f);

        if(!_static_frame_size) {
            pos += _frame_size;
        }
        else {
            pos += _hop_size;
        }
    }

    return _frames;
}
