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

// Find and return all spectral peaks in a given frame of audio
Peaks PeakDetection::find_peaks_in_frame(Frame* frame) {
    Peaks peaks;
    return peaks;
}

// Find and return all spectral peaks in a given audio signal.
// If the signal contains more than 1 frame worth of audio, it will be broken
// up into separate frames, each containing a std::vector of peaks.
// Frames* PeakDetection::find_peaks(const samples& audio)
Frames PeakDetection::find_peaks(int audio_size, sample* audio) {
    clear();
    unsigned int pos = 0;

    while(pos < audio_size - _hop_size) {
        // get the next frame size
        if(!_static_frame_size) {
            _frame_size = next_frame_size();
        }

        // get the next frame
        Frame* f = new Frame(_frame_size);
        f->audio(&audio[pos]);

        // find peaks
        Peaks peaks = find_peaks_in_frame(f);
        f->add_peaks(&peaks);

        _frames.push_back(f);
        pos += _hop_size;
    }

    return _frames;
}


// ---------------------------------------------------------------------------
// SMSPeakDetection
// ---------------------------------------------------------------------------
SMSPeakDetection::SMSPeakDetection() {
}
