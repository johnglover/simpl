#include "base.h"

using namespace std;
using namespace simpl;


// ---------------------------------------------------------------------------
// Peak
// ---------------------------------------------------------------------------
Peak::Peak() {
    amplitude = 0.0;
    frequency = 0.0;
    phase = 0.0;
    next_peak = NULL;
    previous_peak = NULL;
    partial_id = 0;
    partial_position = 0;
    frame_number = 0;
}

Peak::~Peak() {
}

// Returns true iff this peak is unmatched in the given direction, and has positive amplitude
bool Peak::is_free(const string direction) {
    if(amplitude <= 0.0) {
        return false;
    }

    if(direction == "forwards") {
        if(next_peak != NULL) {
            return false;
        }
    }
    else if(direction == "backwards") {
        if(previous_peak != NULL) {
            return false;
        }
    }
    else {
        return false;
    }

    return true;
}


// ---------------------------------------------------------------------------
// Partial
// ---------------------------------------------------------------------------
Partial::Partial() {
    _starting_frame = 0;
    _partial_number = -1;
}

Partial::~Partial() {
    _peaks.clear();
}

void Partial::add_peak(Peak* peak) {
}

int Partial::length() {
    return _peaks.size();
}

int Partial::first_frame_number() {
    return _starting_frame;
}

int Partial::last_frame_number() {
    return _starting_frame + length();
}

Peak* Partial::peak(int peak_number) {
    return _peaks[peak_number];
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------
Frame::Frame() {
    _size = 512;
    init();
}

Frame::Frame(int frame_size) {
    _size = frame_size;
    init();
}

Frame::~Frame() {
    _peaks.clear();
    _partials.clear();
}

void Frame::init() {
    _max_peaks = 100;
    _max_partials = 100;
    _partials.resize(_max_partials);
    _audio = NULL;
    _synth = NULL;
    _residual = NULL;
    _synth_residual = NULL;
}

// Frame - peaks
// -------------

int Frame::num_peaks() {
    return _peaks.size();
}

int Frame::max_peaks() {
    return _max_peaks;
}

void Frame::max_peaks(int new_max_peaks) {
    _max_peaks = new_max_peaks;

    // TODO: potentially losing data here, should prevent or complain
    if((int)_peaks.size() > _max_peaks) {
        _peaks.resize(_max_peaks);
    }
}

void Frame::add_peak(Peak* peak) {
    _peaks.push_back(peak);
}

void Frame::add_peaks(Peaks* peaks) {
    for(Peaks::iterator i = peaks->begin(); i != peaks->end(); i++) {
        add_peak(*i);
    }
}

Peak* Frame::peak(int peak_number) {
    return _peaks[peak_number];
}

void Frame::clear() {
    _peaks.clear();
    _partials.clear();
}

// Frame - partials
// ----------------

int Frame::num_partials() {
    return _partials.size();
}

int Frame::max_partials() {
    return _max_partials;
}

void Frame::max_partials(int new_max_partials) {
    _max_partials = new_max_partials;

    // TODO: potentially losing data here, should prevent or complain
    if((int)_partials.size() > _max_partials) {
        _partials.resize(_max_partials);
    }
}

Peak* Frame::partial(int partial_number) {
    return _partials[partial_number];
}

void Frame::partial(int partial_number, Peak* peak) {
    _partials[partial_number] = peak;
}


// Frame - audio buffers
// ---------------------

int Frame::size() {
    return _size;
}

void Frame::size(int new_size) {
    _size = new_size;
}

void Frame::audio(sample* new_audio) {
    _audio = new_audio;
}

sample* Frame::audio() {
    return _audio;
}

void Frame::synth(sample* new_synth) {
    _synth = new_synth;
}

sample* Frame::synth() {
    return _synth;
}

void Frame::residual(sample* new_residual) {
    _residual = new_residual;
}

sample* Frame::residual() {
    return _residual;
}

void Frame::synth_residual(sample* new_synth_residual) {
    _synth_residual = new_synth_residual;
}

sample* Frame::synth_residual() {
    return _synth_residual;
}


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
// PartialTracking
// ---------------------------------------------------------------------------
PartialTracking::PartialTracking() {
    _sampling_rate = 44100;
    _max_partials = 100;
    _min_partial_length = 0;
    _max_gap = 2;
}

PartialTracking::~PartialTracking() {
    clear();
}

void PartialTracking::clear() {
    _frames.clear();
}

int PartialTracking::sampling_rate() {
    return _sampling_rate;
}

void PartialTracking::sampling_rate(int new_sampling_rate) {
    _sampling_rate = new_sampling_rate;
}

int PartialTracking::max_partials() {
    return _max_partials;
}

void PartialTracking::max_partials(int new_max_partials) {
    _max_partials = new_max_partials;
}

int PartialTracking::min_partial_length() {
    return _min_partial_length;
}

void PartialTracking::min_partial_length(int new_min_partial_length) {
    _min_partial_length = new_min_partial_length;
}

int PartialTracking::max_gap() {
    return _max_gap;
}

void PartialTracking::max_gap(int new_max_gap) {
    _max_gap = new_max_gap;
}

// Streamable (real-time) partial-tracking.
Peaks PartialTracking::update_partials(Frame* frame) {
    Peaks peaks;
    return peaks;
}

// Find partials from the sinusoidal peaks in a list of Frames.
Frames PartialTracking::find_partials(Frames frames) {
    for(int i = 0; i < frames.size(); i++) {
        Peaks peaks = update_partials(frames[i]);
        for(int j = 0; j < peaks.size(); j++) {
            frames[i]->partial(j, peaks[j]);
        }
    }
    _frames = frames;
    return _frames;
}


// ---------------------------------------------------------------------------
// Synthesis
// ---------------------------------------------------------------------------
Synthesis::Synthesis() {
    _frame_size = 512;
    _hop_size = 512;
    _max_partials = 100;
    _sampling_rate = 44100;
}

int Synthesis::frame_size() {
    return _frame_size;
}

void Synthesis::frame_size(int new_frame_size) {
    _frame_size = new_frame_size;
}

int Synthesis::hop_size() {
    return _hop_size;
}

void Synthesis::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
}

int Synthesis::max_partials() {
    return _max_partials;
}

void Synthesis::max_partials(int new_max_partials) {
    _max_partials = new_max_partials;
}

int Synthesis::sampling_rate() {
    return _sampling_rate;
}

void Synthesis::sampling_rate(int new_sampling_rate) {
    _sampling_rate = new_sampling_rate;
}

void Synthesis::synth_frame(Frame* frame) {
}

Frames Synthesis::synth(Frames frames) {
    for(int i = 0; i < frames.size(); i++) {
        synth_frame(frames[i]);
    }
    return frames;
}


// ---------------------------------------------------------------------------
// Residual
// ---------------------------------------------------------------------------
Residual::Residual() {
    _frame_size = 512;
    _hop_size = 512;
    _sampling_rate = 44100;
}

int Residual::frame_size() {
    return _frame_size;
}

void Residual::frame_size(int new_frame_size) {
    _frame_size = new_frame_size;
}

int Residual::hop_size() {
    return _hop_size;
}

void Residual::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
}

int Residual::sampling_rate() {
    return _sampling_rate;
}

void Residual::sampling_rate(int new_sampling_rate) {
    _sampling_rate = new_sampling_rate;
}

void Residual::residual_frame(int synth_size, sample* synth,
                              int original_size, sample* original,
                              int residual_size, sample* residual) {
}

void Residual::find_residual(int synth_size, sample* synth,
                             int original_size, sample* original,
                             int residual_size, sample* residual) {
    for(int i = 0; i < synth_size; i += _hop_size) {
        residual_frame(_hop_size, &synth[i],
                       _hop_size, &original[i],
                       _hop_size, &residual[i]);
    }
}

void Residual::synth_frame(Frame* frame) {
}

// Calculate and return a synthesised residual signal
Frames Residual::synth(Frames frames) {
    for(int i = 0; i < frames.size(); i++) {
        synth_frame(frames[i]);
    }
    return frames;
}
