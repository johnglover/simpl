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
