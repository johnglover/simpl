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
    _alloc_memory = false;
    init();
}

Frame::Frame(int frame_size, bool alloc_memory) {
    _size = frame_size;
    _alloc_memory = alloc_memory;
    init();

    if(_alloc_memory) {
        create_arrays();
    }
}

Frame::~Frame() {
    _peaks.clear();
    _partials.clear();

    if(_alloc_memory) {
        destroy_arrays();
    }
}

void Frame::init() {
    _num_peaks = 0;
    _max_peaks = 100;
    _num_partials = 0;
    _max_partials = 100;
    _peaks.resize(_max_peaks);
    _partials.resize(_max_partials);
    _audio = NULL;
    _synth = NULL;
    _residual = NULL;
    _synth_residual = NULL;
}

void Frame::create_arrays() {
    _audio = new sample[_size];
    _synth = new sample[_size];
    _residual = new sample[_size];
    _synth_residual = new sample[_size];

    memset(_audio, 0.0, sizeof(sample) * _size);
    memset(_synth, 0.0, sizeof(sample) * _size);
    memset(_residual, 0.0, sizeof(sample) * _size);
    memset(_synth_residual, 0.0, sizeof(sample) * _size);
}

void Frame::destroy_arrays() {
    delete [] _audio;
    delete [] _synth;
    delete [] _residual;
    delete [] _synth_residual;
}

// Frame - peaks
// -------------

int Frame::num_peaks() {
    return _num_peaks;
}

void Frame::num_peaks(int new_num_peaks) {
    _num_peaks = new_num_peaks;
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
    _peaks[_num_peaks] = peak;
    _num_peaks++;
}

Peak* Frame::peak(int peak_number) {
    return _peaks[peak_number];
}

void Frame::peak(int peak_number, Peak* peak) {
    _peaks[peak_number] = peak;
}

void Frame::clear() {
    _peaks.clear();
    _partials.clear();
    _num_peaks = 0;
    _num_partials = 0;
}

// Frame - partials
// ----------------

int Frame::num_partials() {
    return _num_partials;
}

void Frame::num_partials(int new_num_partials) {
    _num_partials = new_num_partials;
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

void Frame::add_partial(Peak* peak) {
    _partials[_num_partials] = peak;
    _num_partials++;
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

    if(_alloc_memory) {
        destroy_arrays();
        create_arrays();
    }
}

void Frame::audio(sample* new_audio) {
    if(_alloc_memory) {
        memcpy(_audio, new_audio, sizeof(sample) * _size);
    }
    else {
        _audio = new_audio;
    }
}

sample* Frame::audio() {
    return _audio;
}

void Frame::synth(sample* new_synth) {
    if(_alloc_memory) {
        memcpy(_synth, new_synth, sizeof(sample) * _size);
    }
    else {
        _synth = new_synth;
    }
}

sample* Frame::synth() {
    return _synth;
}

void Frame::residual(sample* new_residual) {
    if(_alloc_memory) {
        memcpy(_residual, new_residual, sizeof(sample) * _size);
    }
    else {
        _residual = new_residual;
    }
}

sample* Frame::residual() {
    return _residual;
}

void Frame::synth_residual(sample* new_synth_residual) {
    if(_alloc_memory) {
        memcpy(_synth_residual, new_synth_residual, sizeof(sample) * _size);
    }
    else {
        _synth_residual = new_synth_residual;
    }
}

sample* Frame::synth_residual() {
    return _synth_residual;
}
