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
    bandwidth = 0.0;
}

Peak::~Peak() {
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
    _synth_size = 512;
    _alloc_memory = false;
    init();
}

Frame::Frame(int frame_size, bool alloc_memory) {
    _size = frame_size;
    _synth_size = 512;
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
    _synth = new sample[_synth_size];
    _residual = new sample[_size];
    _synth_residual = new sample[_synth_size];

    memset(_audio, 0.0, sizeof(sample) * _size);
    memset(_synth, 0.0, sizeof(sample) * _synth_size);
    memset(_residual, 0.0, sizeof(sample) * _size);
    memset(_synth_residual, 0.0, sizeof(sample) * _synth_size);
}

void Frame::destroy_arrays() {
    delete [] _audio;
    delete [] _synth;
    delete [] _residual;
    delete [] _synth_residual;
}

void Frame::clear() {
    clear_peaks();
    clear_partials();

    if(_alloc_memory) {
        memset(_audio, 0.0, sizeof(sample) * _size);
        memset(_synth, 0.0, sizeof(sample) * _synth_size);
        memset(_residual, 0.0, sizeof(sample) * _size);
        memset(_synth_residual, 0.0, sizeof(sample) * _synth_size);
    }
}

void Frame::clear_peaks() {
    _peaks.clear();
    _num_peaks = 0;
}

void Frame::clear_partials() {
    _partials.clear();
    _num_partials = 0;
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

int Frame::synth_size() {
    return _synth_size;
}

void Frame::synth_size(int new_size) {
    _synth_size = new_size;

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

void Frame::audio(sample* new_audio, int size) {
    // this should only be called if the Frame is managing the memory
    // for the sample arrays
    if(!_alloc_memory) {
        throw Exception(std::string("Memory not managed by Frame."));
    }

    // copy size should also be less than or equal to the current frame size
    if(size > _size) {
        throw Exception(std::string("Specified copy size is too large, "
                                    "it must be less than the Frame size."));
    }

    memcpy(_audio, new_audio, sizeof(sample) * size);
}

sample* Frame::audio() {
    return _audio;
}

void Frame::synth(sample* new_synth) {
    if(_alloc_memory) {
        memcpy(_synth, new_synth, sizeof(sample) * _synth_size);
    }
    else {
        _synth = new_synth;
    }
}

void Frame::synth(sample* new_synth, int size) {
    // this should only be called if the Frame is managing the memory
    // for the sample arrays
    if(!_alloc_memory) {
        throw Exception(std::string("Memory not managed by Frame."));
    }

    // copy size should also be less than or equal to the current frame synth size
    if(size > _synth_size) {
        throw Exception(std::string("Specified copy size is too large, "
                                    "it must be less than the Frame synth size."));
    }

    memcpy(_synth, new_synth, sizeof(sample) * size);
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

void Frame::residual(sample* new_residual, int size) {
    // this should only be called if the Frame is managing the memory
    // for the sample arrays
    if(!_alloc_memory) {
        throw Exception(std::string("Memory not managed by Frame."));
    }

    // copy size should also be less than or equal to the current frame size
    if(size > _size) {
        throw Exception(std::string("Specified copy size is too large, "
                                    "it must be less than the Frame size."));
    }

    memcpy(_residual, new_residual, sizeof(sample) * size);
}

sample* Frame::residual() {
    return _residual;
}

void Frame::synth_residual(sample* new_synth_residual) {
    if(_alloc_memory) {
        memcpy(_synth_residual, new_synth_residual, sizeof(sample) * _synth_size);
    }
    else {
        _synth_residual = new_synth_residual;
    }
}

void Frame::synth_residual(sample* new_synth_residual, int size) {
    // this should only be called if the Frame is managing the memory
    // for the sample arrays
    if(!_alloc_memory) {
        throw Exception(std::string("Memory not managed by Frame."));
    }

    // copy size should also be less than or equal to the current frame synth size
    if(size > _synth_size) {
        throw Exception(std::string("Specified copy size is too large, "
                                    "it must be less than the Frame synth size."));
    }

    memcpy(_synth_residual, new_synth_residual, sizeof(sample) * size);
}

sample* Frame::synth_residual() {
    return _synth_residual;
}
