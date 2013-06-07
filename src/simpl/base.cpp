#include "base.h"

using namespace std;
using namespace simpl;


// ---------------------------------------------------------------------------
// Peak
// ---------------------------------------------------------------------------
Peak::Peak() {
    reset();
}

Peak::Peak(sample new_amplitude, sample new_frequency,
           sample new_phase, sample new_bandwidth) {
    amplitude = new_amplitude;
    frequency = new_frequency;
    phase = new_phase;
    bandwidth = new_bandwidth;
}

Peak::~Peak() {
}

void Peak::reset() {
    amplitude = 0.0;
    frequency = 0.0;
    phase = 0.0;
    bandwidth = 0.0;
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
    clear_peaks();
    clear_partials();

    for(int i = 0; i < _peaks.size(); i++) {
        if(_peaks[i]) {
            delete _peaks[i];
            _peaks[i] = NULL;
        }
    }

    for(int i = 0; i < _partials.size(); i++) {
        if(_partials[i]) {
            delete _partials[i];
            _partials[i] = NULL;
        }
    }

    if(_alloc_memory) {
        destroy_arrays();
    }
}

void Frame::init() {
    _num_peaks = 0;
    _max_peaks = 100;
    _num_partials = 0;
    _max_partials = 100;
    _audio = NULL;
    _synth = NULL;
    _residual = NULL;
    _synth_residual = NULL;
    resize_peaks(_max_peaks);
    resize_partials(_max_partials);
}

void Frame::create_arrays() {
    _audio = new sample[_size];
    _residual = new sample[_size];
    memset(_audio, 0.0, sizeof(sample) * _size);
    memset(_residual, 0.0, sizeof(sample) * _size);
    create_synth_arrays();
}

void Frame::destroy_arrays() {
    if(_audio) {
        delete [] _audio;
        _audio = NULL;
    }
    if(_residual) {
        delete [] _residual;
        _residual = NULL;
    }

    destroy_synth_arrays();
}

void Frame::create_synth_arrays() {
    _synth = new sample[_synth_size];
    _synth_residual = new sample[_synth_size];
    memset(_synth, 0.0, sizeof(sample) * _synth_size);
    memset(_synth_residual, 0.0, sizeof(sample) * _synth_size);
}

void Frame::destroy_synth_arrays() {
    if(_synth) {
        delete [] _synth;
        _synth = NULL;
    }
    if(_synth_residual) {
        delete [] _synth_residual;
        _synth_residual = NULL;
    }
}

void Frame::resize_peaks(int new_num_peaks) {
    clear_peaks();

    for(int i = 0; i < _peaks.size(); i++) {
        if(_peaks[i]) {
            delete _peaks[i];
            _peaks[i] = NULL;
        }
    }

    _peaks.resize(new_num_peaks);

    for(int i = 0; i < _peaks.size(); i++) {
        _peaks[i] = new Peak();
    }
}

void Frame::resize_partials(int new_num_partials) {
    clear_partials();

    for(int i = 0; i < _partials.size(); i++) {
        if(_partials[i]) {
            delete _partials[i];
            _partials[i] = NULL;
        }
    }

    _partials.resize(new_num_partials);

    for(int i = 0; i < _partials.size(); i++) {
        _partials[i] = new Peak();
    }
}

void Frame::clear() {
    clear_peaks();
    clear_partials();

    if(_alloc_memory) {
        memset(_audio, 0.0, sizeof(sample) * _size);
    }
    clear_synth();
}

void Frame::clear_peaks() {
    _num_peaks = 0;
    for(int i = 0; i < _peaks.size(); i++) {
        if(_peaks[i]) {
            _peaks[i]->reset();
        }
    }
}

void Frame::clear_partials() {
    _num_partials = 0;
    for(int i = 0; i < _partials.size(); i++) {
        if(_partials[i]) {
            _partials[i]->reset();
        }
    }
}

void Frame::clear_synth() {
    if(_alloc_memory) {
        memset(_synth, 0.0, sizeof(sample) * _synth_size);
        memset(_residual, 0.0, sizeof(sample) * _size);
        memset(_synth_residual, 0.0, sizeof(sample) * _synth_size);
    }
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

    if(_num_peaks > _max_peaks) {
        // losing data here, allow but warn
        printf("Warning: max peaks changed to less than current number "
               "of peaks, some existing data was lost.\n");
    }

    resize_peaks(_max_peaks);
}

void Frame::add_peak(Peak* peak) {
    if(_num_peaks >= _max_peaks) {
        printf("Warning: attempted to add more than the specified"
               " maximum number of peaks (%d) to a frame, ignoring.\n",
               _max_peaks);
        return;
    }

    if(_peaks[_num_peaks]) {
        delete _peaks[_num_peaks];
    }
    _peaks[_num_peaks] = peak;
    _num_peaks++;
}

void Frame::add_peak(sample amplitude, sample frequency,
                     sample phase, sample bandwidth) {
    _peaks[_num_peaks]->amplitude = amplitude;
    _peaks[_num_peaks]->frequency = frequency;
    _peaks[_num_peaks]->phase = phase;
    _peaks[_num_peaks]->bandwidth = bandwidth;
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

    if(_num_partials > _max_partials) {
        // losing data here, allow but warn
        printf("Warning: max partials changed to less than current number"
               " of partials, some existing data was lost.\n");
    }

    resize_partials(_max_partials);
}

void Frame::add_partial(Peak* peak) {
    if(_num_partials >= _max_partials) {
        printf("Warning: attempted to add more than the specified"
               " maximum number of partials (%d) to a frame, ignoring.\n",
               _max_partials);
        return;
    }

    _partials[_num_partials] = peak;
    _num_partials++;
}

void Frame::add_partial(sample amplitude, sample frequency,
                        sample phase, sample bandwidth) {
    _partials[_num_partials]->amplitude = amplitude;
    _partials[_num_partials]->frequency = frequency;
    _partials[_num_partials]->phase = phase;
    _partials[_num_partials]->bandwidth = bandwidth;
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
        destroy_synth_arrays();
        create_synth_arrays();
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
