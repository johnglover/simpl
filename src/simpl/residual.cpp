#include "residual.h"

using namespace std;
using namespace simpl;


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

void Residual::residual_frame(Frame* frame) {
}

void Residual::find_residual(int synth_size, sample* synth,
                             int original_size, sample* original,
                             int residual_size, sample* residual) {
    for(int i = 0; i < synth_size; i += _hop_size) {
        Frame* f = new Frame(_hop_size);
        f->audio(&original[i]);
        f->synth(&synth[i]);
        f->residual(&residual[i]);
        residual_frame(f);
    }
}

void Residual::synth_frame(Frame* frame) {
}

// Calculate and return a synthesised residual signal
Frames Residual::synth(Frames& frames) {
    for(int i = 0; i < frames.size(); i++) {
        synth_frame(frames[i]);
    }
    return frames;
}

Frames Residual::synth(int original_size, sample* original) {
    Frames frames;

    for(int i = 0; i <= original_size - _hop_size; i += _hop_size) {
        Frame* f = new Frame(_hop_size, true);
        f->audio(&original[i]);
        synth_frame(f);
        frames.push_back(f);
    }

    return frames;
}


// ---------------------------------------------------------------------------
// SMSResidual
// ---------------------------------------------------------------------------

SMSResidual::SMSResidual() {
    sms_init();

    sms_initResidualParams(&_residual_params);
    _residual_params.hopSize = _hop_size;
    sms_initResidual(&_residual_params);

    _temp_synth = new sample[_hop_size];

    _pd.hop_size(_hop_size);
    _pd.realtime(1);
    _synth.hop_size(_hop_size);
    _synth.det_synthesis_type(SMS_STYPE_DET);
}

SMSResidual::~SMSResidual() {
    sms_freeResidual(&_residual_params);
    sms_free();

    if(_temp_synth) {
        delete[] _temp_synth;
    }
    _temp_synth = NULL;
}

void SMSResidual::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;

    sms_freeResidual(&_residual_params);
    _residual_params.hopSize = _hop_size;
    sms_initResidual(&_residual_params);

    if(_temp_synth) {
        delete[] _temp_synth;
    }
    _temp_synth = new sample[_hop_size];
}

int SMSResidual::num_stochastic_coeffs() {
    return _residual_params.nCoeffs;
}

void SMSResidual::num_stochastic_coeffs(int new_num_stochastic_coeffs) {
    sms_freeResidual(&_residual_params);
    _residual_params.nCoeffs = new_num_stochastic_coeffs;
    sms_initResidual(&_residual_params);
}

void SMSResidual::residual_frame(Frame* frame) {
    frame->clear_peaks();
    frame->clear_partials();
    frame->clear_synth();

    _pd.find_peaks_in_frame(frame);
    _pt.update_partials(frame);
    _synth.synth_frame(frame);

    sms_findResidual(frame->size(), frame->synth(),
                     frame->size(), frame->audio(),
                     &_residual_params);

    for(int i = 0; i < frame->size(); i++) {
        frame->residual()[i] = _residual_params.residual[i];
    }
}

// Calculate and return one frame of the synthesised residual signal
void SMSResidual::synth_frame(Frame* frame) {
    residual_frame(frame);
    sms_approxResidual(_hop_size, frame->residual(),
                       _hop_size, frame->synth_residual(),
                       &_residual_params);

    // SMS stochastic component is currently a bit loud so scaled here
    for(int i = 0; i < frame->synth_size(); i++) {
        frame->synth_residual()[i] *= 0.2;
    }
}
