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
