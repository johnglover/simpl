#include "synthesis.h"

using namespace std;
using namespace simpl;


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
