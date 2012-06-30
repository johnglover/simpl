#include "partial_tracking.h"

using namespace std;
using namespace simpl;


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
