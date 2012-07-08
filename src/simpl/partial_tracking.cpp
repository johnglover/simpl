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
        if(frames[i]->max_partials() != _max_partials) {
            frames[i]->max_partials(_max_partials);
        }

        Peaks peaks = update_partials(frames[i]);
        for(int j = 0; j < peaks.size(); j++) {
            frames[i]->add_partial(peaks[j]);
        }
    }
    _frames = frames;
    return _frames;
}


// ---------------------------------------------------------------------------
// SMSPartialTracking
// ---------------------------------------------------------------------------

SMSPartialTracking::SMSPartialTracking() {
    sms_init();

    sms_initAnalParams(&_analysis_params);
    _analysis_params.iSamplingRate = _sampling_rate;
    _analysis_params.fHighestFreq = 20000;
    _analysis_params.iMaxDelayFrames = 4;
    _analysis_params.analDelay = 0;
    _analysis_params.minGoodFrames = 1;
    _analysis_params.iCleanTracks = 0;
    _analysis_params.iFormat = SMS_FORMAT_HP;
    _analysis_params.nTracks = _max_partials;
    _analysis_params.maxPeaks = _max_partials;
    _analysis_params.nGuides = _max_partials;
    _analysis_params.preEmphasis = 0;
    sms_initAnalysis(&_analysis_params);

    sms_fillHeader(&_header, &_analysis_params);
    sms_allocFrameH(&_header, &_data);

    _peak_amplitude = NULL;
    _peak_frequency = NULL;;
    _peak_phase = NULL;
    init_peaks();
}

SMSPartialTracking::~SMSPartialTracking() {
    sms_freeAnalysis(&_analysis_params);
    sms_freeFrame(&_data);
    sms_free();

    delete [] _peak_amplitude;
    delete [] _peak_frequency;
    delete [] _peak_phase;

    _peak_amplitude = NULL;
    _peak_frequency = NULL;;
    _peak_phase = NULL;
}

void SMSPartialTracking::init_peaks() {
    if(_peak_amplitude) {
        delete [] _peak_amplitude;
    }
    if(_peak_frequency) {
        delete [] _peak_frequency;
    }
    if(_peak_phase) {
        delete [] _peak_phase;
    }

    _peak_amplitude = new sample[_max_partials];
    _peak_frequency = new sample[_max_partials];
    _peak_phase = new sample[_max_partials];

    memset(_peak_amplitude, 0.0, sizeof(sample) * _max_partials);
    memset(_peak_frequency, 0.0, sizeof(sample) * _max_partials);
    memset(_peak_phase, 0.0, sizeof(sample) * _max_partials);
}

void SMSPartialTracking::max_partials(int new_max_partials) {
    _max_partials = new_max_partials;

    sms_freeAnalysis(&_analysis_params);
    sms_freeFrame(&_data);

    _analysis_params.maxPeaks = _max_partials;
    _analysis_params.nTracks = _max_partials;
    _analysis_params.nGuides = _max_partials;

    sms_initAnalysis(&_analysis_params);
    sms_fillHeader(&_header, &_analysis_params);
    sms_allocFrameH(&_header, &_data);

    init_peaks();
}

Peaks SMSPartialTracking::update_partials(Frame* frame) {
    int num_peaks = _max_partials;
    if(num_peaks > frame->num_peaks()) {
        num_peaks = frame->num_peaks();
    }

    // set peaks in SMSAnalysisParams object
    for(int i = 0; i < num_peaks; i++) {
        _peak_amplitude[i] = frame->peak(i)->amplitude;
        _peak_frequency[i] = frame->peak(i)->frequency;
        _peak_phase[i] = frame->peak(i)->phase;
    }

    sms_setPeaks(&_analysis_params,
                 _max_partials, _peak_amplitude,
                 _max_partials, _peak_frequency,
                 _max_partials, _peak_phase);

    // SMS partial tracking
    sms_findPartials(&_data, &_analysis_params);

    // get partials from SMSData object
    Peaks peaks;

    for(int i = 0; i < _data.nTracks; i++) {
        Peak* p = new Peak();
        p->amplitude = _data.pFSinAmp[i];
        p->frequency = _data.pFSinFreq[i];
        p->phase = _data.pFSinPha[i];
        peaks.push_back(p);
    }

    return peaks;
}
