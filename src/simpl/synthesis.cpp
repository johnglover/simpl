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

void Synthesis::reset() {
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
        frames[i]->synth_size(_hop_size);
        synth_frame(frames[i]);
    }
    return frames;
}


// ---------------------------------------------------------------------------
// MQSynthesis
// ---------------------------------------------------------------------------
MQSynthesis::MQSynthesis() {
    _prev_amps = NULL;
    _prev_freqs = NULL;
    _prev_phases = NULL;
    reset();
}

MQSynthesis::~MQSynthesis() {
    if(_prev_amps) delete [] _prev_amps;
    if(_prev_freqs) delete [] _prev_freqs;
    if(_prev_phases) delete [] _prev_phases;

    _prev_amps = NULL;
    _prev_freqs = NULL;
    _prev_phases = NULL;
}

void MQSynthesis::reset() {
    if(_prev_amps) delete [] _prev_amps;
    if(_prev_freqs) delete [] _prev_freqs;
    if(_prev_phases) delete [] _prev_phases;

    _prev_amps = new sample[_max_partials];
    _prev_freqs = new sample[_max_partials];
    _prev_phases = new sample[_max_partials];

    memset(_prev_amps, 0.0, sizeof(sample) * _max_partials);
    memset(_prev_freqs, 0.0, sizeof(sample) * _max_partials);
    memset(_prev_phases, 0.0, sizeof(sample) * _max_partials);
}

sample MQSynthesis::hz_to_radians(sample f) {
    return (f * 2 * M_PI) / _sampling_rate;
}

void MQSynthesis::max_partials(int new_max_partials) {
    _max_partials = new_max_partials;
    reset();
}

void MQSynthesis::synth_frame(Frame* frame) {
    int num_partials = frame->num_partials();
    if(num_partials > _max_partials) {
        num_partials = _max_partials;
    }

    for(int n = 0; n < _hop_size; n++) {
        frame->synth()[n] = 0.f;
    }

    for(int i = 0; i < num_partials; i++) {
        sample amp = frame->partial(i)->amplitude;
        sample freq = hz_to_radians(frame->partial(i)->frequency);
        sample phase = frame->partial(i)->phase;

        // get values for last amplitude, frequency and phase
        // these are the initial values of the instantaneous
        // amplitude/frequency/phase
        sample prev_amp = _prev_amps[i];
        sample prev_freq = _prev_freqs[i];
        sample prev_phase = _prev_phases[i];

        if(prev_amp == 0) {
            prev_freq = freq;
            prev_phase = frame->partial(i)->phase - (freq * _hop_size);
            while(prev_phase >= M_PI) {
                prev_phase -= (2.0 * M_PI);
            }
            while(prev_phase < -M_PI) {
                prev_phase += (2.0 * M_PI);
            }
        }

        // amplitudes are linearly interpolated between frames
        sample inst_amp = prev_amp;
        sample amp_inc = (frame->partial(i)->amplitude - prev_amp) / _hop_size;

        // freqs/phases are calculated by cubic interpolation
        sample freq_diff = freq - prev_freq;
        sample x = (prev_phase + (prev_freq * _hop_size) - phase) +
                   (freq_diff * (_hop_size / 2.0));
        x /= (2.0 * M_PI);
        int m = floor(x + 0.5);
        sample phase_diff = phase - prev_phase - (prev_freq * _hop_size) +
                            (2.0 * M_PI * m);
        sample alpha = ((3.0 / pow(_hop_size, 2.0)) * phase_diff) -
                       (freq_diff / _hop_size);
        sample beta = ((-2.0 / pow(_hop_size, 3.0)) * phase_diff) +
                      (freq_diff / pow(_hop_size, 2.0));

        // calculate output samples
        sample inst_phase = 0.f;
        for(int n = 0; n < _hop_size; n++) {
            inst_amp += amp_inc;
            inst_phase = prev_phase + (prev_freq * n) +
                         (alpha * pow((sample)n, 2.0)) +
                         (beta * pow((sample)n, 3.0));
            frame->synth()[n] += (2.f * inst_amp) * cos(inst_phase);
        }

        _prev_amps[i] = amp;
        _prev_freqs[i] = freq;
        _prev_phases[i] = phase;
    }
}

// ---------------------------------------------------------------------------
// SMSSynthesis
// ---------------------------------------------------------------------------

SMSSynthesis::SMSSynthesis() {
    sms_init();

    sms_initSynthParams(&_synth_params);
    _synth_params.iSamplingRate = _sampling_rate;
    _synth_params.iDetSynthType = SMS_DET_SIN;
    _synth_params.iSynthesisType = SMS_STYPE_DET;
    _synth_params.iStochasticType = SMS_STOC_NONE;
    _synth_params.sizeHop = _hop_size;
    _synth_params.nTracks = _max_partials;
    _synth_params.deEmphasis = 0;
    sms_initSynth(&_synth_params);

    sms_allocFrame(&_data, _max_partials,
                   num_stochastic_coeffs(), 1,
                   stochastic_type(), 0);
}

SMSSynthesis::~SMSSynthesis() {
    sms_freeSynth(&_synth_params);
    sms_freeFrame(&_data);
    sms_free();
}

void SMSSynthesis::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;

    sms_freeSynth(&_synth_params);
    _synth_params.sizeHop = _hop_size;
    sms_initSynth(&_synth_params);
}

void SMSSynthesis::max_partials(int new_max_partials) {
    _max_partials = new_max_partials;

    sms_freeSynth(&_synth_params);
    sms_freeFrame(&_data);
    _synth_params.nTracks = _max_partials;
    sms_initSynth(&_synth_params);
    sms_allocFrame(&_data, _max_partials,
                   num_stochastic_coeffs(), 1,
                   stochastic_type(), 0);
}

int SMSSynthesis::num_stochastic_coeffs() {
    return _synth_params.nStochasticCoeff;
}

int SMSSynthesis::stochastic_type() {
    return _synth_params.iStochasticType;
}

int SMSSynthesis::det_synthesis_type() {
    return _synth_params.iDetSynthType;
}

void SMSSynthesis::det_synthesis_type(int new_det_synthesis_type) {
    _synth_params.iDetSynthType = new_det_synthesis_type;
}

void SMSSynthesis::synth_frame(Frame* frame) {
    int num_partials = _data.nTracks;
    if(num_partials > frame->num_partials()) {
        num_partials = frame->num_partials();
    }

    for(int i = 0; i < num_partials; i++) {
        _data.pFSinAmp[i] = frame->partial(i)->amplitude;
        _data.pFSinFreq[i] = frame->partial(i)->frequency;
        _data.pFSinPha[i] = frame->partial(i)->phase;
    }

    sms_synthesize(&_data, frame->synth(), &_synth_params);
}


// ---------------------------------------------------------------------------
// SndObjSynthesis
// ---------------------------------------------------------------------------
SimplSndObjAnalysisWrapper::SimplSndObjAnalysisWrapper(int max_partials) {
    partials.resize(max_partials);
}

SimplSndObjAnalysisWrapper::~SimplSndObjAnalysisWrapper() {
    partials.clear();
}

int SimplSndObjAnalysisWrapper::GetTrackID(int track) {
    if(track < partials.size()) {
        return track;
    }
    return 0;
}

int SimplSndObjAnalysisWrapper::GetTracks() {
    return partials.size();
}

double SimplSndObjAnalysisWrapper::Output(int pos) {
    int peak = pos / 3;

    if(peak > partials.size()) {
        return 0.0;
    }

    int data_field = pos % 3;

    if(partials[peak]) {
        if(data_field == 0) {
            return partials[peak]->amplitude;
        }
        else if(data_field == 1) {
            return partials[peak]->frequency;
        }
        return partials[peak]->phase;
    }

    return 0.0;
}

SndObjSynthesis::SndObjSynthesis() {
    _analysis = NULL;
    _table = NULL;
    _synth = NULL;
    reset();
}

SndObjSynthesis::~SndObjSynthesis() {
    if(_analysis) {
        delete _analysis;
    }
    if(_table) {
        delete _table;
    }
    if(_synth) {
        delete _synth;
    }

    _analysis = NULL;
    _table = NULL;
    _synth = NULL;
}

void SndObjSynthesis::reset() {
    if(_analysis) {
        delete _analysis;
    }
    if(_table) {
        delete _table;
    }
    if(_synth) {
        delete _synth;
    }

    _analysis = new SimplSndObjAnalysisWrapper(_max_partials);
    _table = new HarmTable(10000, 1, 1, 0.25);
    _synth = new SimplAdSyn(_analysis, _max_partials, _table, 1, 1, _frame_size);
}

void SndObjSynthesis::frame_size(int new_frame_size) {
    _frame_size = new_frame_size;
    reset();
}

void SndObjSynthesis::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
    reset();
}

void SndObjSynthesis::max_partials(int new_max_partials) {
    _max_partials = new_max_partials;
    reset();
}

void SndObjSynthesis::synth_frame(Frame* frame) {
    int num_partials = _max_partials;
    if(frame->num_partials() < _max_partials) {
        num_partials = frame->num_partials();
    }

    for(int i = 0; i < num_partials; i++) {
        _analysis->partials[i] = frame->partial(i);
    }
    for(int i = num_partials; i < _max_partials; i++) {
        _analysis->partials[i] = NULL;
    }

    _synth->DoProcess();

    for(int i = 0; i < _frame_size; i++) {
        frame->synth()[i] = _synth->Output(i);
    }
}


// ---------------------------------------------------------------------------
// LorisSynthesis
// ---------------------------------------------------------------------------
LorisSynthesis::LorisSynthesis() {
    _bandwidth = 1.0;
    reset();
}

LorisSynthesis::~LorisSynthesis() {
}

void LorisSynthesis::reset() {
    _oscs.clear();
    _oscs.resize(_max_partials);
    for(int i = 0; i < _max_partials; i++) {
        _oscs.push_back(Loris::Oscillator());
    }
}

void LorisSynthesis::max_partials(int new_max_partials) {
    _max_partials = new_max_partials;
    reset();
}

sample LorisSynthesis::bandwidth() {
    return _bandwidth;
}

void LorisSynthesis::bandwidth(sample new_bandwidth) {
    _bandwidth = new_bandwidth;
}

void LorisSynthesis::synth_frame(Frame* frame) {
    int num_partials = frame->num_partials();
    if(num_partials > _max_partials) {
        num_partials = _max_partials;
    }

    for(int i = 0; i < num_partials; i++) {
        Loris::Breakpoint bp = Loris::Breakpoint(
            frame->partial(i)->frequency,
            frame->partial(i)->amplitude,
            frame->partial(i)->bandwidth * _bandwidth,
            frame->partial(i)->phase
        );
        _oscs[i].oscillate(frame->synth(), frame->synth() + _hop_size,
                           bp, _sampling_rate);
    }
}
