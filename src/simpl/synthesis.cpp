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
        sample* synth_audio = new sample[_frame_size];
        memset(synth_audio, 0.0, sizeof(sample) * _frame_size);
        frames[i]->synth(synth_audio);
        frames[i]->synth_size(_frame_size);
        synth_frame(frames[i]);
    }
    return frames;
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

    for(int i = 0; i < _hop_size; i++) {
        frame->synth()[i] = _synth->Output(i);
    }
}
