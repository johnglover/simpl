#include "peak_detection.h"

using namespace std;
using namespace simpl;


// ---------------------------------------------------------------------------
// PeakDetection
// ---------------------------------------------------------------------------

PeakDetection::PeakDetection() {
    _sampling_rate = 44100;
    _frame_size = 2048;
    _static_frame_size = true;
    _hop_size = 512;
    _max_peaks = 100;
    _window_type = "hamming";
    _window_size = 2048;
    _min_peak_separation = 1.0; // in Hz
}

PeakDetection::~PeakDetection() {
    clear();
}

void PeakDetection::clear() {
    for(int i = 0; i < _frames.size(); i++) {
        if(_frames[i]) {
            delete _frames[i];
            _frames[i] = NULL;
        }
    }

    _frames.clear();
}

int PeakDetection::sampling_rate() {
    return _sampling_rate;
}

void PeakDetection::sampling_rate(int new_sampling_rate) {
    _sampling_rate = new_sampling_rate;
}

int PeakDetection::frame_size() {
    return _frame_size;
}

void PeakDetection::frame_size(int new_frame_size) {
    _frame_size = new_frame_size;
}

bool PeakDetection::static_frame_size() {
    return _static_frame_size;
}

void PeakDetection::static_frame_size(bool new_static_frame_size) {
    _static_frame_size = new_static_frame_size;
}

int PeakDetection::next_frame_size() {
    return _frame_size;
}

int PeakDetection::hop_size() {
    return _hop_size;
}

void PeakDetection::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
}

int PeakDetection::max_peaks() {
    return _max_peaks;
}

void PeakDetection::max_peaks(int new_max_peaks) {
    _max_peaks = new_max_peaks;
}

std::string PeakDetection::window_type() {
    return _window_type;
}

void PeakDetection::window_type(std::string new_window_type) {
    _window_type = new_window_type;
}

int PeakDetection::window_size() {
    return _window_size;
}

void PeakDetection::window_size(int new_window_size) {
    _window_size = new_window_size;
}

simpl_sample PeakDetection::min_peak_separation() {
    return _min_peak_separation;
}

void PeakDetection::min_peak_separation(simpl_sample new_min_peak_separation) {
    _min_peak_separation = new_min_peak_separation;
}

int PeakDetection::num_frames() {
    return _frames.size();
}

Frame* PeakDetection::frame(int frame_number) {
    return _frames[frame_number];
}

Frames PeakDetection::frames() {
    return _frames;
}

void PeakDetection::frames(Frames new_frames) {
    _frames = new_frames;
}

// Find and return all spectral peaks in a given frame of audio
void PeakDetection::find_peaks_in_frame(Frame* frame) {
}

// Find and return all spectral peaks in a given audio signal.
// If the signal contains more than 1 frame worth of audio, it will be broken
// up into separate frames, each containing a std::vector of peaks.
// Frames* PeakDetection::find_peaks(const samples& audio)
Frames PeakDetection::find_peaks(int audio_size, simpl_sample* audio) {
    clear();
    unsigned int pos = 0;
    bool alloc_memory_in_frame = true;

    while(pos <= audio_size - _hop_size) {
        if(!_static_frame_size) {
            _frame_size = next_frame_size();
        }

        Frame* f = new Frame(_frame_size, alloc_memory_in_frame);
        f->max_peaks(_max_peaks);

        if((int)pos <= (audio_size - _frame_size)) {
            f->audio(&(audio[pos]), _frame_size);
        }
        else {
            f->audio(&(audio[pos]), audio_size - pos);
        }

        find_peaks_in_frame(f);
        _frames.push_back(f);
        pos += _hop_size;
    }

    return _frames;
}


// ---------------------------------------------------------------------------
// MQPeakDetection
// ---------------------------------------------------------------------------
MQPeakDetection::MQPeakDetection() {
    _mq_params.max_peaks = _max_peaks;
    _mq_params.frame_size = _frame_size;
    _mq_params.num_bins = (_frame_size / 2) + 1;
    _mq_params.peak_threshold = 0.0;
    _mq_params.matching_interval = 100.0;
    _mq_params.fundamental = 44100.0 / _frame_size;
    init_mq(&_mq_params);
}

MQPeakDetection::~MQPeakDetection() {
    destroy_mq(&_mq_params);
}

void MQPeakDetection::reset() {
    reset_mq(&_mq_params);
    destroy_mq(&_mq_params);
    _mq_params.max_peaks = _max_peaks;
    _mq_params.frame_size = _frame_size;
    _mq_params.num_bins = (_frame_size / 2) + 1;
    _mq_params.fundamental = 44100.0 / _frame_size;
    init_mq(&_mq_params);
}

void MQPeakDetection::frame_size(int new_frame_size) {
    _frame_size = new_frame_size;
    reset();
}

void MQPeakDetection::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
}

void MQPeakDetection::max_peaks(int new_max_peaks) {
    _max_peaks = new_max_peaks;
    reset();
}

void MQPeakDetection::find_peaks_in_frame(Frame* frame) {
    MQPeakList* pl = mq_find_peaks(_frame_size, frame->audio(), &_mq_params);
    MQPeakList* pl_start = pl;

    int num_peaks = 0;
    while(pl && pl->peak && num_peaks < _max_peaks) {
        frame->add_peak(pl->peak->amplitude,
                        pl->peak->frequency,
                        pl->peak->phase,
                        0.0);
        pl = pl->next;
        num_peaks++;
    }

    delete_peak_list(pl_start);
}

// ---------------------------------------------------------------------------
// SMSPeakDetection
// ---------------------------------------------------------------------------
SMSPeakDetection::SMSPeakDetection() {
    sms_init();

    sms_initAnalParams(&_analysis_params);
    _analysis_params.iSamplingRate = _sampling_rate;
    _analysis_params.iFrameRate = _sampling_rate / _hop_size;
    _analysis_params.iWindowType = SMS_WIN_HAMMING;
    _analysis_params.fHighestFreq = 20000;
    _analysis_params.iMaxDelayFrames = 4;
    _analysis_params.analDelay = 0;
    _analysis_params.minGoodFrames = 1;
    _analysis_params.iCleanTracks = 0;
    _analysis_params.iFormat = SMS_FORMAT_HP;
    _analysis_params.nTracks = _max_peaks;
    _analysis_params.maxPeaks = _max_peaks;
    _analysis_params.nGuides = _max_peaks;
    _analysis_params.preEmphasis = 0;
    _analysis_params.realtime = 0;
    sms_initAnalysis(&_analysis_params);
    _analysis_params.iSizeSound = _frame_size;

    sms_initSpectralPeaks(&_peaks, _max_peaks);

    // By default, SMS will change the size of the frames being read
    // depending on the detected fundamental frequency (if any) of the
    // input sound. To prevent this behaviour (useful when comparing
    // different analysis algorithms), set the
    // _static_frame_size variable to True
    _static_frame_size = false;
}

SMSPeakDetection::~SMSPeakDetection() {
    sms_freeAnalysis(&_analysis_params);
    sms_freeSpectralPeaks(&_peaks);
    sms_free();
}

int SMSPeakDetection::next_frame_size() {
    return _analysis_params.sizeNextRead;
}

void SMSPeakDetection::frame_size(int new_frame_size) {
    _frame_size = new_frame_size;
    _analysis_params.iSizeSound = _hop_size;
}

void SMSPeakDetection::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
    sms_freeAnalysis(&_analysis_params);
    _analysis_params.iFrameRate = _sampling_rate / _hop_size;
    sms_initAnalysis(&_analysis_params);
}

void SMSPeakDetection::max_peaks(int new_max_peaks) {
    _max_peaks = new_max_peaks;
    if(_max_peaks > SMS_MAX_NPEAKS) {
        _max_peaks = SMS_MAX_NPEAKS;
    }

    sms_freeAnalysis(&_analysis_params);
    sms_freeSpectralPeaks(&_peaks);

    _analysis_params.nTracks = _max_peaks;
    _analysis_params.maxPeaks = _max_peaks;
    _analysis_params.nGuides = _max_peaks;

    sms_initAnalysis(&_analysis_params);
    sms_initSpectralPeaks(&_peaks, _max_peaks);
}

int SMSPeakDetection::realtime() {
    return _analysis_params.realtime;
}

void SMSPeakDetection::realtime(int new_realtime) {
    _analysis_params.realtime = new_realtime;
}

// Find and return all spectral peaks in a given frame of audio
void SMSPeakDetection::find_peaks_in_frame(Frame* frame) {
    int num_peaks = sms_findPeaks(frame->size(), frame->audio(),
                                  &_analysis_params, &_peaks);

    for(int i = 0; i < num_peaks; i++) {
        frame->add_peak(_peaks.pSpectralPeaks[i].fMag,
                        _peaks.pSpectralPeaks[i].fFreq,
                        _peaks.pSpectralPeaks[i].fPhase,
                        0.0);
    }
}

// Find and return all spectral peaks in a given audio signal.
// If the signal contains more than 1 frame worth of audio,
// it will be broken up into separate frames, with a list of
// peaks returned for each frame.
Frames SMSPeakDetection::find_peaks(int audio_size, simpl_sample* audio) {
    clear();
    unsigned int pos = 0;
    bool alloc_memory_in_frame = true;

    _analysis_params.iSizeSound = audio_size;

    while(pos <= audio_size - _hop_size) {
        if(!_static_frame_size) {
            _frame_size = next_frame_size();
        }

        Frame* f = new Frame(_frame_size, alloc_memory_in_frame);
        f->max_peaks(_max_peaks);

        if((int)pos <= (audio_size - _frame_size)) {
            f->audio(&(audio[pos]), _frame_size);
        }
        else {
            f->audio(&(audio[pos]), audio_size - pos);
        }

        find_peaks_in_frame(f);
        _frames.push_back(f);

        if(!_static_frame_size) {
            pos += _frame_size;
        }
        else {
            pos += _hop_size;
        }
    }

    return _frames;
}


// ---------------------------------------------------------------------------
// SndObjPeakDetection
// ---------------------------------------------------------------------------
SndObjPeakDetection::SndObjPeakDetection() {
    _threshold = 0.003;
    _input = NULL;
    _window = NULL;
    _ifgram = NULL;
    _analysis = NULL;
    reset();
}

SndObjPeakDetection::~SndObjPeakDetection() {
    if(_input) {
        delete _input;
        _input = NULL;
    }
    if(_window) {
        delete _window;
        _window = NULL;
    }
    if(_ifgram) {
        delete _ifgram;
        _ifgram = NULL;
    }
    if(_analysis) {
        delete _analysis;
        _analysis = NULL;
    }
}

void SndObjPeakDetection::reset() {
    if(_input) {
        delete _input;
    }
    if(_window) {
        delete _window;
    }
    if(_ifgram) {
        delete _ifgram;
    }
    if(_analysis) {
        delete _analysis;
    }

    _input = new SndObj();
    _input->SetVectorSize(_frame_size);
    _window = new HammingTable(_frame_size, 0.5);
    _ifgram = new IFGram(_window, _input, 1, _frame_size, _hop_size);
    _analysis = new SinAnal(_ifgram, _threshold, _max_peaks);
}

void SndObjPeakDetection::frame_size(int new_frame_size) {
    _frame_size = new_frame_size;
    reset();
}

void SndObjPeakDetection::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
    reset();
}

void SndObjPeakDetection::max_peaks(int new_max_peaks) {
    _max_peaks = new_max_peaks;
    reset();
}

void SndObjPeakDetection::find_peaks_in_frame(Frame* frame) {
    _input->PushIn(frame->audio(), frame->size());
    _ifgram->DoProcess();
    int num_peaks = _analysis->FindPeaks();

    for(int i = 0; i < num_peaks; i++) {
        frame->add_peak(_analysis->Output(i * 3),
                        _analysis->Output((i * 3) + 1),
                        _analysis->Output((i * 3) + 2),
                        0.0);
    }
}


// ---------------------------------------------------------------------------
// LorisPeakDetection
// ---------------------------------------------------------------------------
SimplLorisAnalyzer::SimplLorisAnalyzer(int window_size, simpl_sample resolution,
                                       int hop_size, simpl_sample sampling_rate) :
    Loris::Analyzer(resolution, 2 * resolution) {

    buildFundamentalEnv(false);

    _window_shape = Loris::KaiserWindow::computeShape(sidelobeLevel());
    _window.resize(window_size);
    Loris::KaiserWindow::buildWindow(_window, _window_shape);

    _window_deriv.resize(window_size);
    Loris::KaiserWindow::buildTimeDerivativeWindow(_window_deriv, _window_shape);

    _spectrum = new Loris::ReassignedSpectrum(_window, _window_deriv);
    m_cropTime = 2 * hop_size;
    _peak_selector = new Loris::SpectralPeakSelector(sampling_rate, m_cropTime);

    if(m_bwAssocParam > 0) {
        _bw_associator.reset(new Loris::AssociateBandwidth(bwRegionWidth(), sampling_rate));
    }
}

SimplLorisAnalyzer::~SimplLorisAnalyzer() {
    delete _spectrum;
    delete _peak_selector;
}

void SimplLorisAnalyzer::analyze(int audio_size, simpl_sample* audio) {
    m_ampEnvBuilder->reset();
    m_f0Builder->reset();
    m_partials.clear();
    peaks.clear();

    _spectrum->transform(audio, audio + (audio_size / 2), audio + audio_size);
    peaks = _peak_selector->selectPeaks(*_spectrum, m_freqFloor);

    fixBandwidth(peaks);
    if(m_bwAssocParam > 0) {
        _bw_associator->associateBandwidth(peaks.begin(), peaks.end(), peaks.end());
    }
}

// ---------------------------------------------------------------------------

LorisPeakDetection::LorisPeakDetection() {
    _resolution = (_sampling_rate / 2) / _max_peaks;
    _analyzer = NULL;
    reset();
}

LorisPeakDetection::~LorisPeakDetection() {
    if(_analyzer) {
        delete _analyzer;
    }
}

void LorisPeakDetection::reset() {
    if(_analyzer) {
        delete _analyzer;
    }
    _analyzer = new SimplLorisAnalyzer(_frame_size, _resolution,
                                       _hop_size, _sampling_rate);
}

void LorisPeakDetection::frame_size(int new_frame_size) {
    _frame_size = new_frame_size;
    reset();
}

void LorisPeakDetection::hop_size(int new_hop_size) {
    _hop_size = new_hop_size;
    reset();
}

void LorisPeakDetection::max_peaks(int new_max_peaks) {
    _max_peaks = new_max_peaks;
    _resolution = (_sampling_rate / 2) / _max_peaks;
    reset();
}

void LorisPeakDetection::find_peaks_in_frame(Frame* frame) {
    _analyzer->analyze(frame->size(), frame->audio());

    int num_peaks = _analyzer->peaks.size();
    if(num_peaks > _max_peaks) {
        num_peaks = _max_peaks;
    }

    for(int i = 0; i < num_peaks; i++) {
        frame->add_peak(_analyzer->peaks[i].amplitude(),
                        _analyzer->peaks[i].frequency(),
                        0.0,
                        _analyzer->peaks[i].bandwidth());
    }
}
