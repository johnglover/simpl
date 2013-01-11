#include "test_synthesis.h"

using namespace simpl;

// ---------------------------------------------------------------------------
//	TestMQSynthesis
// ---------------------------------------------------------------------------
void TestMQSynthesis::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }
}

void TestMQSynthesis::test_basic() {
    int num_samples = 4096;

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    _pd.clear();
    _pt.reset();
    _synth.reset();

    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    frames = _pt.find_partials(frames);
    frames = _synth.synth(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);

        double energy = 0.f;
        for(int j = 0; j < _synth.hop_size(); j++) {
            energy += frames[i]->synth()[j] * frames[i]->synth()[j];
        }
        CPPUNIT_ASSERT(energy > 0.f);
    }
}


// ---------------------------------------------------------------------------
//	TestLorisSynthesis
// ---------------------------------------------------------------------------
void TestLorisSynthesis::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }
}

void TestLorisSynthesis::test_basic() {
    int num_samples = 4096;

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    _pd.clear();
    _pt.reset();
    _synth.reset();

    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    frames = _pt.find_partials(frames);
    frames = _synth.synth(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);

        double energy = 0.f;
        for(int j = 0; j < _synth.hop_size(); j++) {
            energy += frames[i]->synth()[j] * frames[i]->synth()[j];
        }
        CPPUNIT_ASSERT(energy > 0.f);
    }
}
