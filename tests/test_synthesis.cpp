#include "test_synthesis.h"

using namespace simpl;

// ---------------------------------------------------------------------------
//	test_basic
// ---------------------------------------------------------------------------
static void test_basic(PeakDetection *pd, PartialTracking* pt,
                       Synthesis* synth, SndfileHandle *sf) {
    int num_samples = 4096;

    std::vector<sample> audio(sf->frames(), 0.0);
    sf->read(&audio[0], (int)sf->frames());

    pd->clear();
    pt->reset();
    synth->reset();

    Frames frames = pd->find_peaks(num_samples,
                                   &(audio[(int)sf->frames() / 2]));
    frames = pt->find_partials(frames);
    frames = synth->synth(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);

        double energy = 0.f;
        for(int j = 0; j < synth->hop_size(); j++) {
            energy += frames[i]->synth()[j] * frames[i]->synth()[j];
        }
        CPPUNIT_ASSERT(energy > 0.f);
    }
}

// ---------------------------------------------------------------------------
//	test_changing_frame_size
// ---------------------------------------------------------------------------
static void test_changing_frame_size(PeakDetection *pd, PartialTracking* pt,
                                     Synthesis* synth, SndfileHandle *sf) {
    int num_samples = 4096;
    int frame_size = 256;
    int hop_size = 128;

    std::vector<sample> audio(sf->frames(), 0.0);
    sf->read(&audio[0], (int)sf->frames());

    pd->clear();
    pt->reset();
    synth->reset();

    pd->frame_size(frame_size);
    pd->hop_size(hop_size);
    synth->frame_size(frame_size);
    synth->hop_size(hop_size);

    Frames frames = pd->find_peaks(num_samples,
                                   &(audio[(int)sf->frames() / 2]));
    frames = pt->find_partials(frames);
    frames = synth->synth(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);

        double energy = 0.f;
        for(int j = 0; j < synth->hop_size(); j++) {
            energy += frames[i]->synth()[j] * frames[i]->synth()[j];
        }
        CPPUNIT_ASSERT(energy > 0.f);
    }

    frame_size = 128;
    hop_size = 64;

    pd->clear();
    pt->reset();
    synth->reset();

    pd->frame_size(frame_size);
    pd->hop_size(hop_size);
    synth->frame_size(frame_size);
    synth->hop_size(hop_size);

    frames = pd->find_peaks(num_samples, &(audio[(int)sf->frames() / 2]));
    frames = pt->find_partials(frames);
    frames = synth->synth(frames);

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);

        double energy = 0.f;
        for(int j = 0; j < synth->hop_size(); j++) {
            energy += frames[i]->synth()[j] * frames[i]->synth()[j];
        }
        CPPUNIT_ASSERT(energy > 0.f);
    }
}

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
    ::test_basic(&_pd, &_pt, &_synth, &_sf);
}

void TestMQSynthesis::test_changing_frame_size() {
    ::test_changing_frame_size(&_pd, &_pt, &_synth, &_sf);
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
    ::test_basic(&_pd, &_pt, &_synth, &_sf);
}
