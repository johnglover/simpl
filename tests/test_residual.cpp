#include "test_residual.h"

using namespace simpl;

// ---------------------------------------------------------------------------
//	test_basic
// ---------------------------------------------------------------------------
static void test_basic(Residual* residual, SndfileHandle *sf) {
    int num_samples = 4096;
    int hop_size = 256;
    int frame_size = 512;

    std::vector<sample> audio(sf->frames(), 0.0);
    sf->read(&audio[0], (int)sf->frames());

    residual->reset();
    residual->frame_size(frame_size);
    residual->hop_size(hop_size);

    Frames frames = residual->synth(num_samples,
                                    &(audio[(int)sf->frames() / 2]));

    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        CPPUNIT_ASSERT(frames[i]->num_partials() > 0);

        double energy = 0.f;
        for(int j = 0; j < residual->hop_size(); j++) {
            energy += frames[i]->synth_residual()[j] *
                      frames[i]->synth_residual()[j];
        }
        CPPUNIT_ASSERT(energy > 0.f);
    }
}


// ---------------------------------------------------------------------------
//	TestSMSResidual
// ---------------------------------------------------------------------------
void TestSMSResidual::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }
}

void TestSMSResidual::test_basic() {
    ::test_basic(&_res, &_sf);
}
