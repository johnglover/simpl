#include <iostream>
#include <cppunit/ui/text/TextTestRunner.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <sndfile.hh>

#include "../src/simpl/base.h"
#include "../src/simpl/peak_detection.h"

namespace simpl
{

// ---------------------------------------------------------------------------
//	TestLorisPeakDetection
// ---------------------------------------------------------------------------
class TestLorisPeakDetection : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestLorisPeakDetection);
    CPPUNIT_TEST(test_find_peaks_in_frame_basic);
    CPPUNIT_TEST(test_find_peaks_basic);
    CPPUNIT_TEST(test_find_peaks_change_hop_frame_size);
    CPPUNIT_TEST(test_find_peaks_audio);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    LorisPeakDetection* pd;
    SndfileHandle sf;
    int num_samples;

    void test_find_peaks_in_frame_basic() {
        pd->clear();
        pd->frame_size(2048);

        Frame* f = new Frame(2048, true);
        Peaks p = pd->find_peaks_in_frame(f);
        CPPUNIT_ASSERT(p.size() == 0);

        delete f;
        pd->clear();
    }

    void test_find_peaks_basic() {
        sample* audio = new sample[1024];
        pd->frame_size(512);

        Frames frames = pd->find_peaks(1024, audio);
        CPPUNIT_ASSERT(frames.size() == 2);
        for(int i = 0; i < frames.size(); i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() == 0);
        }

        delete audio;
    }

    void test_find_peaks_change_hop_frame_size() {
        sample* audio = new sample[1024];
        pd->frame_size(256);
        pd->hop_size(256);

        Frames frames = pd->find_peaks(1024, audio);
        CPPUNIT_ASSERT(frames.size() == 4);
        for(int i = 0; i < frames.size(); i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() == 0);
        }

        delete audio;
    }

    void test_find_peaks_audio() {
        sample* audio = new sample[(int)sf.frames()];
        sf.read(audio, (int)sf.frames());

        Frames frames = pd->find_peaks(num_samples, &(audio[(int)sf.frames() / 2]));
        for(int i = 0; i < frames.size(); i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
        }

        delete audio;
    }

public:
    void setUp() {
        pd = new LorisPeakDetection();
        sf = SndfileHandle("../tests/audio/flute.wav");
        num_samples = 4096;
    }

    void tearDown() {
        delete pd;
    }
};

} // end of namespace simpl

CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestLorisPeakDetection);

int main(int arg, char **argv) {
    CppUnit::TextTestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    return runner.run("", false);
}
