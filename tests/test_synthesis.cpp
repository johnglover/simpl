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
#include "../src/simpl/partial_tracking.h"
#include "../src/simpl/synthesis.h"

namespace simpl
{

// ---------------------------------------------------------------------------
//	TestMQSynthesis
// ---------------------------------------------------------------------------
class TestMQSynthesis : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestMQSynthesis);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    MQPeakDetection* pd;
    MQPartialTracking* pt;
    MQSynthesis* synth;
    SndfileHandle sf;
    int num_samples;

    void test_basic() {
        sample* audio = new sample[(int)sf.frames()];
        sf.read(audio, (int)sf.frames());
        Frames frames = pd->find_peaks(num_samples, &(audio[(int)sf.frames() / 2]));
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

public:
    void setUp() {
        pd = new MQPeakDetection();
        pt = new MQPartialTracking();
        synth = new MQSynthesis();
        sf = SndfileHandle("../tests/audio/flute.wav");
        num_samples = 4096;
    }

    void tearDown() {
        delete pd;
        delete pt;
        delete synth;
    }
};

// ---------------------------------------------------------------------------
//	TestLorisSynthesis
// ---------------------------------------------------------------------------
class TestLorisSynthesis : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestLorisSynthesis);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    LorisPeakDetection* pd;
    LorisPartialTracking* pt;
    LorisSynthesis* synth;
    SndfileHandle sf;
    int num_samples;

    void test_basic() {
        sample* audio = new sample[(int)sf.frames()];
        sf.read(audio, (int)sf.frames());
        Frames frames = pd->find_peaks(num_samples, &(audio[(int)sf.frames() / 2]));
        frames = pt->find_partials(frames);
        frames = synth->synth(frames);

        for(int i = 0; i < frames.size(); i++) {
            // if Loris thinPeaks is used, final frame will have no peaks
            // so don't check it
            if(i < frames.size() - 1) {
                CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
                CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
            }

            double energy = 0.f;
            for(int j = 0; j < synth->hop_size(); j++) {
                energy += frames[i]->synth()[j] * frames[i]->synth()[j];
            }
            CPPUNIT_ASSERT(energy > 0.f);
        }
    }

public:
    void setUp() {
        pd = new LorisPeakDetection();
        pt = new LorisPartialTracking();
        synth = new LorisSynthesis();
        sf = SndfileHandle("../tests/audio/flute.wav");
        num_samples = 4096;
    }

    void tearDown() {
        delete pd;
        delete pt;
        delete synth;
    }
};

} // end of namespace simpl

CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestMQSynthesis);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestLorisSynthesis);

int main(int arg, char **argv) {
    CppUnit::TextTestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    return runner.run("", false);
}
