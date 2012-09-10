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

namespace simpl
{

// ---------------------------------------------------------------------------
//	TestLorisPartialTracking
// ---------------------------------------------------------------------------
class TestLorisPartialTracking : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestLorisPartialTracking);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    LorisPeakDetection* pd;
    LorisPartialTracking* pt;
    SndfileHandle sf;
    int num_samples;

    void test_basic() {
        sample* audio = new sample[(int)sf.frames()];
        sf.read(audio, (int)sf.frames());
        Frames frames = pd->find_peaks(num_samples, &(audio[(int)sf.frames() / 2]));
        frames = pt->find_partials(frames);

        for(int i = 0; i < frames.size(); i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
            CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
        }
    }

public:
    void setUp() {
        pd = new LorisPeakDetection();
        pt = new LorisPartialTracking();
        sf = SndfileHandle("../tests/audio/flute.wav");
        num_samples = 4096;
    }

    void tearDown() {
        delete pd;
        delete pt;
    }
};

} // end of namespace simpl

CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestLorisPartialTracking);

int main(int arg, char **argv) {
    CppUnit::TextTestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    return runner.run("", false);
}