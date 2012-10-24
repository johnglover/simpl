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
//	TestMQPartialTracking
// ---------------------------------------------------------------------------
class TestMQPartialTracking : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestMQPartialTracking);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST(test_peaks);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    MQPeakDetection* pd;
    MQPartialTracking* pt;
    SndfileHandle sf;
    int num_samples;

    void test_basic() {
        pt->reset();
        pd->hop_size(256);
        pd->frame_size(2048);

        sample* audio = new sample[(int)sf.frames()];
        sf.read(audio, (int)sf.frames());

        Frames frames = pd->find_peaks(
            num_samples, &(audio[(int)sf.frames() / 2])
        );
        frames = pt->find_partials(frames);

        for(int i = 0; i < frames.size(); i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
            CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
        }
    }

    void test_peaks() {
        pt->reset();

        Frames frames;
        Peaks peaks;
        int num_frames = 8;

        for(int i = 0; i < num_frames; i++) {
            Peak* p = new Peak();
            p->amplitude = 0.2;
            p->frequency = 220;

            Peak* p2 = new Peak();
            p2->amplitude = 0.2;
            p2->frequency = 440;

            Frame* f = new Frame();
            f->add_peak(p);
            f->add_peak(p2);

            frames.push_back(f);
            peaks.push_back(p);
            peaks.push_back(p2);
        }

        pt->find_partials(frames);
        for(int i = 0; i < num_frames; i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
            CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
            CPPUNIT_ASSERT(frames[i]->partial(0)->amplitude == 0.2);
            CPPUNIT_ASSERT(frames[i]->partial(0)->frequency == 220);
            CPPUNIT_ASSERT(frames[i]->partial(1)->amplitude == 0.2);
            CPPUNIT_ASSERT(frames[i]->partial(1)->frequency == 440);
        }

        for(int i = 0; i < num_frames * 2; i++) {
            delete peaks[i];
        }

        for(int i = 0; i < num_frames; i++) {
            delete frames[i];
        }
    }

public:
    void setUp() {
        pd = new MQPeakDetection();
        pt = new MQPartialTracking();
        sf = SndfileHandle("../tests/audio/flute.wav");
        num_samples = 4096;
    }

    void tearDown() {
        delete pd;
        delete pt;
    }
};


// ---------------------------------------------------------------------------
//	TestSMSPartialTracking
// ---------------------------------------------------------------------------
class TestSMSPartialTracking : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestSMSPartialTracking);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST(test_peaks);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    SMSPeakDetection* pd;
    SMSPartialTracking* pt;
    SndfileHandle sf;
    int num_samples;

    void test_basic() {
        pt->reset();
        pd->hop_size(256);
        pd->frame_size(2048);

        sample* audio = new sample[(int)sf.frames()];
        sf.read(audio, (int)sf.frames());

        Frames frames = pd->find_peaks(
            num_samples, &(audio[(int)sf.frames() / 2])
        );
        frames = pt->find_partials(frames);

        for(int i = 0; i < frames.size(); i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
            CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
        }
    }

    void test_peaks() {
        pt->reset();

        Frames frames;
        Peaks peaks;
        int num_frames = 8;

        for(int i = 0; i < num_frames; i++) {
            Peak* p = new Peak();
            p->amplitude = 0.2;
            p->frequency = 220;

            Peak* p2 = new Peak();
            p2->amplitude = 0.2;
            p2->frequency = 440;

            Frame* f = new Frame();
            f->add_peak(p);
            f->add_peak(p2);

            frames.push_back(f);
            peaks.push_back(p);
            peaks.push_back(p2);
        }

        pt->find_partials(frames);
        for(int i = 0; i < num_frames; i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
            CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
            CPPUNIT_ASSERT(frames[i]->partial(0)->amplitude == 0.2);
            CPPUNIT_ASSERT(frames[i]->partial(0)->frequency == 220);
            CPPUNIT_ASSERT(frames[i]->partial(1)->amplitude == 0.2);
            CPPUNIT_ASSERT(frames[i]->partial(1)->frequency == 440);
        }

        for(int i = 0; i < num_frames * 2; i++) {
            delete peaks[i];
        }

        for(int i = 0; i < num_frames; i++) {
            delete frames[i];
        }
    }

public:
    void setUp() {
        pd = new SMSPeakDetection();
        pt = new SMSPartialTracking();
        pt->realtime(1);
        sf = SndfileHandle("../tests/audio/flute.wav");
        num_samples = 4096;
    }

    void tearDown() {
        delete pd;
        delete pt;
    }
};

// ---------------------------------------------------------------------------
//	TestLorisPartialTracking
// ---------------------------------------------------------------------------
class TestLorisPartialTracking : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestLorisPartialTracking);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST(test_peaks);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    LorisPeakDetection* pd;
    LorisPartialTracking* pt;
    SndfileHandle sf;
    int num_samples;

    void test_basic() {
        pt->reset();
        pd->hop_size(256);
        pd->frame_size(2048);

        sample* audio = new sample[(int)sf.frames()];
        sf.read(audio, (int)sf.frames());

        Frames frames = pd->find_peaks(
            num_samples, &(audio[(int)sf.frames() / 2])
        );
        frames = pt->find_partials(frames);

        for(int i = 0; i < frames.size(); i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
            CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
        }
    }

    void test_peaks() {
        pt->reset();

        Frames frames;
        Peaks peaks;
        int num_frames = 8;

        for(int i = 0; i < num_frames; i++) {
            Peak* p = new Peak();
            p->amplitude = 0.2;
            p->frequency = 220;

            Peak* p2 = new Peak();
            p2->amplitude = 0.2;
            p2->frequency = 440;

            Frame* f = new Frame();
            f->add_peak(p);
            f->add_peak(p2);

            frames.push_back(f);
            peaks.push_back(p);
            peaks.push_back(p2);
        }

        pt->find_partials(frames);
        for(int i = 0; i < num_frames; i++) {
            CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
            CPPUNIT_ASSERT(frames[i]->num_partials() > 0);
            CPPUNIT_ASSERT(frames[i]->partial(0)->amplitude == 0.2);
            CPPUNIT_ASSERT(frames[i]->partial(0)->frequency == 220);
            CPPUNIT_ASSERT(frames[i]->partial(1)->amplitude == 0.2);
            CPPUNIT_ASSERT(frames[i]->partial(1)->frequency == 440);
        }

        for(int i = 0; i < num_frames * 2; i++) {
            delete peaks[i];
        }

        for(int i = 0; i < num_frames; i++) {
            delete frames[i];
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

CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestMQPartialTracking);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestSMSPartialTracking);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestLorisPartialTracking);

int main(int arg, char **argv) {
    CppUnit::TextTestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    return runner.run("", false);
}
