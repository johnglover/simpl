#include <iostream>
#include <cppunit/ui/text/TextTestRunner.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>

#include "../src/simpl/base.h"
#include "../src/simpl/exceptions.h"

namespace Simpl 
{

// ---------------------------------------------------------------------------
//	TestPeak
// ---------------------------------------------------------------------------
class TestPeak : public CPPUNIT_NS::TestCase
{
    CPPUNIT_TEST_SUITE(TestPeak);
    CPPUNIT_TEST(test_constructor);
    CPPUNIT_TEST(test_is_start_of_partial);
    CPPUNIT_TEST(test_is_free);
    CPPUNIT_TEST(test_is_free_invalid_argument);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    Peak* peak;

    void test_constructor() 
    { 
        CPPUNIT_ASSERT_DOUBLES_EQUAL(peak->amplitude, 0.0, PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(peak->frequency, 0.0, PRECISION);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(peak->phase, 0.0, PRECISION);
        CPPUNIT_ASSERT(peak->next_peak == NULL);
        CPPUNIT_ASSERT(peak->previous_peak == NULL);
        CPPUNIT_ASSERT(peak->partial_id == 0);
        CPPUNIT_ASSERT(peak->partial_position == 0);
        CPPUNIT_ASSERT(peak->frame_number == 0);
    }

    void test_is_start_of_partial()
    {
        CPPUNIT_ASSERT(peak->is_start_of_partial());
        Peak* tmp = new Peak();
        peak->previous_peak = tmp;
        CPPUNIT_ASSERT(!peak->is_start_of_partial());
        peak->previous_peak = NULL;
        delete tmp;
    }

    void test_is_free()
    {
        peak->amplitude = 0.0;
        CPPUNIT_ASSERT(!peak->is_free());
        peak->amplitude = 1.0;
        CPPUNIT_ASSERT(peak->is_free());

        Peak* tmp = new Peak();

        peak->next_peak = tmp;
        CPPUNIT_ASSERT(!peak->is_free());
        CPPUNIT_ASSERT(!peak->is_free("forwards"));
        CPPUNIT_ASSERT(peak->is_free("backwards"));
        peak->next_peak = NULL;

        peak->previous_peak = tmp;
        CPPUNIT_ASSERT(peak->is_free());
        CPPUNIT_ASSERT(peak->is_free("forwards"));
        CPPUNIT_ASSERT(!peak->is_free("backwards"));
        peak->previous_peak = NULL;

        delete tmp;
    }

    void test_is_free_invalid_argument()
    {
        peak->amplitude = 1.0;
        CPPUNIT_ASSERT_THROW(peak->is_free("random_text"), InvalidArgument);
        peak->amplitude = 0.0;
    }

public:
    void setUp() 
    {
        peak = new Peak();
    }

    void tearDown() 
    {
        delete peak;
    } 
};

// ---------------------------------------------------------------------------
//	TestFrame
// ---------------------------------------------------------------------------
class TestFrame : public CPPUNIT_NS::TestCase
{
    CPPUNIT_TEST_SUITE(TestFrame);
    CPPUNIT_TEST(test_constructor);
    CPPUNIT_TEST(test_size);
    CPPUNIT_TEST(test_max_peaks);
    CPPUNIT_TEST(test_max_partials);
    CPPUNIT_TEST(test_add_peak);
    CPPUNIT_TEST(test_add_peaks);
    CPPUNIT_TEST(test_peak_clear);
    CPPUNIT_TEST(test_peak_iteration);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    Frame* frame;

    void test_constructor() 
    { 
        CPPUNIT_ASSERT(frame->size() == 512);
        CPPUNIT_ASSERT(frame->max_peaks() == 100);
        CPPUNIT_ASSERT(frame->num_peaks() == 0);
        CPPUNIT_ASSERT(frame->max_partials() == 100);
        CPPUNIT_ASSERT(frame->num_partials() == 0);
    }

    void test_size()
    {
        frame->size(1024);
        CPPUNIT_ASSERT(frame->size() == 1024);
        frame->size(512);
    }

    void test_max_peaks()
    {
        frame->max_peaks(200);
        CPPUNIT_ASSERT(frame->max_peaks() == 200);
        CPPUNIT_ASSERT(frame->num_peaks() == 0);
        frame->max_peaks(100);
    }

    void test_max_partials()
    {
        frame->max_partials(200);
        CPPUNIT_ASSERT(frame->max_partials() == 200);
        CPPUNIT_ASSERT(frame->num_partials() == 0);
        frame->max_partials(100);
    }

    void test_add_peak()
    {
        Peak p = Peak();
        p.amplitude = 1.5;
        frame->add_peak(p);
        CPPUNIT_ASSERT(frame->max_peaks() == 100);
        CPPUNIT_ASSERT(frame->num_peaks() == 1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5, frame->peak(0).amplitude, PRECISION);
        frame->clear_peaks();
    }

    void test_add_peaks()
    {
        Peaks* peaks = new Peaks();

        Peak p1 = Peak();
        p1.amplitude = 1.0;
        peaks->push_back(p1);

        Peak p2 = Peak();
        p2.amplitude = 2.0;
        peaks->push_back(p2);

        frame->add_peaks(peaks);
        CPPUNIT_ASSERT(frame->num_peaks() == 2);

        frame->clear_peaks();
        delete peaks;
    }

    void test_peak_clear()
    {
        Peak p = Peak();
        p.amplitude = 1.5;
        frame->add_peak(p);
        CPPUNIT_ASSERT(frame->num_peaks() == 1);
        frame->clear_peaks();
        CPPUNIT_ASSERT(frame->num_peaks() == 0);
    }

    void test_peak_iteration()
    {
        Peak p1 = Peak();
        p1.amplitude = 1.0;
        frame->add_peak(p1);

        Peak p2 = Peak();
        p2.amplitude = 2.0;
        frame->add_peak(p2);

        CPPUNIT_ASSERT(frame->num_peaks() == 2);

        int peak_num = 0;
        for(Peaks::iterator i = frame->peaks_begin(); i != frame->peaks_end(); i++)
        {
            if(peak_num == 0)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, i->amplitude, PRECISION);
            }
            else if(peak_num == 1)
            {
                CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, i->amplitude, PRECISION);
            }
            peak_num += 1;
        }
        frame->clear_peaks();
    }

public:
    void setUp() 
    {
        frame = new Frame();
    }

    void tearDown() 
    {
        delete frame;
    } 
};

// ---------------------------------------------------------------------------
//	TestPeakDetection
// ---------------------------------------------------------------------------
class TestPeakDetection : public CPPUNIT_NS::TestCase
{
    CPPUNIT_TEST_SUITE(TestPeakDetection);
    CPPUNIT_TEST(test_constructor);
    CPPUNIT_TEST(test_frame_size);
    CPPUNIT_TEST(test_static_frame_size);
    CPPUNIT_TEST(test_next_frame_size);
    CPPUNIT_TEST(test_hop_size);
    CPPUNIT_TEST(test_max_peaks);
    CPPUNIT_TEST(test_window_type);
    CPPUNIT_TEST(test_window_size);
    CPPUNIT_TEST(test_min_peak_separation);
    CPPUNIT_TEST(test_find_peaks_in_frame);
    CPPUNIT_TEST(test_find_peaks);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    PeakDetection* pd;

    void test_constructor() 
    { 
        CPPUNIT_ASSERT(pd->sampling_rate() == 44100);
        CPPUNIT_ASSERT(pd->frame_size() == 2048);
        CPPUNIT_ASSERT(pd->static_frame_size());
        CPPUNIT_ASSERT(pd->hop_size() == 512);
        CPPUNIT_ASSERT(pd->max_peaks() == 100);
        CPPUNIT_ASSERT(pd->window_type() == "hamming");
        CPPUNIT_ASSERT(pd->window_size() == 2048);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, pd->min_peak_separation(), 0.00001);
        CPPUNIT_ASSERT(pd->frames()->size() == 0);
    }

    void test_sampling_rate()
    {
        pd->sampling_rate(96000);
        CPPUNIT_ASSERT(pd->sampling_rate() == 96000);
        pd->sampling_rate(44100);
    }

    void test_frame_size()
    {
        pd->frame_size(1024);
        CPPUNIT_ASSERT(pd->frame_size() == 1024);
        pd->frame_size(2048);
    }

    void test_static_frame_size()
    {
        pd->static_frame_size(false);
        CPPUNIT_ASSERT(!pd->static_frame_size());
        pd->static_frame_size(true);
    }

    void test_next_frame_size()
    {
        CPPUNIT_ASSERT(pd->next_frame_size() == pd->frame_size());
    }

    void test_hop_size()
    {
        pd->hop_size(128);
        CPPUNIT_ASSERT(pd->hop_size() == 128);
        pd->hop_size(512);
    }

    void test_max_peaks()
    {
        pd->max_peaks(20);
        CPPUNIT_ASSERT(pd->max_peaks() == 20);
        pd->max_peaks(100);
    }

    void test_window_type()
    {
        pd->window_type("hanning");
        CPPUNIT_ASSERT(pd->window_type() == "hanning");
        pd->window_type("hamming");
    }

    void test_window_size()
    {
        pd->window_size(2048);
        CPPUNIT_ASSERT(pd->window_size() == 2048);
        pd->window_size(2048);
    }

    void test_min_peak_separation()
    {
        pd->min_peak_separation(0.5);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, pd->min_peak_separation(), 0.00001);
        pd->min_peak_separation(1.0);
    }

    void test_find_peaks_in_frame()
    {
        Frame* f = new Frame();
        Peaks* p = pd->find_peaks_in_frame(*f);
        CPPUNIT_ASSERT(p->size() == 0);
        delete p;
        delete f;
    }

    void test_find_peaks()
    {
        const samples audio = samples(1024);
        pd->frame_size(256);
        pd->hop_size(256);
        Frames* frames = pd->find_peaks(audio);
        CPPUNIT_ASSERT(frames->size() == 4);
        for(Frames::iterator i = frames->begin(); i != frames->end(); i++)
        {
            CPPUNIT_ASSERT(i->num_peaks() == 0);
        }
    }

public:
    void setUp() 
    {
        pd = new PeakDetection();
    }

    void tearDown() 
    {
        delete pd;
    } 
};

} // end of namespace Simpl

CPPUNIT_TEST_SUITE_REGISTRATION(Simpl::TestPeak);
CPPUNIT_TEST_SUITE_REGISTRATION(Simpl::TestFrame);
CPPUNIT_TEST_SUITE_REGISTRATION(Simpl::TestPeakDetection);

int main(int arg, char **argv)
{
    CppUnit::TextTestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    return runner.run("", false);
}
