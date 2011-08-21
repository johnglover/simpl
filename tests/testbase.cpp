/*
 * Copyright (c) 2009-2011 John Glover, National University of Ireland, Maynooth
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

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
    CPPUNIT_TEST(test_is_free);
    CPPUNIT_TEST(test_is_start_of_partial);
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

        CPPUNIT_ASSERT_THROW(peak->is_free("random_text"), InvalidArgument);
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
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    Frame* frame;

    void test_constructor() 
    { 
        CPPUNIT_ASSERT(frame->size() == 512);
        CPPUNIT_ASSERT(frame->audio.size() == 512);
        CPPUNIT_ASSERT(frame->synth.size() == 512);
        CPPUNIT_ASSERT(frame->residual.size() == 512);
        CPPUNIT_ASSERT(frame->synth_residual.size() == 512);
        CPPUNIT_ASSERT(frame->max_peaks() == 100);
        CPPUNIT_ASSERT(frame->peaks.size() == 100);
        CPPUNIT_ASSERT(frame->max_partials() == 100);
        CPPUNIT_ASSERT(frame->partials.size() == 100);
    }

    void test_size()
    {
        frame->size(1024);
        CPPUNIT_ASSERT(frame->size() == 1024);
        CPPUNIT_ASSERT(frame->audio.size() == 1024);
        CPPUNIT_ASSERT(frame->synth.size() == 1024);
        CPPUNIT_ASSERT(frame->residual.size() == 1024);
        CPPUNIT_ASSERT(frame->synth_residual.size() == 1024);
        frame->size(512);
    }

    void test_max_peaks()
    {
        frame->max_peaks(200);
        CPPUNIT_ASSERT(frame->max_peaks() == 200);
        CPPUNIT_ASSERT(frame->peaks.size() == 200);
        frame->max_peaks(100);
    }

    void test_max_partials()
    {
        frame->max_partials(200);
        CPPUNIT_ASSERT(frame->max_partials() == 200);
        CPPUNIT_ASSERT(frame->partials.size() == 200);
        frame->max_partials(100);
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
    CPPUNIT_TEST(test_hop_size);
    CPPUNIT_TEST(test_max_peaks);
    CPPUNIT_TEST(test_window_type);
    CPPUNIT_TEST(test_window_size);
    CPPUNIT_TEST(test_min_peak_separation);
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
