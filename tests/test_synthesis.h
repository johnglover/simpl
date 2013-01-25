#ifndef TEST_SYNTHESIS_H
#define TEST_SYNTHESIS_H

#include <cppunit/extensions/HelperMacros.h>

#include "../src/simpl/base.h"
#include "../src/simpl/peak_detection.h"
#include "../src/simpl/partial_tracking.h"
#include "../src/simpl/synthesis.h"
#include "test_common.h"

namespace simpl
{

// ---------------------------------------------------------------------------
//	TestMQSynthesis
// ---------------------------------------------------------------------------
class TestMQSynthesis : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestMQSynthesis);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST(test_changing_frame_size);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();

protected:
    MQPeakDetection _pd;
    MQPartialTracking _pt;
    MQSynthesis _synth;
    SndfileHandle _sf;

    void test_basic();
    void test_changing_frame_size();
};

// ---------------------------------------------------------------------------
//	TestLorisSynthesis
// ---------------------------------------------------------------------------
class TestLorisSynthesis : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestLorisSynthesis);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST(test_changing_frame_size);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();

protected:
    LorisPeakDetection _pd;
    LorisPartialTracking _pt;
    LorisSynthesis _synth;
    SndfileHandle _sf;

    void test_basic();
    void test_changing_frame_size();
};

} // end of namespace simpl

#endif
