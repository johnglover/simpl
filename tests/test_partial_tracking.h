#ifndef TEST_PARTIAL_TRACKING_H
#define TEST_PARTIAL_TRACKING_H

#include <cppunit/extensions/HelperMacros.h>

#include "../src/simpl/base.h"
#include "../src/simpl/peak_detection.h"
#include "../src/simpl/partial_tracking.h"
#include "test_common.h"

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

public:
    void setUp();

protected:
    MQPeakDetection _pd;
    MQPartialTracking _pt;
    SndfileHandle _sf;

    void test_basic();
    void test_peaks();
};


// ---------------------------------------------------------------------------
//	TestSMSPartialTracking
// ---------------------------------------------------------------------------
class TestSMSPartialTracking : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestSMSPartialTracking);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST(test_change_num_partials);
    CPPUNIT_TEST(test_peaks);
    CPPUNIT_TEST(test_streaming);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();

protected:
    SMSPeakDetection _pd;
    SMSPartialTracking _pt;
    SndfileHandle _sf;

    void test_basic();
    void test_change_num_partials();
    void test_peaks();
    void test_streaming();
};


// ---------------------------------------------------------------------------
//	TestSndObjPartialTracking
// ---------------------------------------------------------------------------
class TestSndObjPartialTracking : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestSndObjPartialTracking);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST(test_change_num_partials);
    CPPUNIT_TEST(test_peaks);
    CPPUNIT_TEST(test_streaming);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();

protected:
    SndObjPeakDetection _pd;
    SndObjPartialTracking _pt;
    SndfileHandle _sf;

    void test_basic();
    void test_change_num_partials();
    void test_peaks();
    void test_streaming();
};

// ---------------------------------------------------------------------------
//	TestLorisPartialTracking
// ---------------------------------------------------------------------------
class TestLorisPartialTracking : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestLorisPartialTracking);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST(test_peaks);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();

protected:
    LorisPeakDetection _pd;
    LorisPartialTracking _pt;
    SndfileHandle _sf;

    void test_basic();
    void test_peaks();
};

} // end of namespace simpl

#endif
