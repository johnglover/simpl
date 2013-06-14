#ifndef TEST_PEAK_DETECTION_H
#define TEST_PEAK_DETECTION_H

#include <cppunit/extensions/HelperMacros.h>

#include "../src/simpl/base.h"
#include "../src/simpl/peak_detection.h"
#include "../src/simpl/exceptions.h"
#include "test_common.h"

namespace simpl
{

// ---------------------------------------------------------------------------
//	TestMQPeakDetection
// ---------------------------------------------------------------------------
class TestMQPeakDetection : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestMQPeakDetection);
    CPPUNIT_TEST(test_find_peaks_in_frame_basic);
    CPPUNIT_TEST(test_find_peaks_basic);
    CPPUNIT_TEST(test_find_peaks_audio);
    CPPUNIT_TEST(test_find_peaks_change_hop_frame_size);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();

protected:
    MQPeakDetection _pd;
    SndfileHandle _sf;

    void test_find_peaks_in_frame_basic();
    void test_find_peaks_basic();
    void test_find_peaks_audio();
    void test_find_peaks_change_hop_frame_size();
};


// ---------------------------------------------------------------------------
//	TestTWM
// ---------------------------------------------------------------------------
class TestTWM : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestTWM);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST_SUITE_END();

protected:
    void test_basic();
};


// ---------------------------------------------------------------------------
//	TestLorisPeakDetection
// ---------------------------------------------------------------------------
class TestLorisPeakDetection : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestLorisPeakDetection);
    CPPUNIT_TEST(test_find_peaks_in_frame_basic);
    CPPUNIT_TEST(test_find_peaks_basic);
    CPPUNIT_TEST(test_find_peaks_audio);
    CPPUNIT_TEST(test_find_peaks_change_hop_frame_size);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();

protected:
    LorisPeakDetection _pd;
    SndfileHandle _sf;

    void test_find_peaks_in_frame_basic();
    void test_find_peaks_basic();
    void test_find_peaks_audio();
    void test_find_peaks_change_hop_frame_size();
};


// ---------------------------------------------------------------------------
//	TestSndObjPeakDetection
// ---------------------------------------------------------------------------
class TestSndObjPeakDetection : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestSndObjPeakDetection);
    CPPUNIT_TEST(test_find_peaks_in_frame_basic);
    CPPUNIT_TEST(test_find_peaks_basic);
    CPPUNIT_TEST(test_find_peaks_audio);
    CPPUNIT_TEST(test_find_peaks_change_hop_frame_size);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();

protected:
    SndObjPeakDetection _pd;
    SndfileHandle _sf;

    void test_find_peaks_in_frame_basic();
    void test_find_peaks_basic();
    void test_find_peaks_audio();
    void test_find_peaks_change_hop_frame_size();
};

} // end of namespace simpl

#endif
