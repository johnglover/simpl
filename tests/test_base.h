#ifndef TEST_BASE_H
#define TEST_BASE_H

#include <iostream>
#include <cppunit/ui/text/TextTestRunner.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>

#include "../src/simpl/base.h"

namespace simpl
{

// ---------------------------------------------------------------------------
//	TestPeak
// ---------------------------------------------------------------------------
class TestPeak : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestPeak);
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    Peak* peak;

public:
    void setUp();
    void tearDown();
};


// ---------------------------------------------------------------------------
//	TestFrame
// ---------------------------------------------------------------------------
class TestFrame : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestFrame);
    CPPUNIT_TEST(test_size);
    CPPUNIT_TEST(test_max_peaks);
    CPPUNIT_TEST(test_max_partials);
    CPPUNIT_TEST(test_add_peak);
    CPPUNIT_TEST(test_clear);
    CPPUNIT_TEST(test_audio);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();
    void tearDown();

protected:
    static const double PRECISION = 0.001;
    Frame* frame;

    void test_size();
    void test_max_peaks();
    void test_max_partials();
    void test_add_peak();
    void test_clear();
    void test_audio();
};

} // end of namespace simpl

#endif
