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
    void setUp() {
        peak = new Peak();
    }

    void tearDown() {
        delete peak;
    }
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
    CPPUNIT_TEST_SUITE_END();

protected:
    static const double PRECISION = 0.001;
    Frame* frame;

    void test_size() {
        frame->size(1024);
        CPPUNIT_ASSERT(frame->size() == 1024);
        frame->size(512);
    }

    void test_max_peaks() {
        frame->max_peaks(200);
        CPPUNIT_ASSERT(frame->max_peaks() == 200);
        CPPUNIT_ASSERT(frame->num_peaks() == 0);
        frame->max_peaks(100);
    }

    void test_max_partials() {
        frame->max_partials(200);
        CPPUNIT_ASSERT(frame->max_partials() == 200);
        CPPUNIT_ASSERT(frame->num_partials() == 0);
        frame->max_partials(100);
    }

    void test_add_peak() {
        Peak p = Peak();
        p.amplitude = 1.5;
        frame->add_peak(&p);
        CPPUNIT_ASSERT(frame->max_peaks() == 100);
        CPPUNIT_ASSERT(frame->num_peaks() == 1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5, frame->peak(0)->amplitude, PRECISION);

        Peak p2 = Peak();
        p2.amplitude = 2.0;
        frame->add_peak(&p2);
        CPPUNIT_ASSERT(frame->max_peaks() == 100);
        CPPUNIT_ASSERT(frame->num_peaks() == 2);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, frame->peak(1)->amplitude, PRECISION);

        frame->clear();
    }

    void test_clear() {
        Peak p = Peak();
        p.amplitude = 1.5;
        frame->add_peak(&p);
        CPPUNIT_ASSERT(frame->num_peaks() == 1);
        frame->clear();
        CPPUNIT_ASSERT(frame->num_peaks() == 0);
    }

public:
    void setUp() {
        frame = new Frame();
    }

    void tearDown() {
        delete frame;
    }
};

} // end of namespace simpl

CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestPeak);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestFrame);

int main(int arg, char **argv) {
    CppUnit::TextTestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    return runner.run("", false);
}
