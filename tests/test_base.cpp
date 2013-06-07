#include "test_base.h"

using namespace simpl;

// ---------------------------------------------------------------------------
//	TestPeak
// ---------------------------------------------------------------------------
void TestPeak::setUp() {
    peak = new Peak();
}

void TestPeak::tearDown() {
    delete peak;
}


// ---------------------------------------------------------------------------
//	TestFrame
// ---------------------------------------------------------------------------

void TestFrame::setUp() {
    frame = new Frame();
}

void TestFrame::tearDown() {
    delete frame;
}

void TestFrame::test_size() {
    frame->size(1024);
    CPPUNIT_ASSERT(frame->size() == 1024);
    frame->size(512);
}

void TestFrame::test_max_peaks() {
    frame->max_peaks(200);
    CPPUNIT_ASSERT(frame->max_peaks() == 200);
    CPPUNIT_ASSERT(frame->num_peaks() == 0);
    frame->max_peaks(100);
}

void TestFrame::test_max_partials() {
    frame->max_partials(200);
    CPPUNIT_ASSERT(frame->max_partials() == 200);
    CPPUNIT_ASSERT(frame->num_partials() == 0);
    frame->max_partials(100);
}

void TestFrame::test_add_peak() {
    frame->add_peak(1.5, 220, 0, 0);
    CPPUNIT_ASSERT(frame->max_peaks() == 100);
    CPPUNIT_ASSERT(frame->num_peaks() == 1);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5, frame->peak(0)->amplitude, PRECISION);

    frame->add_peak(2.0, 440, 0, 0);
    CPPUNIT_ASSERT(frame->max_peaks() == 100);
    CPPUNIT_ASSERT(frame->num_peaks() == 2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, frame->peak(1)->amplitude, PRECISION);

    frame->clear();
}

void TestFrame::test_clear() {
    frame->add_peak(1.5, 220, 0, 0);
    CPPUNIT_ASSERT(frame->num_peaks() == 1);
    frame->clear();
    CPPUNIT_ASSERT(frame->num_peaks() == 0);
}
