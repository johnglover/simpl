#ifndef TEST_RESIDUAL_H
#define TEST_RESIDUAL_H

#include <cppunit/extensions/HelperMacros.h>

#include "../src/simpl/base.h"
#include "../src/simpl/peak_detection.h"
#include "../src/simpl/partial_tracking.h"
#include "../src/simpl/synthesis.h"
#include "../src/simpl/residual.h"
#include "test_common.h"

namespace simpl
{

// ---------------------------------------------------------------------------
//	TestSMSResidual
// ---------------------------------------------------------------------------
class TestSMSResidual : public CPPUNIT_NS::TestCase {
    CPPUNIT_TEST_SUITE(TestSMSResidual);
    CPPUNIT_TEST(test_basic);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();

protected:
    SMSResidual _res;
    SndfileHandle _sf;
    Frames _frames;

    void test_basic();
};

} // end of namespace simpl

#endif
