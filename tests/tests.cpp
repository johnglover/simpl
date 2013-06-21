#include <cppunit/ui/text/TextTestRunner.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>

#include "test_base.h"
#include "test_peak_detection.h"
#include "test_partial_tracking.h"
#include "test_synthesis.h"

CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestPeak);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestFrame);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestMQPeakDetection);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestSndObjPeakDetection);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestTWM);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestLorisPeakDetection);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestMQPartialTracking);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestSMSPartialTracking);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestSndObjPartialTracking);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestLorisPartialTracking);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestMQSynthesis);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestLorisSynthesis);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestSMSSynthesis);
CPPUNIT_TEST_SUITE_REGISTRATION(simpl::TestSndObjSynthesis);

int main(int arg, char **argv) {
    CppUnit::TextTestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    return runner.run("", false);
}
