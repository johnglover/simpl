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

namespace Simpl 
{

class TestBase : public CPPUNIT_NS::TestCase
{
    CPPUNIT_TEST_SUITE(TestBase);
    CPPUNIT_TEST(test_constructor);
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

} // end of namespace Simpl

CPPUNIT_TEST_SUITE_REGISTRATION(Simpl::TestBase);

int main(int arg, char **argv)
{
    CppUnit::TextTestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    return runner.run("", false);
}
