/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/**
 * @file
 */


#include <nupic/utils/LoggingException.hpp>
#include <nupic/utils/Random.hpp>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <gtest/gtest.h>

using namespace nupic;

#include "RandomPrivateOrig.c"

TEST(RandomTest, Seeding)
{

  // test getSeed
  Random r(98765);
  ASSERT_EQ(98765U, r.getSeed());
}



TEST(RandomTest, CapnpSerialization)
{
  // tests for Cap'n Proto serialization
  Random r1(99), r2;
  UInt32 v1, v2;

  const char* outputPath = "RandomTest1.temp";

  {
    std::ofstream out(outputPath, std::ios::binary);
    r1.write(out);
    out.close();
  }
  {
    std::ifstream in(outputPath, std::ios::binary);
    r2.read(in);
    in.close();
  }
  v1 = r1.getSeed();
  v2 = r2.getSeed();
  ASSERT_EQ(v1, v2) << "seeds differ";

  // clean up
  remove(outputPath);
}
