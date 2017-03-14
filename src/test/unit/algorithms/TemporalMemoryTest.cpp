/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2016, Numenta, Inc.  Unless you have an agreement
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
 * ----------------------------------------------------------------------
 */

/** @file
 * Implementation of unit tests for TemporalMemory
 */

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

#include <nupic/algorithms/TemporalMemory.hpp>
#include "gtest/gtest.h"

using namespace nupic::algorithms::temporal_memory;
using namespace std;

#define EPSILON 0.0000001

namespace {
  void check_tm_eq(const TemporalMemory& tm1, const TemporalMemory& tm2)
  {
    ASSERT_EQ(tm1.getCellsPerColumn(), tm2.getCellsPerColumn());
  }

  TEST(TemporalMemoryTest, testInitInvalidParams)
  {

    // Invalid cellsPerColumn
    TemporalMemory tm1;
    EXPECT_THROW(tm1.initialize(0), exception);
  }


  TEST(TemporalMemoryTest, testWrite)
  {
    TemporalMemory tm1(
      /*cellsPerColumn*/ 4,
      /*seed*/ 42
      );

    // Write and read back the proto
    stringstream ss;
    tm1.write(ss);

    TemporalMemory tm2;
    tm2.read(ss);

    // Check that the two temporal memory objects have the same attributes
    check_tm_eq(tm1, tm2);
  }

  // Uncomment these tests individually to save/load from a file.
  // This is useful for ad-hoc testing of backwards-compatibility.

  // TEST(TemporalMemoryTest, saveTestFile)
  // {
  //   TemporalMemory tm(
  //     /*columnDimensions*/ {32},
  //     /*cellsPerColumn*/ 4,
  //     /*activationThreshold*/ 3,
  //     /*initialPermanence*/ 0.21,
  //     /*connectedPermanence*/ 0.50,
  //     /*minThreshold*/ 2,
  //     /*maxNewSynapseCount*/ 3,
  //     /*permanenceIncrement*/ 0.10,
  //     /*permanenceDecrement*/ 0.10,
  //     /*predictedSegmentDecrement*/ 0.0,
  //     /*seed*/ 42
  //     );
  //
  //   serializationTestPrepare(tm);
  //
  //   const char* filename = "TemporalMemorySerializationSave.tmp";
  //   ofstream outfile;
  //   outfile.open(filename, ios::binary);
  //   tm.save(outfile);
  //   outfile.close();
  // }

  // TEST(TemporalMemoryTest, loadTestFile)
  // {
  //   TemporalMemory tm;
  //   const char* filename = "TemporalMemorySerializationSave.tmp";
  //   ifstream infile(filename, ios::binary);
  //   tm.load(infile);
  //   infile.close();
  //
  //   serializationTestVerify(tm);
  // }

  // TEST(TemporalMemoryTest, writeTestFile)
  // {
  //   TemporalMemory tm(
  //     /*columnDimensions*/ {32},
  //     /*cellsPerColumn*/ 4,
  //     /*activationThreshold*/ 3,
  //     /*initialPermanence*/ 0.21,
  //     /*connectedPermanence*/ 0.50,
  //     /*minThreshold*/ 2,
  //     /*maxNewSynapseCount*/ 3,
  //     /*permanenceIncrement*/ 0.10,
  //     /*permanenceDecrement*/ 0.10,
  //     /*predictedSegmentDecrement*/ 0.0,
  //     /*seed*/ 42
  //     );

  //   serializationTestPrepare(tm);

  //   const char* filename = "TemporalMemorySerializationWrite.tmp";
  //   ofstream outfile;
  //   outfile.open(filename, ios::binary);
  //   tm.write(outfile);
  //   outfile.close();
  // }

  // TEST(TemporalMemoryTest, readTestFile)
  // {
  //   TemporalMemory tm;
  //   const char* filename = "TemporalMemorySerializationWrite.tmp";
  //   ifstream infile(filename, ios::binary);
  //   tm.read(infile);
  //   infile.close();

  //   serializationTestVerify(tm);
  // }
} // end namespace nupic
