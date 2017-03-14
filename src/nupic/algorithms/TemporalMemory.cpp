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
 * Implementation of TemporalMemory
 *
 * The functions in this file use the following argument ordering
 * convention:
 *
 * 1. Output / mutated params
 * 2. Traditional parameters to the function, i.e. the ones that would still
 *    exist if this function were a method on a class
 * 3. Model state (marked const)
 * 4. Model parameters (including "learn")
 */

#include <cstring>
#include <climits>
#include <iomanip>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <nupic/algorithms/TemporalMemory.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::temporal_memory;

static const UInt TM_VERSION = 2;


TemporalMemory::TemporalMemory()
{
}

TemporalMemory::TemporalMemory(
  unsigned cellsPerColumn,
  int seed)
{
  initialize(
    cellsPerColumn,
    seed);
}

TemporalMemory::~TemporalMemory()
{
}

void TemporalMemory::initialize(
  unsigned cellsPerColumn,
  int seed)
{
  // Validate all input parameters

  if (cellsPerColumn <= 0)
  {
    NTA_THROW << "Number of cells per column must be greater than 0";
  }

  cellsPerColumn_ = cellsPerColumn;
  seed_((UInt64)(seed < 0 ? rand() : seed));
}


unsigned TemporalMemory::version() const
{
  return TM_VERSION;
}

/**
* Create a RNG with given seed
*/
void TemporalMemory::seed_(unsigned long seed)
{
  rng_ = Random(seed);
}


unsigned TemporalMemory::getCellsPerColumn() const
{
  return cellsPerColumn_;
}


void TemporalMemory::write(TemporalMemoryProto::Builder& proto) const
{

  proto.setCellsPerColumn(cellsPerColumn_);

  auto random = proto.initRandom();
  rng_.write(random);
}

// Implementation note: this method sets up the instance using data from
// proto. This method does not call initialize. As such we have to be careful
// that everything in initialize is handled properly here.
void TemporalMemory::read(TemporalMemoryProto::Reader& proto)
{
  cellsPerColumn_ = proto.getCellsPerColumn();

  auto random = proto.getRandom();
  rng_.read(random);
}
