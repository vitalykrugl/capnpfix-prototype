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

/** @file
    Random Number Generator implementation
*/

#include <cstdlib>
#include <ctime>
#include <cmath> // For ldexp.
#include <iostream> // for istream, ostream

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <nupic/proto/RandomProto.capnp.h>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/Random.hpp>

using namespace nupic;


Random::Random(unsigned long seed)
{
  seed_ = seed;
}


Random::Random(RandomProto::Reader& proto)
{
  seed_ = proto.getSeed();
}


Random::~Random()
{
}


void Random::write(RandomProto::Builder& proto) const
{
  // save Random state
  proto.setSeed(seed_);

  //// save RandomImpl state
  //auto implProto = proto.initImpl();
  //impl_->write(implProto);
}


void Random::read(RandomProto::Reader& proto)
{
  // load Random state
  seed_ = proto.getSeed();

  //// load RandomImpl state
  //auto implProto = proto.getImpl();
  //impl_->read(implProto);
}



