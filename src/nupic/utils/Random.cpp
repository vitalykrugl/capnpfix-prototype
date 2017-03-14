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

const UInt32 MAX32 = (UInt32)((Int32)(-1));

/**
 * Using an Impl provides two things:
 * 1) ability to specify different algorithms (not yet implemented)
 * 2) constructors Random(long) and Random(string) without code duplication.
 */

// Algorithm-level implementation of the random number generator.
// When we have different algorithms RandomImpl will become an interface
// class and subclasses will implement specific algorithms

namespace nupic
{
  class RandomImpl
  {
  public:
    RandomImpl(UInt64 seed);
    ~RandomImpl() {};
    void write(RandomImplProto::Builder& proto) const;
    void read(RandomImplProto::Reader& proto);
    UInt32 getUInt32();
    // Note: copy constructor and operator= are needed
    // The default is ok.
  private:
    const static UInt32 VERSION = 2;
    // internal state
    static const int stateSize_ = 31;
    static const int sep_ = 3;
    UInt32 state_[stateSize_];
    int rptr_;
    int fptr_;

  };
};



Random::Random(const Random& r)
{
  NTA_CHECK(r.impl_ != nullptr);
  seed_ = r.seed_;
  impl_ = new RandomImpl(*r.impl_);
}


Random& Random::operator=(const Random& other)
{
  if (this != &other)
  {
    seed_ = other.seed_;
    if (impl_)
      delete impl_;
    NTA_CHECK(other.impl_ != nullptr);
    impl_ = new RandomImpl(*other.impl_);
  }
  return *this;
}


void Random::write(RandomProto::Builder& proto) const
{
  // save Random state
  proto.setSeed(seed_);

  // save RandomImpl state
  auto implProto = proto.initImpl();
  impl_->write(implProto);
}

void Random::read(RandomProto::Reader& proto)
{
  // load Random state
  seed_ = proto.getSeed();

  // load RandomImpl state
  auto implProto = proto.getImpl();
  impl_->read(implProto);
}


Random::~Random()
{
  delete impl_;
}


Random::Random(unsigned long seed)
{
  seed_ = seed;

  impl_ = new RandomImpl(seed_);
}


// ---- RandomImpl follows ----




UInt32 RandomImpl::getUInt32(void)
{
  UInt32 i;
#ifdef RANDOM_SUPERDEBUG
  printf("Random::get *fptr = %ld; *rptr = %ld fptr = %ld rptr = %ld\n", state_[fptr_], state_[rptr_], fptr_, rptr_);
#endif
  state_[fptr_] = (UInt32)(
    ((UInt64)state_[fptr_] + (UInt64)state_[rptr_]) % MAX32);
  i = state_[fptr_];
  i = (i >> 1) & 0x7fffffff;	/* chucking least random bit */
  if (++fptr_ >= stateSize_) {
    fptr_ = 0;
    ++rptr_;
  } else if (++rptr_ >= stateSize_)
    rptr_ = 0;
#ifdef RANDOM_SUPERDEBUG
  printf("Random::get returning %ld\n", i);
  for (int j = 0; j < stateSize_; j++) {
    printf("Random:get: %d  %ld\n", j, state_[j]);
  }
#endif

  return i;
}



RandomImpl::RandomImpl(UInt64 seed)
{

  /**
   * Initialize our state. Taken from BSD source for random()
   */
  state_[0] = (UInt32)(seed % MAX32);
  for (long i = 1; i < stateSize_; i++) {
    /*
     * Implement the following, without overflowing 31 bits:
     *
     *	state[i] = (16807 * state[i - 1]) % 2147483647;
     *
     *	2^31-1 (prime) = 2147483647 = 127773*16807+2836
     */
    Int32 quot = state_[i-1] / 127773;
    Int32 rem = state_[i-1] % 127773;
    Int32 test = 16807 * rem - 2836 * quot;
    state_[i] = (UInt32)((test + (test < 0 ? 2147483647 : 0)) % MAX32);
  }
  fptr_ = sep_;
  rptr_ = 0;
#ifdef RANDOM_SUPERDEBUG
  printf("Random: init for seed = %lu\n", seed);
  for (int i = 0; i < stateSize_; i++) {
    printf("Random: %d  %ld\n", i, state_[i]);
  }
#endif

  for (long i = 0; i < 10 * stateSize_; i++)
    (void)getUInt32();
#ifdef RANDOM_SUPERDEBUG
  printf("Random: after init for seed = %lu\n", seed);
  printf("Random: *fptr = %ld; *rptr = %ld fptr = %ld rptr = %ld\n", state_[fptr_], state_[rptr_], fptr_, rptr_);
  for (long i = 0; i < stateSize_; i++) {
    printf("Random: %d  %ld\n", i, state_[i]);
  }
#endif
}


void RandomImpl::write(RandomImplProto::Builder& proto) const
{
  auto state = proto.initState(stateSize_);
  for (UInt i = 0; i < stateSize_; ++i)
  {
    state.set(i, state_[i]);
  }
  proto.setRptr(rptr_);
  proto.setFptr(fptr_);
}


void RandomImpl::read(RandomImplProto::Reader& proto)
{
  auto state = proto.getState();
  for (UInt i = 0; i < state.size(); ++i)
  {
    state_[i] = state[i];
  }
  rptr_ = proto.getRptr();
  fptr_ = proto.getFptr();
}







