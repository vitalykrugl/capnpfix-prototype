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

%module(package="bindings") math
//%include <nupic/bindings/exception.i>

%pythoncode %{
_MATH = _math
%}

%{
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

#include <cmath>
#include <nupic/types/Types.hpp>
//#include <nupic/math/Utils.hpp>
//#include <nupic/math/Math.hpp>
#include <nupic/proto/RandomProto.capnp.h>
#include <nupic/utils/Random.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#if !CAPNP_LITE
#include <nupic/py_support/PyCapnp.hpp>
#endif

using namespace nupic;

%}

%naturalvar;

%{
#define SWIG_FILE_WITH_INIT
%}


%init %{

// Perform necessary library initialization (in C++).
import_array();

%}


///////////////////////////////////////////////////////////////////
/// Utility functions that are expensive in Python but fast in C.
///////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------

// ----- Random -----

%include <nupic/utils/LoggingException.hpp>
%include <nupic/utils/Random.hpp>

%extend nupic::Random {


inline void write(PyObject* pyBuilder) const
{
%#if !CAPNP_LITE
  RandomProto::Builder proto = nupic::getBuilder<RandomProto>(pyBuilder);
  self->write(proto);
%#else
  throw std::logic_error(
      "Random.write is not implemented when compiled with CAPNP_LITE=1.");
%#endif
}

inline void read(PyObject* pyReader)
{
%#if !CAPNP_LITE
  RandomProto::Reader proto = nupic::getReader<RandomProto>(pyReader);
  self->read(proto);
%#else
  throw std::logic_error(
      "Random.read is not implemented when compiled with CAPNP_LITE=1.");
%#endif
}

} // End extend nupic::Random.
