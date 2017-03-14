/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

%module(package="bindings", directors="1") algorithms
//%include <nupic/bindings/exception.i>
%import <nupic/bindings/math.i>

%pythoncode %{
import os

_ALGORITHMS = _algorithms
%}

%{
/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

#include <Python.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>

//#include <nupic/math/Types.hpp>
#include <nupic/proto/TemporalMemoryProto.capnp.h>

#include <nupic/algorithms/TemporalMemory.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#if !CAPNP_LITE
#include <nupic/py_support/PyCapnp.hpp>
#endif
#include <nupic/py_support/PyHelpers.hpp>

// Hack to fix SWIGPY_SLICE_ARG not found bug
#if PY_VERSION_HEX >= 0x03020000
# define SWIGPY_SLICE_ARG(obj) ((PyObject*) (obj))
#else
# define SWIGPY_SLICE_ARG(obj) ((PySliceObject*) (obj))
#endif

/// %template(_InSynapse) nupic::algorithms::Cells3::InSynapse<nupic::UInt32, nupic::Real32>;
/// %template(Segment3_32) nupic::algorithms::Cells3::Segment<nupic::UInt32, nupic::Real32>;
/// %template(Cell3_32) nupic::algorithms::Cells3::Cell<nupic::UInt32, nupic::Real32>;
/// %template(Cells3_32) nupic::algorithms::Cells3::Cells3<nupic::UInt32, nupic::Real32>;
//using namespace nupic::algorithms::temporal_memory;
using namespace nupic;



#define CHECKSIZE(var) \
  NTA_ASSERT(PyArray_DESCR(var)->elsize == 4) << " elsize:" << PyArray_DESCR(var)->elsize

%}


// %pythoncode %{
//   import numpy
//   from bindings import math
// %}

%pythoncode %{
  uintDType = "uint32"
%}

%naturalvar;




//--------------------------------------------------------------------------------
// Temporal Memory
//--------------------------------------------------------------------------------
%include <nupic/algorithms/TemporalMemory.hpp>

%extend nupic::algorithms::temporal_memory::TemporalMemory
{
  %pythoncode %{

    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance
  %}


  inline void write(PyObject* pyBuilder) const
  {
%#if !CAPNP_LITE
    TemporalMemoryProto::Builder proto =
        getBuilder<TemporalMemoryProto>(pyBuilder);
    self->write(proto);
  %#else
    throw std::logic_error(
        "TemporalMemory.write is not implemented when compiled with CAPNP_LITE=1.");
  %#endif
  }

  inline void convertedRead(PyObject* pyReader)
  {
%#if !CAPNP_LITE
    TemporalMemoryProto::Reader proto =
        getReader<TemporalMemoryProto>(pyReader);
    self->read(proto);
  %#else
    throw std::logic_error(
        "TemporalMemory.read is not implemented when compiled with CAPNP_LITE=1.");
  %#endif
  }
}


//%include <nupic/algorithms/TemporalMemory.hpp>
