/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

try:
  import capnp
except ImportError:
  capnp = None
else:
  from nupic.proto.RandomProto_capnp import RandomProto


_MATH = _math
%}

%{
/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
#include <Python.h>
#include <nupic/types/Types.hpp>
//#include <nupic/math/Utils.hpp>
//#include <nupic/math/Math.hpp>
#include <nupic/proto/RandomProto.capnp.h>
#include <nupic/utils/Random.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#if !CAPNP_LITE
  #include <capnp/common.h> // for `class word`
  #include <capnp/message.h>
  #include <capnp/schema-parser.h>
  #include <capnp/serialize.h>
  //#include <nupic/py_support/PyCapnp.hpp>
#endif

#include <nupic/py_support/PyHelpers.hpp>


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



//--------------------------------------------------------------------------------

// ----- Random -----

%include <nupic/utils/LoggingException.hpp>
%include <nupic/utils/Random.hpp>

%extend nupic::Random {
  %pythoncode %{

    def writeOut(self):
      """ Serialize the instance using pycapnp.

      :returns: RandomProto message reader containing the serialized data. This
                value may be assigned to the corresponding property of the
                higher-level message builder.
      """
      return RandomProto.from_bytes(self._writeAsBytes()) # copy


    @staticmethod
    def readIn(proto):
      """ Deserialize the given RandomProto reader into a new Random instance.

      :param proto: RandomProto message reader containing data from a previously
                    serialized Random instance.

      :returns: A new Random instance initialized from the contents of the given
                RandomProto message reader.

      """
      return Random._readFromBytes(proto.as_builder().to_bytes()) # copy * 2 ?
  %}


  inline PyObject* _writeAsBytes() const
  {
  %#if !CAPNP_LITE
    capnp::MallocMessageBuilder message;
    RandomProto::Builder proto = message.initRoot<RandomProto>();

    self->write(proto);

    // Extract message data and convert to Python byte object
    kj::Array<capnp::word> array = capnp::messageToFlatArray(message); // copy
    kj::ArrayPtr<kj::byte> byteArray = array.asBytes();
    PyObject* result = PyString_FromStringAndSize(
      (const char*)byteArray.begin(),
      byteArray.size()); // copy
    return result;
  %#else
    throw std::logic_error(
        "Random._writeAsBytes is not implemented when compiled with CAPNP_LITE=1.");
  %#endif
  }


  inline static Random* _readFromBytes(PyObject* bytesPyObj) const
  {
  %#if !CAPNP_LITE
    //const char * srcBytes = nullptr;
    //int srcNumBytes = 0;
    char * srcBytes = nullptr;
    Py_ssize_t srcNumBytes = 0;
    //PyArg_Parse(bytesPyObj, "s#", &srcBytes, &srcNumBytes);
    PyString_AsStringAndSize(bytesPyObj, &srcBytes, &srcNumBytes);

    if (srcNumBytes % sizeof(capnp::word) != 0)
    {
      throw std::logic_error(
          "Random._readFromBytes input length must be a multiple of capnp::word.");
    }
    const int srcNumWords = srcNumBytes / sizeof(capnp::word);

    // Ensure alignment on capnp::word boundary; TODO can we do w/o this copy or
    // make copy conditional on alignment like pycapnp does?
    kj::Array<capnp::word> array = kj::heapArray<capnp::word>(srcNumWords);
    memcpy(array.asBytes().begin(), srcBytes, srcNumBytes);

    capnp::FlatArrayMessageReader reader(array.asPtr());
    RandomProto::Reader proto = reader.getRoot<RandomProto>();
    return new Random(proto);
  %#else
    throw std::logic_error(
        "Random._readFromBytes is not implemented when compiled with CAPNP_LITE=1.");
  %#endif
  }


  //inline void read(PyObject* pyReader)
  //{
  //%#if !CAPNP_LITE
  //  RandomProto::Reader proto = nupic::getReader<RandomProto>(pyReader);
  //  self->read(proto);
  //%#else
  //  throw std::logic_error(
  //      "Random.read is not implemented when compiled with CAPNP_LITE=1.");
  //%#endif
  //}

} // End extend nupic::Random
