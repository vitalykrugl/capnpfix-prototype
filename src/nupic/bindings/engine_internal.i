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

%module(package="bindings") engine_internal
//%include <nupic/bindings/exception.i>

%pythoncode %{

try:
  import capnp
except ImportError:
  capnp = None
else:
  from nupic.proto.PyRegionProto_capnp import PyRegionProto
  from nupic.proto.NetworkProto_capnp import NetworkProto
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

#include <nupic/proto/NetworkProto.capnp.h>
#include <nupic/engine/Network.hpp>

#if !CAPNP_LITE
  #include <capnp/common.h> // for `class word`
  #include <capnp/message.h>
  #include <capnp/schema-parser.h>
  #include <capnp/serialize.h>
  //#include <nupic/py_support/PyCapnp.hpp>
#endif

using namespace nupic;

%}

%naturalvar;

%{
#define SWIG_FILE_WITH_INIT
%}


%init %{

// Perform necessary library initialization (in C++).

%}


//--------------------------------------------------------------------------------


// ----- Newtwork -----

%include <nupic/utils/LoggingException.hpp>
%include <nupic/engine/Network.hpp>

%extend nupic::Network {
  %pythoncode %{

    def writeOut(self):
      """ Serialize the instance using pycapnp.

      :returns: NetworkProto message reader containing the serialized data. This
                value may be assigned to the corresponding property of the
                higher-level message builder.
      """
      return NetworkProto.from_bytes(self._writeAsBytes()) # copy


    @staticmethod
    def readIn(proto):
      """ Deserialize the given NetworkProto reader into a new Network instance.

      :param proto: NetworkProto message reader containing data from a previously
                    serialized Network instance.

      :returns: A new Network instance initialized from the contents of the given
                NetworkProto message reader.

      """
      return Network._readFromBytes(proto.as_builder().to_bytes()) # copy * 2 ?
  %}


  inline PyObject* _writeAsBytes() const
  {
  %#if !CAPNP_LITE
    capnp::MallocMessageBuilder message;
    NetworkProto::Builder proto = message.initRoot<NetworkProto>();

    self->write(proto);

    // Extract message data and convert to Python byte object
    auto array = capnp::messageToFlatArray(message);
    const char* ptr = (const char *)array.begin();
    PyObject* result = PyString_FromStringAndSize(ptr, sizeof(capnp::word)*array.size()); // copy
    return result;
  %#else
    throw std::logic_error(
        "Network._writeAsBytes is not implemented when compiled with CAPNP_LITE=1.");
  %#endif
  }


  inline static Network* _readFromBytes(PyObject* bytesPyObj) const
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
          "Network._readFromBytes input length must be a multiple of capnp::word.");
    }
    const int srcNumWords = srcNumBytes / sizeof(capnp::word);

    // Ensure alignment on capnp::word boundary; TODO can we do w/o this copy or
    // make copy conditional on alignment like pycapnp does?
    kj::Array<capnp::word> array = kj::heapArray<capnp::word>(srcNumWords);
    memcpy(array.asBytes().begin(), srcBytes, srcNumBytes);

    capnp::FlatArrayMessageReader reader(array.asPtr());
    NetworkProto::Reader proto = reader.getRoot<NetworkProto>();
    auto net = new nupic::Network();
    net->read(proto);
    return net;
  %#else
    throw std::logic_error(
        "Network._readFromBytes is not implemented when compiled with CAPNP_LITE=1.");
  %#endif
  }

} // End extend nupic::Network



%pythoncode %{

class _PyCapnpHelper(object):
  """Only for use by the extension layer. Wraps certain serialization operations
  from the C++ extension layer to simplify python-side implementation
  """

  @staticmethod
  def writePyRegion(region, methodName):
    """ Serialize the given python region using the given method name

    :param region: Python region instance
    :param methodName: Name of method to invoke on the region to serialize it.

    :returns: Data bytes corresponding to the serialized PyRegionProto message
    """
    builderProto = PyRegionProto.new_message()
    # Serialize
    getattr(region, methodName)(builderProto)

    return builderProto.to_bytes()


  @staticmethod
  def readPyRegion(pyRegionProtoBytes, regionCls, methodName):
    """ Deserialize the given python region data bytes using the given method
    name on the given class

    :param pyRegionProtoBytes: data bytes string corresponding to the
                               PyRegionProto message.
    :param regionCls: Python region class
    :param methodName: Name of method to invoke on the region to deserialize it.

    :returns: The deserialized python region instance.
    """
    pyRegionProto = PyRegionProto.from_bytes(pyRegionProtoBytes)

    return getattr(regionCls, methodName)(pyRegionProto)


%} // pythoncode
