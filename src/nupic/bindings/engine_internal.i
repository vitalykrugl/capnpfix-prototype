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
%}

%naturalvar;

%{
#define SWIG_FILE_WITH_INIT
%}


%init %{

// Perform necessary library initialization (in C++).

%}


//--------------------------------------------------------------------------------


%pythoncode %{

class PyCapnpHelper(object):
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
    getattr(region, methodName)()

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
