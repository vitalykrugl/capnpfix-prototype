/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

// This file contains utility functions for converting from pycapnp schema to
// compiled in schema and vice versa.
// It requires linking to both libcapnp and libcapnpc.

#ifndef NTA_PY_CAPNP_HPP
#define NTA_PY_CAPNP_HPP

#if !CAPNP_LITE
#include <Python.h>

#include <capnp/any.h>
#include <capnp/dynamic.h>
#include <capnp/message.h>
#include <capnp/schema-parser.h>

#include <nupic/types/Serializable.hpp>

namespace nupic
{
  class PyCapnpHelper
  {
  public:
    /**
     * Serialize object returning a capnp byte buffer as python byte string.
     *
     * :param obj: The Serializable object
     *
     * :returns: capnp byte buffer encoded as python byte string.
     *
     * :example: PyObject* pyBytes = PyCapnpHelper::writeAsBytes(*netPtr);
     */
    template<class MessageType>
    static PyObject* writeAsBytes(const nupic::Serializable<MessageType>& obj)
    {
      capnp::MallocMessageBuilder message;
      typename MessageType::Builder proto =
        message.initRoot<MessageType>();

      obj.write(proto);

      // Extract message data and convert to Python byte object
      kj::Array<capnp::word> array = capnp::messageToFlatArray(message); // copy
      kj::ArrayPtr<kj::byte> byteArray = array.asBytes();
      PyObject* result = PyString_FromStringAndSize(
        (const char*)byteArray.begin(),
        byteArray.size()); // copy
      return result;
    }

    /**
     * Serialize object returning a capnp byte buffer as python byte string.
     *
     * :param ObjectCls: template arg; type of result object to allocate on heap
     *                   without constructor args. Class must be compatible with
     *                   nupic::Serializable deserialization API. E.g.,
     *                   `nupic::Network`.
     * :param MessageType: template arg; type of Capnp Proto Message that
     *                     corresponds to the capnp encoding contained within
     *                     the pyBytes arg and the ObjectCls's `read` instance
     *                     method accepts. E.g., `NetworkProto`.
     *
     * :param pyBytes: The Serializable object
     *
     * :returns: capnp byte buffer encoded as python byte string.
     *
     * :example: auto net =
     *             PyCapnpHelper::readFromPyBytes<nupic::Network,
     *                                            NetworkProto>(pyBytes);
     */
    template<class ObjectCls, class MessageType>
    static ObjectCls* readFromPyBytes(const PyObject* pyBytes)
    {
      char * srcBytes = nullptr;
      Py_ssize_t srcNumBytes = 0;
      PyString_AsStringAndSize(const_cast<PyObject*>(pyBytes),
                               &srcBytes,
                               &srcNumBytes);

      if (srcNumBytes % sizeof(capnp::word) != 0)
      {
        throw std::logic_error(
          "PyCapnpHelper.readFromPyBytes input length must be a multiple of "
          "capnp::word.");
      }
      const int srcNumWords = srcNumBytes / sizeof(capnp::word);

      // Ensure alignment on capnp::word boundary; TODO can we do w/o this copy or
      // make copy conditional on alignment like pycapnp does?
      kj::Array<capnp::word> array = kj::heapArray<capnp::word>(srcNumWords);
      memcpy(array.asBytes().begin(), srcBytes, srcNumBytes);

      capnp::FlatArrayMessageReader reader(array.asPtr());
      typename MessageType::Reader proto = reader.getRoot<MessageType>();
      auto obj = new ObjectCls();
      obj->read(proto);
      return obj;
    }

  }; // class PyCapnpHelper

}  // namespace nupic

#endif // !CAPNP_LITE

#endif  // NTA_PY_CAPNP_HPP
