//#include <cstring>
//#include <cstdlib>
// TODO Strange, if I eclude it, then get compiler error on sstream included
//from LoggingException.hpp.
#include <sstream>
#include <string>

#include <Python.h>

#if !CAPNP_LITE
  //#include <capnp/common.h> // for `class word`
  #include <capnp/message.h>
  #include <capnp/schema-parser.h>
  #include <capnp/serialize.h>
#endif


#include <nupic/proto/NetworkProto.capnp.h>
#include <nupic/proto/RegionProto.capnp.h>
#include <nupic/proto/PyRegionProto.capnp.h>
#include <nupic/py_support/PyCapnp.hpp>
#include <nupic/py_support/PyHelpers.hpp>

#include <nupic/engine/Network.hpp>


namespace nupic
{

Network::Network()
  :_pyRegionModuleName(),
   _pyRegionClassName(),
   _pyRegion()
{
}


Network::~Network()
{
}


std::string Network::getPythonRegionClassName()
{
  return _pyRegionClassName;
}


void Network::setPythonRegion(char* module, char* className,
                              unsigned long width, unsigned long seed)
{
  _pyRegionModuleName = module;
  _pyRegionClassName = className;

  py::Tuple args((Py_ssize_t)0);
  py::Dict kwargs;
  kwargs.setItem("width", py::UnsignedLong(width));
  kwargs.setItem("seed", py::UnsignedLong(seed));

  _pyRegion.assign(py::Instance(module, className, args, kwargs));
  NTA_CHECK(_pyRegion);
}


PyObject* Network::getPythonRegion()
{
  return _pyRegion;
}



PyRegionProto::Reader Network::_writePyRegion() const
{
#if !CAPNP_LITE
  // Request python object to write itself out and return PyRegionProto
  // serialized as a python byte array

  // NOTE Wrap the operation in our _PyCapnpHelper class to simplify the
  // interface for the targer python region implementation
  py::Class pyCapnpHelperCls("nupic.bindings.engine_internal",
                             "_PyCapnpHelper");
  py::Tuple args((Py_ssize_t)0);
  py::Dict kwargs;
  // NOTE py::Dict::setItem doesn't accept a const PyObject*, however we know
  // that we won't modify it, so casting trickery is okay here
  kwargs.setItem("region", (PyObject*)static_cast<const PyObject*>(_pyRegion));
  kwargs.setItem("methodName", py::String("write"));

  // Wrap result in py::Ptr to force dereferencing when going out of scope
  py::Ptr pyRegionProtoBytes(
    pyCapnpHelperCls.invoke("writePyRegion", args, kwargs));

  char * srcBytes = nullptr;
  Py_ssize_t srcNumBytes = 0;
  // NOTE: srcBytes will be set to point to the internal buffer inside
  // pyRegionProtoBytes'
  PyString_AsStringAndSize(pyRegionProtoBytes, &srcBytes, &srcNumBytes);

  // Ensure alignment on capnp::word boundary; TODO can we do w/o this copy or
  // make copy conditional on existing alignment like pycapnp does?
  //
  // TODO Verify that we don't end up with a dangling reference to deleted
  // array data when we return the resulting PyRegionProto::Reader
  const int srcNumWords = srcNumBytes / sizeof(capnp::word);
  kj::Array<capnp::word> array = kj::heapArray<capnp::word>(srcNumWords);
  std::memcpy(array.asBytes().begin(), srcBytes, srcNumBytes);

  capnp::FlatArrayMessageReader reader(array.asPtr());
  PyRegionProto::Reader proto = reader.getRoot<PyRegionProto>();

  return proto;
#else
  throw std::logic_error(
    "Network::_writePyRegion is not implemented when compiled with CAPNP_LITE=1.");
#endif
}


PyObject* Network::_readPyRegion(const std::string& moduleName,
                                 const std::string& className,
                                 const RegionProto::Reader& proto)
{
#if !CAPNP_LITE
  capnp::AnyPointer::Reader implProto = proto.getRegionImpl();

  PyRegionProto::Reader pyRegionImplProto = implProto.getAs<PyRegionProto>();

  // See PyRegion::read implementation for reference

  // Extract data bytes from reader to pass to python layer
  //
  // NOTE: this requires conversion to builder, because readers don't have the
  // method messageToFlatArray. TODO check if we could do something with
  // `getSegment` on readers)
  //
  // TODO Need to look into reducing the number of copy operations as well as
  // the number of data copies present simultaneously, since regions can be very
  // large.
  capnp::MallocMessageBuilder builder;
  builder.setRoot(pyRegionImplProto); // copy
  auto array = capnp::messageToFlatArray(builder); // copy
  py::String pyRegionImplBytes((const char *)array.begin(),
                               sizeof(capnp::word)*array.size()); // copy

  // Construct the python region instance by thunking into python

  // Wrap the operation in our _PyCapnpHelper class to simplify the interface
  // for the targer python region implementation
  py::Class regionCls(moduleName, className);

  py::Tuple args((Py_ssize_t)0);
  py::Dict kwargs;
  kwargs.setItem("pyRegionProtoBytes", pyRegionImplBytes);
  kwargs.setItem("regionCls", regionCls);
  kwargs.setItem("methodName", py::String("read"));

  py::Class pyCapnpHelperCls("nupic.bindings.engine_internal", "_PyCapnpHelper");

  PyObject* pyRegion = pyCapnpHelperCls.invoke("readPyRegion", args, kwargs);
  NTA_CHECK(pyRegion);
  return pyRegion;
#else
  throw std::logic_error(
    "Network::_readPyRegion is not implemented when compiled with CAPNP_LITE=1.");
#endif
}


void Network::write(NetworkProto::Builder& proto) const
{
  // Serialize C++ native data
  proto.setPyRegionModuleName(_pyRegionModuleName.c_str());
  proto.setPyRegionClassName(_pyRegionClassName.c_str());

  // Serialize the python region
  auto regionProto = proto.initRegion();
  regionProto.getRegionImpl().setAs<PyRegionProto>(_writePyRegion()); // copy
}


void Network::read(NetworkProto::Reader& proto)
{
  // Deserialize the C++ native data
  _pyRegionModuleName = proto.getPyRegionModuleName().cStr();
  _pyRegionClassName = proto.getPyRegionClassName().cStr();

  // Deserialize the python region
  _pyRegion.assign(_readPyRegion(_pyRegionModuleName,
                                 _pyRegionClassName,
                                 proto.getRegion()));
  NTA_CHECK(_pyRegion);
}


} // namespace nupic
