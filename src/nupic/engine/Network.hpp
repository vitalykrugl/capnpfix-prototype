#ifndef NTA_NETWORK_HPP
#define NTA_NETWORK_HPP

#include <string>

#include <Python.h>

#include <nupic/proto/NetworkProto.capnp.h>
#include <nupic/proto/PyRegionProto.capnp.h>
#include <nupic/proto/RegionProto.capnp.h>

#include <nupic/py_support/PyHelpers.hpp>
#include <nupic/types/Serializable.hpp>


namespace nupic
{
  /**
   * This Network class is used for prototyping capnp-based serialization tasks
   * that would normally be divided among Network, RegionImpl, and PyRegion
   * classes in an actual nupic.core-based network.
   */
  class Network : public nupic::Serializable<NetworkProto>
  {
  public:
    Network();

    virtual ~Network();

    std::string getPythonRegionClassName();

    void setPythonRegion(const char* module, const char* className,
                         unsigned long width, unsigned long seed);

    PyObject* getPythonRegion();

    using Serializable::write;
    void write(NetworkProto::Builder& proto) const;
    using Serializable::read;
    void read(NetworkProto::Reader& proto);
  private:
    PyRegionProto::Reader _writePyRegion() const;
    static PyObject* _readPyRegion(const std::string& moduleName,
                                   const std::string& className,
                                   const RegionProto::Reader& proto);

    std::string _pyRegionModuleName;
    std::string _pyRegionClassName;
    py::Instance _pyRegion;
  };
} // namespace nupic

#endif // NTA_NETWORK_HPP
