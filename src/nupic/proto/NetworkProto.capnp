@0xab1dba0bf5b97202;

using import "/nupic/proto/RegionProto.capnp".RegionProto;

struct NetworkProto {
  pyRegionModuleName @0 : Text;
  pyRegionClassName @1 : Text;
  region @2 : RegionProto;
}