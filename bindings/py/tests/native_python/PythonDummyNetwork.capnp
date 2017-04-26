@0x9dc095996ceba1c1;

using import "/nupic/proto/NetworkProto.capnp".NetworkProto;

struct PythonDummyNetworkProto {
  phaseMax @0 :UInt32;
  extNetwork @1 :NetworkProto;
}
