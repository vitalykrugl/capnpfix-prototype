@0xe1b7642b83ef342d;

using import "/nupic/proto/RandomProto.capnp".RandomProto;

struct PythonRandomParentProto {
  width @0 :UInt32;
  random @1 :RandomProto;
}
