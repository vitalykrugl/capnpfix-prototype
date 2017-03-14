@0xc5bf8243b0c10764;

using import "/nupic/proto/RandomProto.capnp".RandomProto;

# Next ID: 3
struct TemporalMemoryProto {

  cellsPerColumn @0 :UInt32;

  random @1 :RandomProto;
}
