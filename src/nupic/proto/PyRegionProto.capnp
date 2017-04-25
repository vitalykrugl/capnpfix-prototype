@0xb36d5f0d0a026a2e;

# This is used to wrap region implementation proto objects so they can be
# passed between Python and C++. It is necessary because the C++ function to
# convert to a pycapnp PyObject takes a DynamicStruct builder/reader and
# creating these from an AnyPointer requires the schema while a known type
# does not. Wrapping the AnyPointer here gives us a known type.

struct PyRegionProto {
  regionImpl @0 :AnyPointer;
}
