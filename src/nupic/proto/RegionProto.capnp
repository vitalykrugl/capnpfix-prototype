@0xab0b6bd212b9b640;


struct RegionProto {
  # This stores the data for the RegionImpl. This will be a PyRegionProto
  # instance if it is a PyRegion.
  regionImpl @0 :AnyPointer;
}