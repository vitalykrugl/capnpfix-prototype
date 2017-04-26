# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import tempfile
import unittest

from nupic.bindings.engine_internal import Network as CppNetwork
from nupic.bindings.math import Random

# NOTE need to import capnp first to activate the magic necessary for
# PythonDummyRegion_capnp, etc.
import capnp
from PythonDummyRegion_capnp import PythonDummyRegionProto
from PythonDummyNetwork_capnp import PythonDummyNetworkProto


class PythonDummyRegion(object):
  """This class represents a serializable object that contains both native
  python properties as well as properties implemented in an extension.
  """

  def __init__(self, width, seed):
    # Arbitrary value that's compatible with UInt32 in the proto schema
    # for testing serialization of python-native property
    self.width = width

    # For testing serialization of object implemented in the extension
    self.rand = Random(seed)


  def write(self, proto):
    """ Serialize this instance into PyRegionProto builder. Emulates `PyRegion.write`.

    NOTE Called from nupic.bindings extension.

    :param proto: PyRegionProto builder
    """
    regionImpl = proto.regionImpl.as_struct(PythonDummyRegionProto)
    self.writeToProto(regionImpl)


  def writeToProto(self, proto):
    """ Serialize this instance into PythonDummyRegionProto builder

    :param proto: PythonDummyRegionProto builder
    """
    proto.width = self.width
    proto.random = self.rand.writeOut()


  @classmethod
  def read(cls, proto):
    """Deserialize from the given PyRegionProto reader. Emulates `PyRegion.read`.

    NOTE Called from nupic.bindings extension.

    :param proto: PyRegionProto reader

    :returns: Instance of PythonDummyRegion initialized from proto
    """
    regionImpl = proto.regionImpl.as_struct(PythonDummyRegionProto)
    return cls.readFromProto(regionImpl)


  @classmethod
  def readFromProto(cls, proto):
    """Deserialize from the given PythonDummyRegionProto reader

    :param proto: PythonDummyRegionProto reader

    :returns: Instance of PythonDummyRegion initialized from proto
    """
    obj = object.__new__(cls)
    obj.width = proto.width
    obj.rand = Random.readIn(proto.random)

    return obj



class PythonDummyNetwork(object):
  """This class represents a serializable object that is used for testing
  cyclical serialization using the Network class implemented in an extension.
  """

  def __init__(self, maxPhase, width, seed):
    # Arbitrary value that's compatible with UInt32 in the proto schema
    # for testing serialization of python-native property
    self.maxPhase = maxPhase

    # For testing cyclical serialization via "Network" implemented in the
    # extension
    self.cppNetwork = CppNetwork()
    self.cppNetwork.setPythonRegion(module=__name__,
                                    className="PythonDummyRegion",
                                    width=width, seed=seed)


  def writeToProto(self, proto):
    """ Serialize this instance into PythonDummyNetworkProto builder

    :param proto: PythonDummyNetworkProto builder
    """
    proto.phaseMax = self.maxPhase
    proto.extNetwork = self.cppNetwork.writeOut()


  @classmethod
  def readFromProto(cls, proto):
    """Deserialize from the given PythonDummyNetworkProto reader

    :param proto: PythonDummyNetworkProto reader

    :returns: Instance of PythonDummyNetwork initialized from proto
    """
    obj = object.__new__(cls)
    obj.maxPhase = proto.phaseMax
    obj.cppNetwork = CppNetwork.readIn(proto.extNetwork)

    return obj



class HybridSerializationTest(unittest.TestCase):


  def testPythonToExtensionWriteAndRead(self):
    # Test unidirectional python-to-extension serialization.

    srcObj = PythonDummyRegion(width=10, seed=99)

    # Serialize
    builderProto = PythonDummyRegionProto.new_message()
    srcObj.writeToProto(builderProto)

    # NOTE proto.write requires a file descriptor
    with tempfile.TemporaryFile() as f:
      builderProto.write(f)

      # Load serialized data into PythonDummyRegionProto reader
      f.seek(0)
      readerProto = PythonDummyRegionProto.read(f)

    # Deserialize
    destObj = PythonDummyRegion.readFromProto(readerProto)

    # Verify
    self.assertEqual(destObj.width, 10)
    self.assertEqual(destObj.rand.getSeed(), 99)


  def testCyclicalWriteAndRead(self):
    # Test cyclical serialization:

    srcNet = PythonDummyNetwork(maxPhase=5, width=10, seed=99)

    # Serialize
    builderProto = PythonDummyNetworkProto.new_message()
    srcNet.writeToProto(builderProto)

    # NOTE proto.write requires a file descriptor
    with tempfile.TemporaryFile() as f:
      builderProto.write(f)

      # Load serialized data into PythonDummyNetworkProto reader
      f.seek(0)
      readerProto = PythonDummyNetworkProto.read(f)

    # Deserialize
    destNet = PythonDummyNetwork.readFromProto(readerProto)

    # Verify
    self.assertEqual(destNet.maxPhase, 5)

    pyRegion = destNet.cppNetwork.getPythonRegion()
    self.assertEqual(pyRegion.width, 10)
    self.assertEqual(pyRegion.rand.getSeed(), 99)



if __name__ == "__main__":
  unittest.main()
