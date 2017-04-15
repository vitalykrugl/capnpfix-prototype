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

import capnp

from nupic.bindings.math import Random

from nupic.proto.RandomProto_capnp import RandomProto
from PythonRandomParent_capnp import PythonRandomParentProto


class PythonRandomParent(object):


  def __init__(self, width, seed):
    self.width = width
    self.rand = Random(seed)


  def write(self, proto):
    proto.width = self.width
    proto.random = RandomProto.from_bytes(self.rand.writeAsBytes())


  @classmethod
  def read(cls, proto):
    obj = object.__new__(cls)
    obj.width = proto.width
    obj.rand = Random.readFromBytes(proto.random.as_builder().to_bytes())

    return obj



class HybridSerializationTest(unittest.TestCase):


  def testPythonRandomParentWrite(self):
    srcObj = PythonRandomParent(width=10, seed=99)

    builderProto = PythonRandomParentProto.new_message()
    srcObj.write(builderProto)

    # NOTE proto.write requires a file descriptor
    with tempfile.TemporaryFile() as f:
      builderProto.write(f)
      f.seek(0)
      readerProto = PythonRandomParentProto.read(f)

    destObj = PythonRandomParent.read(readerProto)

    self.assertEqual(destObj.width, 10)
    self.assertEqual(destObj.rand.getSeed(), 99)




if __name__ == "__main__":
  unittest.main()
