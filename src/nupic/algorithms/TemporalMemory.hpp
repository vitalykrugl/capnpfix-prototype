/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2016, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ----------------------------------------------------------------------
 */

/** @file
 * Definitions for the Temporal Memory in C++
 */

#ifndef NTA_TEMPORAL_MEMORY_HPP
#define NTA_TEMPORAL_MEMORY_HPP

#include <vector>
#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Random.hpp>

#include <nupic/proto/TemporalMemoryProto.capnp.h>

using namespace std;
using namespace nupic;

namespace nupic {
  namespace algorithms {
    namespace temporal_memory {

      /**
       * Temporal Memory implementation in C++.
       *
       * Example usage:
       *
       *     SpatialPooler sp(inputDimensions, columnDimensions, <parameters>);
       *     TemporalMemory tm(columnDimensions, <parameters>);
       *
       *     while (true) {
       *        <get input vector, streaming spatiotemporal information>
       *        sp.compute(inputVector, learn, activeColumns)
       *        tm.compute(number of activeColumns, activeColumns, learn)
       *        <do something with the tm, e.g. classify tm.getActiveCells()>
       *     }
       *
       * The public API uses C arrays, not std::vectors, as inputs. C arrays are
       * a good lowest common denominator. You can get a C array from a vector,
       * but you can't get a vector from a C array without copying it. This is
       * important, for example, when using numpy arrays. The only way to
       * convert a numpy array into a std::vector is to copy it, but you can
       * access a numpy array's internal C array directly.
       */
      class TemporalMemory : public Serializable<TemporalMemoryProto> {
      public:
        TemporalMemory();

        /**
         * Initialize the temporal memory (TM) using the given parameters.
         *
         * @param cellsPerColumn
         * Number of cells per column
         *
         * @param activationThreshold
         * If the number of active connected synapses on a segment is at least
         * this threshold, the segment is said to be active.
         *
         * @param initialPermanence
         * Initial permanence of a new synapse.
         *
         * @param connectedPermanence
         * If the permanence value for a synapse is greater than this value, it
         * is said to be connected.
         *
         * @param minThreshold
         * If the number of potential synapses active on a segment is at least
         * this threshold, it is said to be "matching" and is eligible for
         * learning.
         *
         * @param maxNewSynapseCount
         * The maximum number of synapses added to a segment during learning.
         *
         * @param permanenceIncrement
         * Amount by which permanences of synapses are incremented during
         * learning.
         *
         * @param permanenceDecrement
         * Amount by which permanences of synapses are decremented during
         * learning.
         *
         * @param predictedSegmentDecrement
         * Amount by which segments are punished for incorrect predictions.
         *
         * @param seed
         * Seed for the random number generator.
         *
         * @param maxSegmentsPerCell
         * The maximum number of segments per cell.
         *
         * @param maxSynapsesPerSegment
         * The maximum number of synapses per segment.
         *
         * Notes:
         *
         * predictedSegmentDecrement: A good value is just a bit larger than
         * (the column-level sparsity * permanenceIncrement). So, if column-level
         * sparsity is 2% and permanenceIncrement is 0.01, this parameter should be
         * something like 4% * 0.01 = 0.0004).
         */
        TemporalMemory(
          unsigned cellsPerColumn,
          int seed = 42);

        virtual void initialize(
          unsigned cellsPerColumn = 32,
          int seed = 42);

        virtual ~TemporalMemory();

        //----------------------------------------------------------------------
        //  Main functions
        //----------------------------------------------------------------------

        /**
         * Returns the number of cells per column.
         *
         * @returns Integer number of cells per column
         */
        unsigned getCellsPerColumn() const;

        /**
         * Get the version number of for the TM implementation.
         *
         * @returns Integer version number.
         */
        virtual unsigned version() const;

        /**
         * This *only* updates _rng to a new Random using seed.
         *
         * @returns Integer version number.
         */
        void seed_(unsigned long seed);


        // ==============================
        //  Helper functions
        // ==============================


        using Serializable::write;
        virtual void write(TemporalMemoryProto::Builder& proto) const override;

        using Serializable::read;
        virtual void read(TemporalMemoryProto::Reader& proto) override;

      protected:
        UInt cellsPerColumn_;

        Random rng_;
      };

    } // end namespace temporal_memory
  } // end namespace algorithms
} // end namespace nta

#endif // NTA_TEMPORAL_MEMORY_HPP
