//===- KnownBitsAnalysis.h - Sparse forward analysis over KnownBitsState
//------------===//

#ifndef KNOWN_BITS_ANALYSIS_H
#define KNOWN_BITS_ANALYSIS_H

#include "KnownBitsDomain.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace known_bits {

using KnownBitsLattice = mlir::dataflow::Lattice<KnownBitsState>;

class KnownBitsAnalysis : public mlir::dataflow::SparseForwardDataFlowAnalysis<KnownBitsLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// Transfer function: given the states of `op`'s operands, set the states of
  /// its results.  Must be monotone in the operand states.
  mlir::LogicalResult visitOperation(
      mlir::Operation *op,
      llvm::ArrayRef<const KnownBitsLattice *> operands,
      llvm::ArrayRef<KnownBitsLattice *> results
  ) override;

  /// The state of anything entering the analysis from outside: function
  /// arguments, and results the transfer function declines to reason about.
  void setToEntryState(KnownBitsLattice *lattice) override;
};

} // namespace known_bits

#endif
