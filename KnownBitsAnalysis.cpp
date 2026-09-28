//===- KnownBitsAnalysis.cpp - Transfer functions
//------------------------------===//
//
// The transfer function: given what is known about an operation's operands,
// state what is known about its results.  This file and KnownBitsDomain.h are
// the two to replace when building a different analysis; the rest of the
// project is scaffolding.
//
// There are deliberately only two rules here, one of each kind an analysis
// needs: one that introduces facts out of nothing (constants), and one that
// propagates facts it was given (`and`).  Everything else is unknown.  Adding
// a third rule should be a matter of adding a third `if`.
//
//===----------------------------------------------------------------------===//

#include "KnownBitsAnalysis.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace known_bits {

void KnownBitsAnalysis::setToEntryState(KnownBitsLattice *lattice) {
  propagateIfChanged(lattice, lattice->join(KnownBitsState::top()));
}

LogicalResult KnownBitsAnalysis::visitOperation(
    Operation *op, ArrayRef<const KnownBitsLattice *> operands, ArrayRef<KnownBitsLattice *> results
) {
  // Raising a result to top says "this operation could produce anything",
  // which is always a sound answer and is what every unhandled case does.
  auto unknown = [&] {
    setAllToEntryStates(results);
    return success();
  };

  // Only single-result integer operations are interesting here.  Calls, loads,
  // floats, and vectors all land in `unknown`.
  if (op->getNumResults() != 1 || !op->getResult(0).getType().isIntOrIndex())
    return unknown();
  KnownBitsLattice *result = results[0];

  // Rule 1: a constant is zero or nonzero according to what it says.
  // This is the only rule that does not consult its operands, and without some
  // rule of this kind the analysis would have no facts to propagate at all.
  IntegerAttr value;
  if (matchPattern(op, m_Constant(&value))) {
    KnownBitsState state = value.getValue().isZero() ? Kind::Zero : Kind::NonZero;
    propagateIfChanged(result, result->join(state));
    return success();
  }

  // Rule 2: `x & y` is zero if either operand is zero, since a zero operand
  // clears every bit.  Note what this rule does *not* say: two nonzero
  // operands tell us nothing, because 1 & 2 is 0.
  if (isa<LLVM::AndOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    KnownBitsState rhs = operands[1]->getValue();

    // Bottom means the solver has not yet proved anything reaches this
    // operand.  Leaving the result alone keeps the analysis optimistic; the
    // solver will call back here once the operand moves up the lattice.
    if (lhs.isBottom() || rhs.isBottom())
      return success();

    if (lhs.kind == Kind::Zero || rhs.kind == Kind::Zero) {
      propagateIfChanged(result, result->join(KnownBitsState(Kind::Zero)));
      return success();
    }
  }

  return unknown();
}

} // namespace known_bits
