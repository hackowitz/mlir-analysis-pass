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
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace known_bits {

void KnownBitsAnalysis::setToEntryState(KnownBitsLattice *lattice) {
  propagateIfChanged(lattice, lattice->join(KnownBitsState::bottom()));
}

LogicalResult KnownBitsAnalysis::visitOperation(
    Operation *op,                               //
    ArrayRef<const KnownBitsLattice *> operands, // 1 for unary, 2 for binary op
    ArrayRef<KnownBitsLattice *> results         // result->join(state) then propagateIfChanged
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

  // Transfer functions into our abstract domain;
  // without some rule of this kind the analysis would have no facts to propagate at all.
  IntegerAttr value;
  KnownBitsState known;
  if (matchPattern(op, m_Constant(&value))) {
    // A constant has all known bits.
    known = KnownBitsState::fromConstant(value.getValue());
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }
  if (isa<LLVM::ZeroOp>(op)) {
    // bits are all 0 not 1
    known = KnownBitsState(op->getResult(0).getType().getIntOrFloatBitWidth(), ~0, 0);
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }
  if (isa<LLVM::LoadOp>(op)) {
    // For a load op we at least know the bit width
    known = KnownBitsState::top(op->getResult(0).getType().getIntOrFloatBitWidth());
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }

  // Match the LLVM dialect operation, coopy/pasted from https://mlir.llvm.org/docs/Dialects/LLVM/
  // Start with the simplest: bitwise/binary operations
  if (isa<LLVM::AndOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    KnownBitsState rhs = operands[1]->getValue();
    if (lhs.isBottom() || rhs.isBottom()) {
      return success();
    }
    KnownBitsState known = lhs & rhs;
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }
  if (isa<LLVM::OrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    KnownBitsState rhs = operands[1]->getValue();
    if (lhs.isBottom() || rhs.isBottom()) {
      return success();
    }
    KnownBitsState known = lhs | rhs;
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }
  if (isa<LLVM::XOrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    KnownBitsState rhs = operands[1]->getValue();
    if (lhs.isBottom() || rhs.isBottom()) {
      return success();
    }
    KnownBitsState known = lhs | rhs;
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }
  if (isa<LLVM::ShlOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    KnownBitsState rhs = operands[1]->getValue();
    if (lhs.isBottom() || rhs.isBottom()) {
      return success();
    }
    KnownBitsState known = lhs << rhs;
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }
  if (isa<LLVM::LShrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    KnownBitsState rhs = operands[1]->getValue();
    if (lhs.isBottom() || rhs.isBottom()) {
      return success();
    }
    KnownBitsState known = lhs >> rhs;
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }
  if (isa<LLVM::AShrOp>(op))
    return unknown(); // I'm choosing to defer this one
  if (isa<LLVM::SExtOp>(op)) {
    auto nbits = op->getResult(0).getType().getIntOrFloatBitWidth();
    KnownBitsState lhs = operands[0]->getValue();
    if (lhs.isBottom()) {
      return success();
    }
    KnownBitsState known = lhs.signExtend(nbits);
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }
  if (isa<LLVM::ZExtOp>(op)) {
    auto nbits = op->getResult(0).getType().getIntOrFloatBitWidth();
    KnownBitsState lhs = operands[0]->getValue();
    if (lhs.isBottom()) {
      return success();
    }
    KnownBitsState known = lhs.zeroExtend(nbits);
    llvm::dbgs() << "// " << *op << "\n//\t" << known << "\n";
    propagateIfChanged(result, result->join(known));
    return success();
  }

  // // Arithmetic operations are harder but definitely possible
  // if (isa<LLVM::AddOp>(op)) {
  //   return unknown();
  // }
  // if (isa<LLVM::MulOp>(op)) {
  //   return unknown();
  // }
  // if (isa<LLVM::SDivOp>(op)) {
  //   return unknown();
  // }
  // if (isa<LLVM::SubOp>(op)) {
  //   return unknown();
  // }
  // if (isa<LLVM::TruncOp>(op)) {
  //   return unknown();
  // }
  // if (isa<LLVM::UDivOp>(op)) {
  //   return unknown();
  // }

  return unknown();
}

} // namespace known_bits
