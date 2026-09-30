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

  // Only single-result integer operations are interesting here.
  // Calls, loads, floats, and vectors all land in `unknown`.
  if (op->getNumResults() != 1 || !op->getResult(0).getType().isIntOrIndex())
    return unknown();

  KnownBitsLattice *result = results[0];
  KnownBitsState known;
  auto nbits = op->getResult(0).getType().getIntOrFloatBitWidth();

  // The first few cases bring numeric data into our abstract domain;
  // without some rule of this kind the analysis would have no facts to propagate at all.
  IntegerAttr value;
  if (matchPattern(op, m_Constant(&value)))
    // A constant has all known bits.
    known = KnownBitsState::fromConstant(value.getValue());
  else if (isa<LLVM::ZeroOp>(op))
    // bits are all 0 not 1
    known = KnownBitsState::zero(nbits);
  else if (isa<LLVM::LoadOp>(op))
    // For a load op we at least know the bit width
    known = KnownBitsState::top(nbits);
  else if (op->getNumOperands()) {
    // Remaining cases are operations on abstract values, startign with unary opernads
    // Stupid sanity check - does anything ever not have operands?

    // exit early on unreachable code
    KnownBitsState lhs = operands[0]->getValue();
    if (lhs.isBottom())
      return success();
    else if (isa<LLVM::SExtOp>(op))
      known = lhs.signExtend(nbits);
    else if (isa<LLVM::ZExtOp>(op))
      known = lhs.zeroExtend(nbits);
    else if (isa<LLVM::AShrOp>(op))
      return unknown(); // I'm choosing to defer this one
    else if (isa<LLVM::TruncOp>(op))
      return unknown(); // TODO
    else if (isa<LLVM::CountLeadingZerosOp>(op))
      // There are plenty of edge cases I'm ignoring for this assignment...
      return unknown(); // TODO
    else if (op->getNumOperands() > 1) {
      // Now on to binary operations on abstract values

      // exit early on unreachable code
      KnownBitsState rhs = operands[1]->getValue();
      if (rhs.isBottom())
        return success();
      else if (isa<LLVM::AndOp>(op))
        known = lhs & rhs;
      else if (isa<LLVM::OrOp>(op))
        known = lhs | rhs;
      else if (isa<LLVM::XOrOp>(op))
        known = lhs ^ rhs;
      else if (isa<LLVM::ShlOp>(op))
        known = lhs << rhs;
      else if (isa<LLVM::LShrOp>(op))
        known = lhs >> rhs;
      else if (isa<LLVM::AddOp>(op))
        known = lhs + rhs;
      else if (isa<LLVM::SubOp>(op))
        known = lhs - rhs;
      else if (auto icmp = dyn_cast<LLVM::ICmpOp>(op)) {
        switch (icmp.getPredicate()) {
        case LLVM::ICmpPredicate::eq: // 5264 occurrences
          known = KnownBitsState::top(1);
          break;
        case LLVM::ICmpPredicate::ne: // 7806 occurrences
          known = KnownBitsState::top(1);
          break;
        case LLVM::ICmpPredicate::sge: //  557 occurrences
          known = KnownBitsState::top(1);
          break;
        case LLVM::ICmpPredicate::sgt: //  719 occurrences
          known = KnownBitsState::top(1);
          break;
        case LLVM::ICmpPredicate::sle: //  228 occurrences
          known = KnownBitsState::top(1);
          break;
        case LLVM::ICmpPredicate::slt: // 1195 occurrences
          known = KnownBitsState::top(1);
          break;
        case LLVM::ICmpPredicate::uge: //  132 occurrences
          known = lhs.uge(rhs);
          break;
        case LLVM::ICmpPredicate::ugt: //  224 occurrences
          known = lhs.ugt(rhs);
          break;
        case LLVM::ICmpPredicate::ule: //  166 occurrences
          known = lhs.ule(rhs);
          break;
        case LLVM::ICmpPredicate::ult: //  262 occurrences
          known = lhs.ult(rhs);
          break;
        default:
          // Over-approximate comparisons: last bit is unknown
          known = KnownBitsState::top(1);
        }
      }
      // TODO list, sorted by number of occurrences in sqlite3.
      // I'll go through these roughly in order as time allows
      else if (isa<LLVM::MulOp>(op)) //   179 llvm.mul
        return unknown();
      else if (isa<LLVM::SDivOp>(op)) //   70 llvm.sdiv
        return unknown();
      else if (isa<LLVM::UDivOp>(op)) //   48 llvm.udiv
        return unknown();
      else if (isa<LLVM::SRemOp>(op)) //   37 llvm.srem
        return unknown();
      else if (isa<LLVM::URemOp>(op)) //   22 llvm.urem
        return unknown();
      else if (isa<LLVM::SelectOp>(op)) //  3 llvm.select
        return unknown();
      else {
        llvm::dbgs() << "// Unknown binary op: " << *op << "\n";
        return unknown();
      }
    } else {
      llvm::dbgs() << "// Unknown unary op: " << *op << "\n";
      return unknown();
    }
  } else {
    llvm::dbgs() << "// No operands? Insane! " << *op << "\n";
    return unknown();
  }
  propagateIfChanged(result, result->join(known));
  return success();
}

} // namespace known_bits
