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

  // A constant has all known bits. This is the transfer function into our abstract domain;
  // without some rule of this kind the analysis would have no facts to propagate at all.
  IntegerAttr value;
  if (matchPattern(op, m_Constant(&value))) {
    propagateIfChanged(result, result->join(KnownBitsState::fromConstant(value.getValue())));
    return success();
  }

  // Match the LLVM dialect operation, coopy/pasted from https://mlir.llvm.org/docs/Dialects/LLVM/
  // Start with the simplest: bitwise/binary operations
  if (isa<LLVM::AndOp>(op)) {
    propagateIfChanged(result, result->join(operands[0]->getValue() & operands[1]->getValue()));
    return success();
  }
  if (isa<LLVM::OrOp>(op)) {
    propagateIfChanged(result, result->join(operands[0]->getValue() | operands[1]->getValue()));
    return success();
  }
  if (isa<LLVM::XOrOp>(op)) {
    propagateIfChanged(result, result->join(operands[0]->getValue() | operands[1]->getValue()));
    return success();
  }
  if (isa<LLVM::ShlOp>(op)) {
    propagateIfChanged(result, result->join(operands[0]->getValue() << operands[1]->getValue()));
    return success();
  }
  if (isa<LLVM::LShrOp>(op)) {
    propagateIfChanged(result, result->join(operands[0]->getValue() >> operands[1]->getValue()));
    return success();
  }
  if (isa<LLVM::AShrOp>(op))
    return unknown(); // I'm choosing to defer this one

  if (isa<LLVM::SExtOp>(op)) {
    auto nbits = op->getResult(0).getType().getIntOrFloatBitWidth();
    propagateIfChanged(result, result->join(operands[0]->getValue().signExtend(nbits)));
    return success();
  }
  if (isa<LLVM::ZExtOp>(op)) {
    auto nbits = op->getResult(0).getType().getIntOrFloatBitWidth();
    propagateIfChanged(result, result->join(nbits));
    return success();
  }

  // Arithmetic operations are harder but definitely possible
  if (isa<LLVM::AddOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ZeroOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::MulOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::SDivOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::SubOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::TruncOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::UDivOp>(op)) {
    return unknown();
  }

  // Floating point arithmetic is possible but I won't get to it
  if (isa<LLVM::FAddOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FRemOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FSubOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FCmpOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FDivOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FenceOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FMulOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FNegOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FPExtOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FPToSIOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FPToUIOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FPTruncOp>(op)) {
    return unknown();
  }

  // Most of these will stay unknown() until they are removed
  if (isa<LLVM::AddrSpaceCastOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::AllocaOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::AtomicRMWOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::BitcastOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::BlockAddressOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::BlockTagOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::BrOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::CallOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::CallIntrinsicOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::AtomicCmpXchgOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ComdatOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ComdatSelectorOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::CondBrOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::DSOLocalEquivalentOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ExtractElementOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ExtractValueOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::FreezeOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::LLVMFuncOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::GEPOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ICmpOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::IndirectBrOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::InlineAsmOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::InsertElementOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::InsertValueOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::IntToPtrOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::InvokeOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::LandingpadOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::LinkerOptionsOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::LoadOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::AddressOfOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::AliasOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ConstantOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::GlobalOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::GlobalCtorsOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::GlobalDtorsOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::IFuncOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::MetadataAsValueOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::NoneTokenOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::PoisonOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::UndefOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ModuleFlagsOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::NamedMetadataOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::PtrToAddrOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::PtrToIntOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ResumeOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ReturnOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::SelectOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::ShuffleVectorOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::SIToFPOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::SRemOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::StoreOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::SwitchOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::UIToFPOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::UnreachableOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::URemOp>(op)) {
    return unknown();
  }
  if (isa<LLVM::VaArgOp>(op)) {
    return unknown();
  }
  return unknown();
}

} // namespace known_bits
