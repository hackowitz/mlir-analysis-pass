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
  // Most of these will stay unknown() until they are removed
  if (isa<LLVM::AddOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::AddrSpaceCastOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::AllocaOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::AndOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::AShrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::AtomicRMWOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::BitcastOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::BlockAddressOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::BlockTagOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::BrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::CallOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::CallIntrinsicOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::AtomicCmpXchgOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ComdatOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ComdatSelectorOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::CondBrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::DSOLocalEquivalentOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ExtractElementOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ExtractValueOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FAddOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FCmpOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FDivOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FenceOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FMulOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FNegOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FPExtOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FPToSIOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FPToUIOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FPTruncOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FreezeOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FRemOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::FSubOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::LLVMFuncOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::GEPOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ICmpOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::IndirectBrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::InlineAsmOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::InsertElementOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::InsertValueOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::IntToPtrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::InvokeOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::LandingpadOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::LinkerOptionsOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::LoadOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::LShrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::AddressOfOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::AliasOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ConstantOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::GlobalOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::GlobalCtorsOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::GlobalDtorsOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::IFuncOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::MetadataAsValueOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::NoneTokenOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::PoisonOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::UndefOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ZeroOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ModuleFlagsOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::MulOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::NamedMetadataOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::OrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::PtrToAddrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::PtrToIntOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ResumeOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ReturnOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::SDivOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::SelectOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::SExtOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ShlOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ShuffleVectorOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::SIToFPOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::SRemOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::StoreOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::SubOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::SwitchOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::TruncOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::UDivOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::UIToFPOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::UnreachableOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::URemOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::VaArgOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::XOrOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  if (isa<LLVM::ZExtOp>(op)) {
    KnownBitsState lhs = operands[0]->getValue();
    return unknown();
  }
  return unknown();
}

} // namespace known_bits
