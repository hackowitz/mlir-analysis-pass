//===- Plugin.cpp - Pass definition and plugin entry point ----------------===//
//
// Scaffolding: wires the analysis into a pass and exposes it to mlir-opt.
//
//===----------------------------------------------------------------------===//

#include "Annotate.h"
#include "KnownBitsAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct KnownBitsAnalysisPass : PassWrapper<KnownBitsAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KnownBitsAnalysisPass)

  StringRef getArgument() const final { return "known-bits-analysis"; }

  StringRef getDescription() const final {
    return "Determine which bits are known zero or known nonzero";
  }

  void runOnOperation() override {
    DataFlowConfig config;
    config.setInterprocedural(false);

    DataFlowSolver solver(config);
    // DeadCodeAnalysis supplies reachability, without which the solver must
    // assume every branch is taken; SparseConstantPropagation resolves branch
    // conditions for it.  Both are prerequisites, not extras.
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<known_bits::KnownBitsAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation().emitError("known bits analysis failed to reach a fixed point");
      return signalPassFailure();
    }

    struct Target {
      Value value;
      known_bits::KnownBitsState state;
    };
    SmallVector<Target> targets;

    // Two loops: one to find targets and one to modify them, since we can't modify the iterator
    // while looping through it
    getOperation()->walk([&](Operation *op) {
      for (Value value : op->getResults()) {
        const auto *lattice = solver.lookupState<known_bits::KnownBitsLattice>(value);
        if (lattice && lattice->getValue().isInteresting())
          targets.push_back({value, lattice->getValue()});
      }
    });
    for (const auto &[value, state] : targets) {
      Location loc = value.getDefiningOp()->getLoc();
      Type type = value.getType();
      OpBuilder builder(value.getContext());

      // New control flow starting at the value to test. The check runs like:
      // ex: if ((value ^ expected) & mask != 0) panic()
      builder.setInsertionPointAfter(value.getDefiningOp());
      Block *block = builder.getBlock();
      Block *okBlock = block->splitBlock(builder.getInsertionPoint());
      Block *failBlock = builder.createBlock(okBlock);
      builder.setInsertionPointToEnd(block);
      // ex:     (value ^ expected)
      Value expected = LLVM::ConstantOp::create(
          builder, loc, type, builder.getIntegerAttr(type, state.ones & ~state.zeroes)
      );
      Value diff = LLVM::XOrOp::create(builder, loc, value, expected);
      // ex:     .................. & mask
      Value mask = LLVM::ConstantOp::create(
          builder, loc, type, builder.getIntegerAttr(type, state.zeroes ^ state.ones)
      );
      Value mismatch = LLVM::AndOp::create(builder, loc, diff, mask);
      // ex:    (......................... != 0)
      Value zero = LLVM::ConstantOp::create(builder, loc, type, builder.getIntegerAttr(type, 0));
      Value fail = LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne, mismatch, zero);
      // ex: if ................................
      LLVM::CondBrOp::create(builder, loc, fail, failBlock, okBlock);
      // ex: ................................... panic()
      builder.setInsertionPointToEnd(failBlock);
      LLVM::UnreachableOp::create(builder, loc);
    }

    //   // Stderr, so that mlir-opt's stdout stays the unmodified IR and the two can
    //   // be redirected independently.
    //   known_bits::printAnnotated(
    //       getOperation(),
    //       [&](Value value, AsmState &asmState) -> std::string {
    //         const auto *lattice = solver.lookupState<known_bits::KnownBitsLattice>(value);
    //         std::string description;
    //         if (lattice) {
    //           llvm::raw_string_ostream os(description);
    //           auto state = lattice->getValue();
    //           if (state.isInteresting())
    //             os << "known bits: " << state;
    //         }
    //         return description;
    //       },
    //       llvm::errs()
    //   );
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  // LLVM_VERSION_STRING is baked in at compile time and checked by mlir-opt at
  // load time, which is what turns an ABI mismatch into a clear diagnostic.
  return {MLIR_PLUGIN_API_VERSION, "KnownBitsAnalysis", LLVM_VERSION_STRING, []() {
            PassRegistration<KnownBitsAnalysisPass>();
          }};
}
