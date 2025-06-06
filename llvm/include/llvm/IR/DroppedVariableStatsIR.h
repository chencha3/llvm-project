///===- DroppedVariableStatsIR.h - Opt Diagnostics -*- C++ -*--------------===//
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM
/// Exceptions. See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===---------------------------------------------------------------------===//
/// \file
/// Dropped Variable Statistics for Debug Information. Reports any number
/// of #dbg_value that get dropped due to an optimization pass.
///
///===---------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DROPPEDVARIABLESTATSIR_H
#define LLVM_CODEGEN_DROPPEDVARIABLESTATSIR_H

#include "llvm/IR/DroppedVariableStats.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Any;
class StringRef;
class PassInstrumentationCallbacks;
class Function;
class Module;
class DILocation;

/// A class to collect and print dropped debug information due to LLVM IR
/// optimization passes. After every LLVM IR pass is run, it will print how many
/// #dbg_values were dropped due to that pass.
class LLVM_ABI DroppedVariableStatsIR : public DroppedVariableStats {
public:
  DroppedVariableStatsIR(bool DroppedVarStatsEnabled)
      : llvm::DroppedVariableStats(DroppedVarStatsEnabled) {}

  void runBeforePass(StringRef P, Any IR);

  void runAfterPass(StringRef P, Any IR);

  void registerCallbacks(PassInstrumentationCallbacks &PIC);

private:
  const Function *Func;

  void runAfterPassFunction(StringRef PassID, const Function *F);

  void runAfterPassModule(StringRef PassID, const Module *M);

  /// Populate DebugVariablesBefore, DebugVariablesAfter, InlinedAts before or
  /// after a pass has run to facilitate dropped variable calculation for an
  /// llvm::Function.
  void runOnFunction(StringRef PassID, const Function *F, bool Before);

  /// Iterate over all Instructions in a Function and report any dropped debug
  /// information.
  void calculateDroppedVarStatsOnFunction(const Function *F, StringRef PassID,
                                          StringRef FuncOrModName,
                                          StringRef PassLevel);

  /// Populate DebugVariablesBefore, DebugVariablesAfter, InlinedAts before or
  /// after a pass has run to facilitate dropped variable calculation for an
  /// llvm::Module. Calls runOnFunction on every Function in the Module.
  void runOnModule(StringRef PassID, const Module *M, bool Before);

  /// Iterate over all Functions in a Module and report any dropped debug
  /// information. Will call calculateDroppedVarStatsOnFunction on every
  /// Function.
  void calculateDroppedVarStatsOnModule(const Module *M, StringRef PassID,
                                        StringRef FuncOrModName,
                                        StringRef PassLevel);

  /// Override base class method to run on an llvm::Function specifically.
  virtual void
  visitEveryInstruction(unsigned &DroppedCount,
                        DenseMap<VarID, DILocation *> &InlinedAtsMap,
                        VarID Var) override;

  /// Override base class method to run on #dbg_values specifically.
  virtual void visitEveryDebugRecord(
      DenseSet<VarID> &VarIDSet,
      DenseMap<StringRef, DenseMap<VarID, DILocation *>> &InlinedAtsMap,
      StringRef FuncName, bool Before) override;

  template <typename IRUnitT> static const IRUnitT *unwrapIR(Any IR);
};

} // namespace llvm

#endif
