//===------------ MachineStableHash.h - MIR Stable Hashing Utilities ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stable hashing for MachineInstr and MachineOperand. Useful or getting a
// hash across runs, modules, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINESTABLEHASH_H
#define LLVM_CODEGEN_MACHINESTABLEHASH_H

#include "llvm/ADT/StableHashing.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MachineBasicBlock;
class MachineFunction;
class MachineInstr;
class MachineOperand;

LLVM_ABI stable_hash stableHashValue(const MachineOperand &MO);
LLVM_ABI stable_hash stableHashValue(const MachineInstr &MI,
                                     bool HashVRegs = false,
                                     bool HashConstantPoolIndices = false,
                                     bool HashMemOperands = false);
LLVM_ABI stable_hash stableHashValue(const MachineBasicBlock &MBB);
LLVM_ABI stable_hash stableHashValue(const MachineFunction &MF);

} // namespace llvm

#endif
