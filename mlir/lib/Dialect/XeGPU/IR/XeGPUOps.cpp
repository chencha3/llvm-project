//===- XeGPUOps.cpp - MLIR XeGPU ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xegpu"

namespace mlir {
namespace xegpu {

static void transpose(llvm::ArrayRef<int64_t> trans,
                      std::vector<int64_t> &shape) {
  std::vector<int64_t> old = shape;
  for (size_t i = 0; i < trans.size(); i++)
    shape[i] = old[trans[i]];
}

template <typename T>
static std::string makeString(T array, bool breakline = false) {
  std::string buf;
  buf.clear();
  llvm::raw_string_ostream os(buf);
  os << "[";
  for (size_t i = 1; i < array.size(); i++) {
    os << array[i - 1] << ", ";
    if (breakline)
      os << "\n\t\t";
  }
  os << array.back() << "]";
  os.flush();
  return buf;
}

static std::vector<int64_t> getShapeOf(Type type) {
  std::vector<int64_t> shape;
  if (llvm::isa<VectorType>(type))
    shape = llvm::dyn_cast<VectorType>(type).getShape().vec();
  else if (type.isIntOrIndexOrFloat())
    shape.push_back(1);
  else 
    llvm_unreachable("Unsupported type.");
  return shape;
};

static bool isValidReadHintOrNone(const CachePolicyAttr &attr) {
    if (!attr)
      return true;
    auto kind = attr.getValue();
    return kind == CachePolicy::CACHED || 
           kind == CachePolicy::UNCACHED ||
           kind == CachePolicy::STREAMING || 
           kind == CachePolicy::READ_INVALIDATE;
  };

//===----------------------------------------------------------------------===//
// XeGPU_CreateNdDescOp
//===----------------------------------------------------------------------===//
void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<MemRefType> source,
                           llvm::ArrayRef<OpFoldResult> offsets) {
  [[maybe_unused]] auto ty = source.getType();
  assert(ty.hasStaticShape() && offsets.size() == (size_t)ty.getRank());

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, tdesc, source, dynamicOffsets /* dynamic offsets */,
        ValueRange({}) /* empty dynamic shape */,
        ValueRange({}) /* empty dynamic strides */,
        staticOffsets /* const offsets */, {} /* empty const shape*/,
        {} /* empty const strides*/);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<IntegerType> source,
                           llvm::ArrayRef<OpFoldResult> offsets,
                           llvm::ArrayRef<OpFoldResult> shape,
                           llvm::ArrayRef<OpFoldResult> strides) {
  assert(shape.size() && offsets.size() && strides.size() &&
         shape.size() == strides.size() && shape.size() == offsets.size());

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<int64_t> staticShape;
  llvm::SmallVector<int64_t> staticStrides;
  llvm::SmallVector<Value> dynamicOffsets;
  llvm::SmallVector<Value> dynamicShape;
  llvm::SmallVector<Value> dynamicStrides;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticOffsets);

  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);
  auto staticShapeAttr = builder.getDenseI64ArrayAttr(staticShape);
  auto staticStridesAttr = builder.getDenseI64ArrayAttr(staticStrides);

  build(builder, state, tdesc, source, dynamicOffsets, dynamicShape,
        dynamicStrides, staticOffsetsAttr, staticShapeAttr, staticStridesAttr);
}

LogicalResult CreateNdDescOp::verify() {
  auto rank = (int64_t)getMixedOffsets().size();
  bool invalidRank = (rank != 2);
  bool invalidElemTy = false;

  // check source type matches the rank if it is a memref.
  // It also should have the same ElementType as TensorDesc.
  auto memrefTy = getSourceType().dyn_cast<MemRefType>();
  if (memrefTy) {
    invalidRank |= (memrefTy.getRank() != rank);
    invalidElemTy |= memrefTy.getElementType() != getElementType();
  }

  // check result type matches the rank
  invalidRank = (getType().getRank() != rank);

  // mismatches among shape, strides, and offsets are
  // already handeled by OffsetSizeAndStrideOpInterface.
  // So they are not check here.
  if (invalidRank)
    return emitOpError(
        "Expecting the rank of shape, strides, offsets, "
        "source memref type (if source is a memref) and TensorDesc "
        "should match with each other. They currenlty are 2D.");

  if (invalidElemTy)
    return emitOpError("TensorDesc should have the same element "
                       "type with the source if it is a memref.\n");

  if (getType().getScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchNdOp
//===----------------------------------------------------------------------===//
LogicalResult PrefetchNdOp::verify() {
  auto tdescTy = getTensorDescType();
  if (tdescTy.getScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadNdOp
//===----------------------------------------------------------------------===//
LogicalResult LoadNdOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getType();

  if (tdescTy.getRank() != 2)
    return emitOpError("Expecting a 2D TensorDesc.\n");

  if (tdescTy.getScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (!valueTy)
    return emitOpError("Invalid result, it should be a VectorType.\n");

  auto tdescElemTy = tdescTy.getElementType();
  auto valueElemTy = valueTy.getElementType();

  if (tdescElemTy != valueElemTy)
    return emitOpError("Value should have the same element type as TensorDesc.");

  auto array_len = tdescTy.getArrayLength();
  auto tdescShape = tdescTy.getShape().vec();
  auto valueShape = valueTy.getShape().vec();

  if (getTranspose()) {
    auto trans = getTranspose().value();
    if (tdescShape.size() >= trans.size())
      transpose(trans, tdescShape);
    else
      emitWarning("Invalid transpose attr. It is ignored.");
  }

  if (getVnniAxis()) {
    auto axis = getVnniAxis().value();
    auto vnni_factor = valueShape.back();
    tdescShape[axis] /= vnni_factor;
    tdescShape.push_back(vnni_factor);
  }

  if (array_len > 1) {
    auto it = tdescShape.begin();
    tdescShape.insert(it, array_len);
  }

  if (tdescShape != valueShape)
    return emitOpError() << "Result shape doesn't match TensorDesc shape."
                         << "The expected shape is " << makeString(tdescShape)
                         << ". But the given shape is "
                         << makeString(valueShape) << ".\n";
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreNdOp
//===----------------------------------------------------------------------===//
LogicalResult StoreNdOp::verify() {
  auto dstTy = getTensorDescType();               // Tile
  auto valTy = getValueType();                    // Vector

  if (dstTy.getRank() != 2)
    return emitOpError("Expecting a 2D TensorDesc.\n");

  if (dstTy.getScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (!valTy)
    return emitOpError("Exepcting a VectorType result.\n");

  auto dstElemTy = dstTy.getElementType();
  auto valElemTy = valTy.getElementType();

  if (dstElemTy != valElemTy)
    return emitOpError("Value should have the same element type as TensorDesc.");

  if (dstTy.getShape() != valTy.getShape())
    return emitOpError("Value should have the same shape as TensorDesc.\n");
  return success();
}


//===----------------------------------------------------------------------===//
// XeGPU_CreateDescOp
//===----------------------------------------------------------------------===//
void CreateDescOp::build(OpBuilder &builder, OperationState &state,
                         TensorDescType TensorDesc, Value source, 
                         llvm::ArrayRef<OpFoldResult> offsets, uint32_t chunk_size) {
  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  build(builder, state, TensorDesc, source, dynamicOffsets, staticOffsets, chunk_size);
}

LogicalResult CreateDescOp::verify() {
  auto tdescTy = getTensorDescType();
  auto chunkSize = getChunkSize();

  auto getRankOf = [&](Value val){
    auto type = val.getType();
    if (auto ty = llvm::dyn_cast<MemRefType>(type))
      return ty.getRank();
    return (int64_t)0;
  };

  if (getRankOf(getSource()) > 2)
    return emitOpError("Expecting the source is a 1D memref or pointer (uint64_t).");

  if (!tdescTy.getScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  std::vector<int64_t> shape({getNumOffsets()});
  if (chunkSize != 1)
    shape.push_back(chunkSize);

  auto tdescShape = tdescTy.getShape();
  if (shape != tdescShape.vec()) {
    return emitOpError("Expecting dimensions of offsets is the same as the "
                       "tensor descriptor, or one less than.");
  }

  return success();
}


//===----------------------------------------------------------------------===//
// XeGPU_PrefetchOp
//===----------------------------------------------------------------------===//
LogicalResult PrefetchOp::verify() {
  auto tdescTy = getTensorDescType();



  if (!tdescTy.getScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  if (!isValidHint(getL1HintAttr()))
    return emitOpError("invlid l1_hint: ") << getL1HintAttr();

  if (!isValidHint(getL2HintAttr()))
    return emitOpError("invlid l2_hint: ") << getL2HintAttr();

  if (!isValidHint(getL3HintAttr()))
    return emitOpError("invlid l3_hint: ") << getL3HintAttr();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadGatherOp
//===----------------------------------------------------------------------===//
LogicalResult LoadGatherOp::verify() {
  auto tdescTy = getTensorDescType();
  auto maskTy = getMask().getType();
  auto valueTy = getValue().getType();

  if (!tdescTy.getScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  auto tdescElemTy = tdescTy.getElementType();
  auto valueElemTy = getElementType();
  if (tdescElemTy != valueElemTy)
    return emitOpError(
        "Value should have the same element type as TensorDesc.");

  std::vector<int64_t> maskShape = getShapeOf(maskTy);
  std::vector<int64_t> valueShape = getShapeOf(valueTy);
  std::vector<int64_t> tdescShape = tdescTy.getShape().vec();

  if (tdescShape != maskShape)
    return emitOpError("Mask should have the same shape as TensorDesc.");

  if (getTransposeAttr()) {
    auto trans = getTranspose().value();
    if (tdescShape.size() < trans.size())
      return emitWarning("Invalid transpose attr. It is ignored.");
    transpose(trans, tdescShape);
  }

  if (getVnniAxis()) {
    auto axis = getVnniAxis().value();
    auto vnni_factor = valueShape.back();
    tdescShape[axis] /= vnni_factor;
    tdescShape.push_back(vnni_factor);
  }

  if (valueShape != tdescShape)
    return emitOpError(
        "Result shape doesn't match TensorDesc shape. when VNNI is not enabled,"
        "the result should have the same shape (or transposed shape if "
        "transpose is also enabled) as TensorDesc. When VNNI is enabled, "
        "the result should have one more dimention than the TensorDesc, "
        "with last dimention having vnni factor, but having same number of"
        "total data elements. The vnni factor are typically calculated as "
        "simd_lane_width/elementTypeBitWidth. For element type having "
        "more than 32 bits, vnni shouldn't be used.\n");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreScatterOp
//===----------------------------------------------------------------------===//
LogicalResult StoreScatterOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getValueType();
  auto maskTy = getMaskType();

  if (!tdescTy.getScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  std::vector<int64_t> maskShape = getShapeOf(maskTy);
  std::vector<int64_t> valueShape = getShapeOf(valueTy);
  std::vector<int64_t> tdescShape = tdescTy.getShape().vec();

  if (tdescShape != valueShape || tdescShape != valueShape)
    return emitOpError("Shape among TensorDesc, Value, and Mask don't match.\n");

  return success();
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
