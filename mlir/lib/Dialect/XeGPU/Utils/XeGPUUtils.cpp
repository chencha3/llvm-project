//===---- XeGPUUtils.cpp - MLIR Utilities for XeGPUOps   ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility methods for working with the XeGPU dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

FailureOr<VectorType>
mlir::xegpu::getDistributedVectorType(xegpu::TensorDescType tdescTy) {
  auto layout = llvm::dyn_cast_if_present<LayoutAttr>(tdescTy.getLayout());
  // It only works for subgroup level layout, which only has lane_layout
  // and lane_data, and is to distribute a SIMD code into SIMT code.
  if (!layout || !layout.isSgLayout())
    return failure();

  SmallVector<int64_t> laneData(layout.getLaneData().asArrayRef());
  SmallVector<int64_t> laneLayout(layout.getLaneLayout().asArrayRef());
  auto tdescShape = tdescTy.getShape();
  auto elementType = tdescTy.getElementType();

  // compute sgSize by multiply elements of laneLayout
  // e.g. for 2D layout, sgSize = laneLayout[0] * laneLayout[1]
  // e.g. for 1D layout, sgSize = laneLayout[0]
  auto sgSize = std::accumulate(laneLayout.begin(), laneLayout.end(), 1,
                                std::multiplies<int64_t>());

  // Case 1: regular loads/stores
  auto scatterAttr = tdescTy.getEncodingAsScatterTensorDescAttr();
  if (scatterAttr) {
    auto chunkSize = scatterAttr.getChunkSize().getInt();
    // Verify if the first dimension of the tensor descriptor shape is
    // distributable.
    assert(tdescShape[0] == laneLayout[0] &&
           "tensor descriptor shape is not distributable");
    return VectorType::get({chunkSize}, elementType);
  }

  // Case 2: block loads/stores
  // Check if the tensor descriptor shape is distributable.
  int64_t tensorSize = 1;
  for (auto [tdescDim, laneDim, laneDataDim] :
       llvm::zip_equal(tdescShape, laneLayout, laneData)) {
    assert((tdescDim % (laneDim * laneDataDim) == 0) &&
           "tensor descriptor shape is not distributable");
    tensorSize *= tdescDim;
  }
  // tensorSize must be adjusted for array_length.
  tensorSize *= tdescTy.getArrayLength();

  return VectorType::get({tensorSize / sgSize}, elementType);
}

FailureOr<VectorType>
mlir::xegpu::getDistributedVectorType(VectorType originalType,
                                      xegpu::LayoutAttr layout) {
  int64_t rank = originalType.getRank();
  // Distributed vector type is only supported for 1D, 2D and 3D vectors.
  if (rank < 1 || rank > 3)
    return failure();
  ArrayRef<int64_t> shape = originalType.getShape();
  // arrayLength is 1 for 1D and 2D vectors, and equal to the first dimension
  // of the 3D vector.
  int arrayLength = 1;
  if (rank == 3) {
    arrayLength = shape[0];
    shape = shape.drop_front();
  }
  auto helperTdescTy = xegpu::TensorDescType::get(
      shape, originalType.getElementType(), arrayLength,
      /*boundary_check=*/true,
      /*memory_space=*/xegpu::MemorySpace::Global, layout);
  return xegpu::getDistributedVectorType(helperTdescTy);
}

xegpu::LayoutAttr xegpu::getLayoutAttr(Value value) {
  if (!value)
    return nullptr;

  if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(value.getType()))
    return tdescTy.getLayoutAttr();

  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *defOp = result.getDefiningOp();
    assert(defOp && "result must have a defining op");

    // for LoadNdOp, the layout is stored in the tensor descriptor
    if (auto loadNd = dyn_cast<xegpu::LoadNdOp>(defOp))
      return getLayoutAttr(loadNd.getTensorDesc());

    std::string layoutName = getLayoutName(result);
    if (defOp->hasAttr(layoutName))
      return defOp->getAttrOfType<xegpu::LayoutAttr>(layoutName);
  }

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto parentOp = arg.getOwner()->getParentOp();
    if (auto loop = dyn_cast<LoopLikeOpInterface>(parentOp)) {
      OpOperand *tiedInit = loop.getTiedLoopInit(arg);
      return getLayoutAttr(tiedInit->get());
    }
  }

  return nullptr;
}

std::string xegpu::getLayoutName(OpOperand &opr) {
  const StringRef prefix("layout_operand_");
  return llvm::formatv("{0}{1}", prefix, opr.getOperandNumber()).str();
}

std::string xegpu::getLayoutName(OpResult res) {
  const StringRef prefix = "layout_result_";
  return llvm::formatv("{0}{1}", prefix, res.getResultNumber()).str();
}

void xegpu::doSCFStructuralTypeConversionWithTensorType(Operation *op) {
  MLIRContext *context = op->getContext();

  auto materializeCast = [&](OpBuilder &builder, Type type, ValueRange inputs,
                             Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
        .getResult(0);
  };

  { // convert VectorType to RankedTensorType for SCF Structural ops
    TypeConverter converter;
    converter.addConversion([&](Type type) -> Type { return type; });
    converter.addConversion([&](VectorType type) -> Type {
      return RankedTensorType::get(type.getShape(), type.getElementType());
    });
    converter.addSourceMaterialization(materializeCast);
    converter.addTargetMaterialization(materializeCast);

    mlir::ConversionTarget target(*context);
    target.addLegalOp<UnrealizedConversionCastOp>();

    mlir::RewritePatternSet patterns(context);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    (void)mlir::applyPartialConversion(op, target, std::move(patterns));
  }

  { // propagate the layout attribute to RankedTensorType by checking
    // BuiltInUnrealizedCastOps
    // for VectorType to RankedTensorType cast.
    op->walk([&](UnrealizedConversionCastOp castOp) {
      if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1)
        return WalkResult::skip();

      Value input = castOp.getInputs()[0];
      Value result = castOp.getResults()[0];
      auto inputTy = dyn_cast<VectorType>(input.getType());
      auto resultTy = dyn_cast<RankedTensorType>(result.getType());

      // Only look at ops casting from VectorType to RankedTensorType
      if (!isa<VectorType>(inputTy) || !isa<RankedTensorType>(resultTy))
        return WalkResult::skip();

      xegpu::LayoutAttr layout = xegpu::getLayoutAttr(input);
      if (!layout)
        return WalkResult::skip();

      RankedTensorType newTy = resultTy.cloneWithEncoding(layout);
      result.setType(newTy);

      // update the arguments if user is a LoopLike op.
      for (OpOperand &use : result.getUses()) {
        if (auto loop = dyn_cast<LoopLikeOpInterface>(use.getOwner())) {
          BlockArgument arg = loop.getTiedLoopRegionIterArg(&use);
          arg.setType(newTy);
        }
        // whileOp has two regions, the BlockArgument of the after region
        // is not exposed by LoopLikeOpInterface
        if (auto whileOp = dyn_cast<scf::WhileOp>(use.getOwner())) {
          unsigned idx = use.getOperandNumber();
          BlockArgument arg = whileOp.getAfterArguments()[idx];
          arg.setType(newTy);
        }
      }
      return WalkResult::advance();
    });

    // using yieldOp as anchor to update the result type of its ParentOp
    op->walk([&](scf::YieldOp yieldOp) {
      Operation *parentOp = yieldOp->getParentOp();
      for (OpResult r : parentOp->getOpResults()) {
        unsigned idx = r.getResultNumber();
        Type resultTy = r.getType();
        Type yieldTy = yieldOp.getResults()[idx].getType();
        if (isa<RankedTensorType>(resultTy) && yieldTy != resultTy)
          r.setType(yieldTy);
      }
    });
  }

  { // perform the conversion from RankedTensorType to VectorType based on the
    // LayoutAttr

    auto computeTileShapeAndCount = [&](ArrayRef<int64_t> shape,
                                        DenseI32ArrayAttr sgDataAttr,
                                        DenseI32ArrayAttr sgLayoutAttr) {
      SmallVector<int64_t> tileShape;
      auto sgLayout = llvm::to_vector_of<int64_t>(sgLayoutAttr.asArrayRef());
      if (sgDataAttr)
        tileShape = llvm::to_vector_of<int64_t>(sgDataAttr.asArrayRef());
      else
        tileShape = computeShapeRatio(shape, sgLayout).value_or(tileShape);
      assert(tileShape.size() && "failed to compute tileShape");
      SmallVector<int64_t> distUnit =
          computeElementwiseMul(sgLayout, tileShape);
      int count = computeProduct(shape) / computeProduct(distUnit);
      return std::make_pair(tileShape, count);
    };

    TypeConverter converter;
    converter.addConversion([&](Type type) -> Type { return type; });
    converter.addConversion(
        [&](RankedTensorType type,
            SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
          ArrayRef<int64_t> shape = type.getShape();
          auto encoding = type.getEncoding();
          Type elemTy = type.getElementType();

          // init count and subShape to the default value. If the LayoutAttr
          // is not present, it will return a VectorType with original shape.
          int count = 1;
          SmallVector<int64_t> subShape(shape);

          if (auto layout =
                  llvm::dyn_cast_if_present<xegpu::LayoutAttr>(encoding)) {
            if (layout.isWgLayout()) {
              // for WgToSg, the subShape is either from sgData or computed as
              // shape/sgLayout
              std::tie(subShape, count) = computeTileShapeAndCount(
                  shape, layout.getSgData(), layout.getSgLayout());
            } else if (DenseI32ArrayAttr instData = layout.getInstData()) {
              // for unrolling, the subShape is determined by inst_data
              subShape = llvm::to_vector_of<int64_t>(instData.asArrayRef());
              count = computeProduct(shape) / computeProduct(subShape);
            }
          }
          auto newTy = VectorType::get(subShape, elemTy);
          result.append(count, newTy);
          return success();
        });

    converter.addConversion(
        [&](xegpu::TensorDescType type,
            SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
          MLIRContext *ctx = type.getContext();
          Type elemTy = type.getElementType();
          Attribute encoding = type.getEncoding();
          ArrayRef<int64_t> shape = type.getShape();

          // init count and newTy to the default value. If the layout attribute
          // is not present, it will return the original type.
          int count = 1;
          Type newTy = type;

          if (xegpu::LayoutAttr layout = type.getLayoutAttr()) {
            SmallVector<int64_t> subShape, distUnit;
            if (layout.isWgLayout()) {
              // for WgToSg, the subShape is either from sgData or computed as
              // shape/sgLayout
              std::tie(subShape, count) = computeTileShapeAndCount(
                  shape, layout.getSgData(), layout.getSgLayout());
              layout = layout.dropSgLayoutAndData();
            } else if (DenseI32ArrayAttr instData = layout.getInstData()) {
              // for unrolling, the subShape is determined by inst_data
              subShape = llvm::to_vector_of<int64_t>(instData.asArrayRef());
              count = computeProduct(shape) / computeProduct(subShape);
              layout = layout.dropInstData();
            }
            newTy = xegpu::TensorDescType::get(ctx, subShape, elemTy, encoding,
                                               layout);
          }

          result.append(count, newTy);
          return success();
        });

    converter.addSourceMaterialization(materializeCast);
    converter.addTargetMaterialization(materializeCast);

    mlir::ConversionTarget target(*context);
    target.addLegalOp<UnrealizedConversionCastOp>();

    mlir::RewritePatternSet patterns(context);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    (void)mlir::applyPartialConversion(op, target, std::move(patterns));
  }
}
