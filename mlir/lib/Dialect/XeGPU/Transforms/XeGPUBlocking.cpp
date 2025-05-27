//===---- XeGPUBlocking.cpp ---- XeGPU Blocking Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUBLOCKING
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-blocking"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

// reslove the unrealized conversion cast ops generated when doing SCF
// Structural Type Conversion. It will have two formats, N:1 vector
// cast and 1:N vector cast. vector::insert_strided_slice ops will be
// used for the first case, and vector::extract_strided_slice ops will be
// used for the second case.
static void
resolveUnrealizedConversionCastOp(UnrealizedConversionCastOp castOp) {
  ValueRange inputs = castOp.getInputs();
  ValueRange outputs = castOp.getOutputs();
  if (inputs.empty() || outputs.empty()) {
    LDBG("erase unrealized conversion cast op has no inputs/outputs.");
    castOp->erase();
    return;
  }

  VectorType inputTy = dyn_cast<VectorType>(inputs[0].getType());
  VectorType outputTy = dyn_cast<VectorType>(outputs[0].getType());
  if (!inputTy || !outputTy) {
    LDBG("skip unrealized conversion cast op has non-vector inputs/outputs.");
    return;
  }

  // We only interest in the case where all inputs and outputs have the
  // identical types
  if (llvm::any_of(castOp->getOperandTypes(),
                   [&](Type t) { return t != inputTy; }) ||
      llvm::any_of(castOp->getResultTypes(),
                   [&](Type t) { return t != outputTy; })) {
    LDBG("skip unrealized conversion cast op not emulating pack/unpack.");
    return;
  }

  OpBuilder builder(castOp);
  if (inputs.size() > 1 && outputs.size() == 1) {
    // the castOp is emulating an unpack op
    ArrayRef<int64_t> shape = outputTy.getShape();
    Value result = xegpu::createVectorWithShapeFromValues(
        builder, castOp.getLoc(), inputs, shape);
    castOp->replaceAllUsesWith(ValueRange(result));
    castOp->erase();
  } else if (castOp.getNumResults() > 1 && castOp.getNumOperands() == 1) {
    // the castOp is emulating a pack op
    ArrayRef<int64_t> tileShape = outputTy.getShape();
    SmallVector<Value> results = xegpu::extractVectorsWithShapeFromValue(
        builder, castOp.getLoc(), inputs[0], tileShape);
    castOp->replaceAllUsesWith(results);
    castOp->erase();
  }
}

/// Unroll XeGPU ops to their instruction-level representation.
class XeGPUBlockingPass final
    : public xegpu::impl::XeGPUBlockingBase<XeGPUBlockingPass> {
public:
  void runOnOperation() override;

private:
  // Get the tile shape for a given operand by examining the layout attribute.
  // If layout is not present or is not a subgroup level layout, it returns
  // std::nullopt.
  std::optional<SmallVector<int64_t>> getTileShape(OpOperand &operand) const;

  // Get the tile shape for a given result by examining the layout attribute.
  // If layout is not present or is not a subgroup level layout, it returns
  // std::nullopt.
  std::optional<SmallVector<int64_t>> getTileShape(OpResult result) const;

  // Get the tile shape for a given operation.
  std::optional<SmallVector<int64_t>> getTileShape(Operation *op) const;

  // Determine if the operation requires unrolling. Return false if all operands
  // and results have tile shapes identical to their original types. Otherwise,
  // return true.
  bool needsUnroll(Operation *op) const;
};
} // namespace

std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(OpOperand &operand) const {
  xegpu::LayoutAttr layout = xegpu::getLayoutAttr(operand);
  if (layout && layout.isSgLayout()) {
    if (auto inst_data = layout.getInstData())
      return llvm::to_vector_of<int64_t>(inst_data.asArrayRef());

    if (auto type = dyn_cast<ShapedType>(operand.get().getType()))
      return llvm::to_vector(type.getShape());
  }
  LDBG("failed to getTileShape for operand: " << operand.get());
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(OpResult result) const {
  xegpu::LayoutAttr layout = xegpu::getLayoutAttr(result);
  if (layout && layout.isSgLayout()) {
    if (auto inst_data = layout.getInstData())
      return llvm::to_vector_of<int64_t>(inst_data.asArrayRef());

    if (auto type = dyn_cast<ShapedType>(result.getType()))
      return llvm::to_vector(type.getShape());
  }
  LDBG("failed to getTileShape for result: " << result);
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
XeGPUBlockingPass::getTileShape(Operation *op) const {
  if (isa<xegpu::CreateNdDescOp, xegpu::UpdateNdOffsetOp>(op))
    return getTileShape(op->getOpResult(0));
  if (isa<xegpu::PrefetchNdOp, xegpu::LoadNdOp>(op))
    return getTileShape(op->getOpOperand(0));
  if (isa<xegpu::StoreNdOp>(op))
    return getTileShape(op->getOpOperand(1));

  if (isa<xegpu::DpasOp>(op)) {
    std::optional<SmallVector<int64_t>> aTile =
        getTileShape(op->getOpOperand(0));
    std::optional<SmallVector<int64_t>> bTile =
        getTileShape(op->getOpOperand(1));

    if (!aTile || aTile->size() != 2 || !bTile || bTile->size() != 2)
      return std::nullopt;

    // semantic check for A and B
    if ((*aTile)[1] != (*bTile)[0])
      return std::nullopt;

    // semantic check for C
    if (op->getNumOperands() == 3) {
      std::optional<SmallVector<int64_t>> cTile =
          getTileShape(op->getOpOperand(2));
      int64_t expectedCTile[2] = {(*aTile)[0], (*bTile)[1]};
      if (!cTile || !llvm::equal(*cTile, expectedCTile))
        return std::nullopt;
    }

    return SmallVector<int64_t>({(*aTile)[0], (*aTile)[1], (*bTile)[1]});
  }

  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)
    return getTileShape(op->getOpResult(0));

  return std::nullopt;
}

bool XeGPUBlockingPass::needsUnroll(Operation *op) const {
  if (isa<LoopLikeOpInterface>(op))
    return false;

  auto isUnrollable = [](Value value,
                         ArrayRef<int64_t> tileShape) -> std::optional<bool> {
    Type valTy = value.getType();
    if (auto tdesc = dyn_cast<xegpu::TensorDescType>(valTy)) {
      xegpu::LayoutAttr layout = tdesc.getLayoutAttr();
      if (!layout)
        return std::nullopt;
      if (layout.isWgLayout())
        return false;
      if (layout.getInstData())
        return true;
    }

    auto shapedType = dyn_cast<ShapedType>(valTy);
    if (shapedType && !llvm::equal(tileShape, shapedType.getShape()))
      return true;

    return std::nullopt;
  };

  for (OpOperand &opr : op->getOpOperands()) {
    std::optional<SmallVector<int64_t>> tileShape = getTileShape(opr);
    if (!tileShape)
      continue;

    std::optional<bool> unrollable = isUnrollable(opr.get(), *tileShape);
    if (unrollable.has_value())
      return unrollable.value();
  }

  for (OpResult result : op->getOpResults()) {
    std::optional<SmallVector<int64_t>> tileShape = getTileShape(result);
    if (!tileShape)
      continue;

    std::optional<bool> unrollable = isUnrollable(result, *tileShape);
    if (unrollable.has_value())
      return unrollable.value();
  }
  return false;
}

void XeGPUBlockingPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  Operation *op = getOperation();

  // Preserve the LayoutAttr for each operand to the owner's DictionaryAttr.
  // This ensures that the LayoutAttr remains accessible even if the defining
  // operation is replaced.
  xegpu::setLayoutAttrs(op, [](Value v) { return xegpu::getLayoutAttr(v); });

  auto getTileShapeAndCount = [](llvm::ArrayRef<int64_t> shape,
                                 xegpu::LayoutAttr layout) {
    int count = 1;
    SmallVector<int64_t> tileShape(shape);
    if (layout && layout.getInstData()) {
      DenseI32ArrayAttr instData = layout.getInstData();
      tileShape = llvm::to_vector_of<int64_t>(instData.asArrayRef());
      count = computeProduct(shape) / computeProduct(tileShape);
    }
    return std::make_pair(tileShape, count);
  };

  // Perform type conversion for SCF control folow ops
  TypeConverter converter;
  converter.addConversion([](Type type) -> Type { return type; });
  converter.addConversion(
      [&](RankedTensorType type,
          SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        Type elemTy = type.getElementType();
        ArrayRef<int64_t> shape = type.getShape();

        auto layout =
            llvm::dyn_cast_if_present<xegpu::LayoutAttr>(type.getEncoding());
        if (layout && layout.isWgLayout())
          return failure();

        int count;
        SmallVector<int64_t> subShape;
        std::tie(subShape, count) = getTileShapeAndCount(shape, layout);
        auto newTy = VectorType::get(subShape, elemTy);
        result.append(count, newTy);
        return success();
      });
  converter.addConversion(
      [&](xegpu::TensorDescType type,
          SmallVectorImpl<Type> &result) -> std::optional<LogicalResult> {
        Type elemTy = type.getElementType();
        ArrayRef<int64_t> shape = type.getShape();

        xegpu::LayoutAttr layout = type.getLayoutAttr();
        if (layout && layout.isWgLayout())
          return failure();

        int count;
        SmallVector<int64_t> subShape;
        std::tie(subShape, count) = getTileShapeAndCount(shape, layout);

        if (layout)
          layout = layout.dropInstData();

        auto newTy = xegpu::TensorDescType::get(
            type.getContext(), subShape, elemTy, type.getEncoding(), layout);
        result.append(count, newTy);
        return success();
      });

  xegpu::doSCFStructuralTypeConversionWithTensorType(op, converter);

  xegpu::UnrollOptions options;
  options.setFilterConstraint(
      [&](Operation *op) -> LogicalResult { return success(needsUnroll(op)); });

  options.setNativeShapeFn([&](Operation *op) { return getTileShape(op); });

  options.setUnrolledTypesFn([&](ShapedType type, ArrayRef<int64_t> tileShape) {
    Type elemTy = type.getElementType();
    Type newTy;

    if (auto tdescTy = dyn_cast<xegpu::TensorDescType>(type))
      newTy = xegpu::TensorDescType::get(
          ctx, tileShape, elemTy, tdescTy.getEncoding(),
          tdescTy.getLayoutAttr().dropInstData());
    else
      newTy = type.clone(tileShape, elemTy);

    std::optional<SmallVector<int64_t>> ratio =
        computeShapeRatio(type.getShape(), tileShape);
    assert(ratio && "The shape of the type must be a multiple of tileShape.");
    return SmallVector<Type>(computeProduct(*ratio), newTy);
  });

  RewritePatternSet patterns(ctx);

  vector::UnrollVectorOptions vectorOptions;
  vectorOptions.setNativeShapeFn(options.nativeShape);

  populateXeGPUUnrollPatterns(patterns, options);
  vector::populateVectorUnrollPatterns(patterns, vectorOptions);

  (void)applyPatternsGreedily(op, std::move(patterns));

  op->walk([](Operation *op) {
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op))
      resolveUnrealizedConversionCastOp(castOp);

    for (OpOperand &opr : op->getOpOperands()) {
      std::string name = xegpu::getLayoutName(opr);
      if (auto layout = op->getAttrOfType<xegpu::LayoutAttr>(name))
        op->removeAttr(name);
    }

    for (OpResult result : op->getOpResults()) {
      std::string name = xegpu::getLayoutName(result);
      if (auto layout = op->getAttrOfType<xegpu::LayoutAttr>(name)) {
        op->removeAttr(name);
        if (!isa<LoopLikeOpInterface>(op))
          xegpu::setLayoutAttr(result, layout.dropInstData());
      }
    }
  });
}
