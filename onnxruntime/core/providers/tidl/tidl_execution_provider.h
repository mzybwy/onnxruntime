// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct TIDL execution providers.
struct TIDLExecutionProviderInfo {
  bool create_arena{true};

  explicit TIDLExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  TIDLExecutionProviderInfo() = default;
};

// Logical device representation.
class TIDLExecutionProvider : public IExecutionProvider {
 public:
  explicit TIDLExecutionProvider(const TIDLExecutionProviderInfo& info);
  virtual ~TIDLExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& kernel_registries) const override;

  const void* GetExecutionHandle() const noexcept override {
    // The TIDL interface does not return anything interesting.
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime
