// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/tidl/tidl_provider_factory.h"
#include <atomic>
#include "tidl_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct TIDLProviderFactory : IExecutionProviderFactory {
  TIDLProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~TIDLProviderFactory() override {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> TIDLProviderFactory::CreateProvider() {
  TIDLExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return std::make_unique<TIDLExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_TIDL(int use_arena) {
  return std::make_shared<onnxruntime::TIDLProviderFactory>(use_arena != 0);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_TIDL, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_TIDL(use_arena));
  return nullptr;
}
