// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/tidl/tidl_provider_factory.h"
#include <atomic>
#include "tidl_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {
struct TidlProviderFactory : IExecutionProviderFactory {
  TidlProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~TidlProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> TidlProviderFactory::CreateProvider() {
  TIDLExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return onnxruntime::make_unique<TIDLExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tidl(int device_id) {
  return std::make_shared<onnxruntime::TidlProviderFactory>(device_id);
  //TODO: This is apparently a bug. The consructor parameter is create-arena-flag, not the device-id
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tidl, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tidl(use_arena));
  return nullptr;
}
