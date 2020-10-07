// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_c_api.h>
/* #include <onnxruntime/core/session/onnxruntime_cxx_api.h> */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_arena zero: false. non-zero: true.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Tidl, _In_ OrtSessionOptions* options, int use_arena);

#ifdef __cplusplus
}
#endif