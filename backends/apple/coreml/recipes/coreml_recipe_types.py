# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.export import RecipeType


class CoreMLRecipeType(RecipeType):
    """CoreML-specific recipe types - combinations of precision and compute units"""

    # FP32 precision with different compute units
    FP32_CPU = "fp32_cpu"
    FP32_GPU = "fp32_gpu" 
    FP32_NEURAL_ENGINE = "fp32_neural_engine"
    FP32_ALL = "fp32_all"
    
    # FP16 precision with different compute units
    FP16_CPU = "fp16_cpu"
    FP16_GPU = "fp16_gpu"
    FP16_NEURAL_ENGINE = "fp16_neural_engine" 
    FP16_ALL = "fp16_all"

    @classmethod
    def get_backend_name(cls) -> str:
        return "coreml" 