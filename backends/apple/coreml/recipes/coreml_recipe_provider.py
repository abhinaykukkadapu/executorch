# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Optional, Sequence

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    ct = None

from executorch.backends.apple.coreml.recipes.coreml_recipe_types import CoreMLRecipeType
from executorch.export import (
    BackendRecipeProvider,
    ExportRecipe,
    LoweringRecipe,
    RecipeType,
)

# Import CoreML components only when needed
def _get_coreml_backend():
    try:
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        return CoreMLBackend
    except ImportError:
        return None

def _get_coreml_partitioner():
    try:
        from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
        return CoreMLPartitioner
    except ImportError:
        return None


class CoreMLRecipeProvider(BackendRecipeProvider):
    @property
    def backend_name(self) -> str:
        return "coreml"

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return list(CoreMLRecipeType)

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        """Create CoreML recipe with precision and compute unit combinations"""

        if recipe_type not in self.get_supported_recipes():
            return None

        if not COREML_AVAILABLE:
            raise ImportError(
                "coremltools is required for CoreML recipes. "
                "Install it with: pip install coremltools"
            )

        # Validate kwargs - these simple recipes don't accept parameters
        if kwargs:
            unexpected = list(kwargs.keys())
            raise ValueError(
                f"Recipe '{recipe_type.value}' does not accept any parameters. "
                f"Unexpected parameters: {unexpected}"
            )

        # Parse recipe type to get precision and compute unit
        recipe_value = recipe_type.value
        if recipe_value.startswith("fp32_"):
            precision = ct.precision.FLOAT32
            compute_unit_str = recipe_value[5:]  # Remove "fp32_"
        elif recipe_value.startswith("fp16_"):
            precision = ct.precision.FLOAT16
            compute_unit_str = recipe_value[5:]  # Remove "fp16_"
        else:
            raise ValueError(f"Unknown recipe type: {recipe_type.value}")

        # Map compute unit string to CoreML compute unit
        compute_unit_map = {
            "cpu": ct.ComputeUnit.CPU_ONLY,
            "gpu": ct.ComputeUnit.CPU_AND_GPU,
            "neural_engine": ct.ComputeUnit.CPU_AND_NE,
            "all": ct.ComputeUnit.ALL,
        }
        
        if compute_unit_str not in compute_unit_map:
            raise ValueError(f"Unknown compute unit: {compute_unit_str}")
            
        compute_unit = compute_unit_map[compute_unit_str]

        return self._build_recipe(recipe_type, precision, compute_unit)

    def _build_recipe(
        self, 
        recipe_type: RecipeType, 
        precision: "ct.precision", 
        compute_unit: "ct.ComputeUnit"
    ) -> ExportRecipe:
        """Build a CoreML recipe with the specified precision and compute unit"""
        
        lowering_recipe = self._get_coreml_lowering_recipe(
            compute_precision=precision,
            compute_unit=compute_unit,
        )
        
        return ExportRecipe(
            name=recipe_type.value,
            lowering_recipe=lowering_recipe,
        )

    def _get_coreml_lowering_recipe(
        self,
        compute_unit: "ct.ComputeUnit",
        compute_precision: "ct.precision",
        minimum_deployment_target = None,  # Default to iOS15
        take_over_mutable_buffer: bool = True,
    ) -> LoweringRecipe:
        """Create lowering recipe with CoreML partitioner configuration"""
        
        # Get CoreML components with conditional imports
        CoreMLBackend = _get_coreml_backend()
        CoreMLPartitioner = _get_coreml_partitioner()
        
        if CoreMLBackend is None or CoreMLPartitioner is None:
            raise ImportError("CoreML backend components are not available")
            
        # Set default deployment target
        if minimum_deployment_target is None and ct is not None:
            minimum_deployment_target = ct.target.iOS15
            
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_unit=compute_unit,
            minimum_deployment_target=minimum_deployment_target,
            compute_precision=compute_precision,
        )
        
        partitioner = CoreMLPartitioner(
            compile_specs=compile_specs,
            take_over_mutable_buffer=take_over_mutable_buffer,
        )
        
        return LoweringRecipe(partitioners=[partitioner]) 