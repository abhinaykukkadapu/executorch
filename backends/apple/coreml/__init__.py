# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Exposed Partitioners in CoreML Package
from .partition.coreml_partitioner import CoreMLPartitioner

# CoreML Backend
from .compiler.coreml_preprocess import CoreMLBackend

# Exposed Recipes in CoreML Package - with optional registration
try:
    from .recipes.coreml_recipe_provider import CoreMLRecipeProvider
    from .recipes.coreml_recipe_types import CoreMLRecipeType
    
    # Auto-register CoreML recipe provider when available
    try:
        from executorch.export import recipe_registry
        recipe_registry.register_backend_recipe_provider(CoreMLRecipeProvider())
    except ImportError:
        # Recipe registry not available - skip auto-registration
        pass
        
except ImportError:
    # coremltools or other dependencies not available - skip recipes
    CoreMLRecipeProvider = None
    CoreMLRecipeType = None

__all__ = [
    "CoreMLPartitioner",
    "CoreMLBackend",
    "CoreMLRecipeType",
] 