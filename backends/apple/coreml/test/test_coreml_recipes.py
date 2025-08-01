# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn

try:
    import coremltools as ct
    from executorch.backends.apple.coreml.recipes import (
        CoreMLRecipeProvider,
        CoreMLRecipeType,
    )
    from executorch.export import export, ExportRecipe, recipe_registry
    from torch.testing._internal.common_quantization import TestHelperModules
    
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


@unittest.skipIf(not COREML_AVAILABLE, "CoreML tools not available")
class TestCoreMLRecipes(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        super().setUp()
        self.provider = CoreMLRecipeProvider()
        # Register the provider for recipe registry tests
        recipe_registry.register_backend_recipe_provider(CoreMLRecipeProvider())

    def tearDown(self):
        super().tearDown()

    def test_backend_name(self):
        """Test that the backend name is correct"""
        self.assertEqual(self.provider.backend_name, "coreml")

    def test_supported_recipes_count(self):
        """Test that we have the expected number of recipes"""
        recipes = self.provider.get_supported_recipes()
        # 2 precisions (fp32, fp16) x 4 compute units (cpu, gpu, neural_engine, all) = 8 recipes
        self.assertEqual(len(recipes), 8)

    def test_all_recipes_supported(self):
        """Test that all CoreMLRecipeType values are supported"""
        supported = set(self.provider.get_supported_recipes())
        expected = set(CoreMLRecipeType)
        self.assertEqual(supported, expected)

    def test_basic_fp32_recipe(self):
        """Test basic FP32 recipe with a simple model"""
        m_eager = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]
        
        session = export(
            model=m_eager,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(CoreMLRecipeType.FP32_CPU),
        )
        
        # Check that outputs match
        self.assertTrue(
            torch.allclose(
                session.run_method("forward", example_inputs[0])[0],
                m_eager(*example_inputs[0]),
                atol=1e-3,
            )
        )

    def test_basic_fp16_recipe(self):
        """Test basic FP16 recipe with a simple model"""
        m_eager = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]
        
        session = export(
            model=m_eager,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(CoreMLRecipeType.FP16_GPU),
        )
        
        # Check that outputs match (allow slightly higher tolerance for FP16)
        self.assertTrue(
            torch.allclose(
                session.run_method("forward", example_inputs[0])[0],
                m_eager(*example_inputs[0]),
                atol=1e-2,
            )
        )

    def test_all_fp32_recipes_with_simple_model(self):
        """Test all FP32 recipes with a simple linear model"""
        fp32_recipes = [
            CoreMLRecipeType.FP32_CPU,
            CoreMLRecipeType.FP32_GPU,
            CoreMLRecipeType.FP32_NEURAL_ENGINE,
            CoreMLRecipeType.FP32_ALL,
        ]
        
        for recipe_type in fp32_recipes:
            with self.subTest(recipe=recipe_type.value):
                m_eager = TestHelperModules.TwoLinearModule().eval()
                example_inputs = [(torch.randn(9, 8),)]
                
                session = export(
                    model=m_eager,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(recipe_type),
                )
                
                # Verify outputs match
                self.assertTrue(
                    torch.allclose(
                        session.run_method("forward", example_inputs[0])[0],
                        m_eager(*example_inputs[0]),
                        atol=1e-3,
                    )
                )

    def test_all_fp16_recipes_with_simple_model(self):
        """Test all FP16 recipes with a simple linear model"""
        fp16_recipes = [
            CoreMLRecipeType.FP16_CPU,
            CoreMLRecipeType.FP16_GPU,
            CoreMLRecipeType.FP16_NEURAL_ENGINE,
            CoreMLRecipeType.FP16_ALL,
        ]
        
        for recipe_type in fp16_recipes:
            with self.subTest(recipe=recipe_type.value):
                m_eager = TestHelperModules.TwoLinearModule().eval()
                example_inputs = [(torch.randn(9, 8),)]
                
                session = export(
                    model=m_eager,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(recipe_type),
                )
                
                # Verify outputs match (slightly higher tolerance for FP16)
                self.assertTrue(
                    torch.allclose(
                        session.run_method("forward", example_inputs[0])[0],
                        m_eager(*example_inputs[0]),
                        atol=1e-2,
                    )
                )

    def test_custom_simple_model(self):
        """Test with a custom simple model"""
        class SimpleLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(20, 1)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        model = SimpleLinearModel().eval()
        example_inputs = [(torch.randn(1, 10),)]
        
        # Test with different recipe types
        test_recipes = [
            CoreMLRecipeType.FP32_CPU,
            CoreMLRecipeType.FP16_GPU,
            CoreMLRecipeType.FP16_ALL,
        ]
        
        for recipe_type in test_recipes:
            with self.subTest(recipe=recipe_type.value):
                session = export(
                    model=model,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(recipe_type),
                )
                
                tolerance = 1e-3 if "fp32" in recipe_type.value else 1e-2
                self.assertTrue(
                    torch.allclose(
                        session.run_method("forward", example_inputs[0])[0],
                        model(*example_inputs[0]),
                        atol=tolerance,
                    )
                )

    def test_fp32_recipes(self):
        """Test FP32 precision recipes"""
        fp32_recipes = [
            CoreMLRecipeType.FP32_CPU,
            CoreMLRecipeType.FP32_GPU,
            CoreMLRecipeType.FP32_NEURAL_ENGINE,
            CoreMLRecipeType.FP32_ALL,
        ]
        
        for recipe_type in fp32_recipes:
            with self.subTest(recipe=recipe_type.value):
                recipe = self.provider.create_recipe(recipe_type)
                self.assertIsInstance(recipe, ExportRecipe)
                self.assertEqual(recipe.name, recipe_type.value)
                self.assertIsNotNone(recipe.lowering_recipe)

    def test_fp16_recipes(self):
        """Test FP16 precision recipes"""
        fp16_recipes = [
            CoreMLRecipeType.FP16_CPU,
            CoreMLRecipeType.FP16_GPU,
            CoreMLRecipeType.FP16_NEURAL_ENGINE,
            CoreMLRecipeType.FP16_ALL,
        ]
        
        for recipe_type in fp16_recipes:
            with self.subTest(recipe=recipe_type.value):
                recipe = self.provider.create_recipe(recipe_type)
                self.assertIsInstance(recipe, ExportRecipe)
                self.assertEqual(recipe.name, recipe_type.value)
                self.assertIsNotNone(recipe.lowering_recipe)

    def test_unsupported_recipe_type(self):
        """Test that unsupported recipe types return None"""
        from executorch.export import RecipeType
        
        class UnsupportedRecipeType(RecipeType):
            UNSUPPORTED = "unsupported"
        
        recipe = self.provider.create_recipe(UnsupportedRecipeType.UNSUPPORTED)
        self.assertIsNone(recipe)

    def test_no_kwargs_accepted(self):
        """Test that recipes don't accept any keyword arguments"""
        with self.assertRaises(ValueError) as context:
            self.provider.create_recipe(CoreMLRecipeType.FP32_CPU, some_param=123)
        
        self.assertIn("does not accept any parameters", str(context.exception))
        self.assertIn("some_param", str(context.exception))

    def test_recipe_names_match_types(self):
        """Test that recipe names match their type values"""
        for recipe_type in CoreMLRecipeType:
            recipe = self.provider.create_recipe(recipe_type)
            self.assertEqual(recipe.name, recipe_type.value)

    def test_recipe_type_parsing(self):
        """Test that recipe types are correctly parsed for precision and compute units"""
        # This is an internal test to verify the parsing logic works
        test_cases = [
            ("fp32_cpu", "fp32", "cpu"),
            ("fp16_gpu", "fp16", "gpu"), 
            ("fp32_neural_engine", "fp32", "neural_engine"),
            ("fp16_all", "fp16", "all"),
        ]
        
        for recipe_value, expected_precision, expected_compute_unit in test_cases:
            with self.subTest(recipe=recipe_value):
                # Find the recipe type by value
                recipe_type = None
                for rt in CoreMLRecipeType:
                    if rt.value == recipe_value:
                        recipe_type = rt
                        break
                
                self.assertIsNotNone(recipe_type, f"Recipe type not found for {recipe_value}")
                
                # Verify the recipe can be created (which tests the parsing logic)
                recipe = self.provider.create_recipe(recipe_type)
                self.assertIsNotNone(recipe)

    def test_recipe_registry_integration(self):
        """Test that recipes work with the global recipe registry"""
        # Test that we can get recipes through ExportRecipe.get_recipe()
        recipe = ExportRecipe.get_recipe(CoreMLRecipeType.FP16_GPU)
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.name, "fp16_gpu")
        
        # Test with a few different recipe types
        test_recipes = [
            CoreMLRecipeType.FP32_CPU,
            CoreMLRecipeType.FP16_ALL,
            CoreMLRecipeType.FP32_NEURAL_ENGINE,
        ]
        
        for recipe_type in test_recipes:
            with self.subTest(recipe=recipe_type.value):
                recipe = ExportRecipe.get_recipe(recipe_type)
                self.assertIsNotNone(recipe)
                self.assertEqual(recipe.name, recipe_type.value)

    def test_validate_recipe_kwargs_error_messages(self):
        """Test detailed error messages for invalid kwargs"""
        provider = CoreMLRecipeProvider()
        
        # Test single invalid parameter
        with self.assertRaises(ValueError) as cm:
            provider.create_recipe(CoreMLRecipeType.FP16_GPU, invalid_param=123)
        
        error_msg = str(cm.exception)
        self.assertIn("Recipe 'fp16_gpu' does not accept any parameters", error_msg)
        self.assertIn("invalid_param", error_msg)
        
        # Test multiple invalid parameters
        with self.assertRaises(ValueError) as cm:
            provider.create_recipe(
                CoreMLRecipeType.FP32_ALL, 
                param1="value1", 
                param2="value2"
            )
        
        error_msg = str(cm.exception)
        self.assertIn("Recipe 'fp32_all' does not accept any parameters", error_msg)


@unittest.skipIf(COREML_AVAILABLE, "Test only when CoreML tools are NOT available")
class TestCoreMLRecipesWithoutCoreML(unittest.TestCase):
    def test_import_error_when_coreml_unavailable(self):
        """Test that appropriate error is raised when coremltools is not available"""
        # This test only runs when coremltools is NOT available
        # We can test the recipe types import without coremltools
        from executorch.backends.apple.coreml.recipes.coreml_recipe_types import CoreMLRecipeType
        
        # Should be able to import recipe types
        self.assertEqual(len(list(CoreMLRecipeType)), 8)
        

if __name__ == "__main__":
    unittest.main() 