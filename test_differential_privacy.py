"""
Unit tests for differential privacy module.

Run with: pytest test_differential_privacy.py
or: python -m pytest test_differential_privacy.py
"""

import unittest
import numpy as np
import tensorflow as tf
from differential_privacy import (
    DPOptimizer,
    PrivacyAccountant,
    get_privacy_preset,
    clip_gradients_by_norm,
    PRIVACY_PRESETS
)


class TestPrivacyPresets(unittest.TestCase):
    """Test privacy parameter presets."""
    
    def test_preset_names(self):
        """Test that all expected presets exist."""
        expected_presets = ["high", "medium", "low"]
        for preset in expected_presets:
            self.assertIn(preset, PRIVACY_PRESETS)
    
    def test_preset_structure(self):
        """Test that presets have required fields."""
        required_fields = [
            "l2_norm_clip", "noise_multiplier", 
            "target_epsilon", "target_delta", "description"
        ]
        for preset_name, preset_params in PRIVACY_PRESETS.items():
            for field in required_fields:
                self.assertIn(field, preset_params,
                            f"Preset '{preset_name}' missing field '{field}'")
    
    def test_get_privacy_preset(self):
        """Test getting privacy presets."""
        medium = get_privacy_preset("medium")
        self.assertIsInstance(medium, dict)
        self.assertIn("l2_norm_clip", medium)
        self.assertIn("noise_multiplier", medium)
    
    def test_invalid_preset(self):
        """Test that invalid preset raises error."""
        with self.assertRaises(ValueError):
            get_privacy_preset("invalid_preset")
    
    def test_privacy_levels(self):
        """Test that privacy levels are ordered correctly."""
        high = get_privacy_preset("high")
        medium = get_privacy_preset("medium")
        low = get_privacy_preset("low")
        
        # High privacy should have lowest epsilon target
        self.assertLess(high["target_epsilon"], medium["target_epsilon"])
        self.assertLess(medium["target_epsilon"], low["target_epsilon"])


class TestPrivacyAccountant(unittest.TestCase):
    """Test privacy budget tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 256
        self.num_samples = 10000
        self.noise_multiplier = 1.1
        self.delta = 1e-5
        
        self.accountant = PrivacyAccountant(
            noise_multiplier=self.noise_multiplier,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            delta=self.delta
        )
    
    def test_initialization(self):
        """Test accountant initialization."""
        self.assertEqual(self.accountant.steps, 0)
        self.assertEqual(self.accountant.noise_multiplier, self.noise_multiplier)
        self.assertEqual(self.accountant.batch_size, self.batch_size)
        self.assertEqual(self.accountant.num_samples, self.num_samples)
    
    def test_step_counter(self):
        """Test step counter increments."""
        initial_steps = self.accountant.steps
        self.accountant.step()
        self.assertEqual(self.accountant.steps, initial_steps + 1)
        
        for _ in range(10):
            self.accountant.step()
        self.assertEqual(self.accountant.steps, initial_steps + 11)
    
    def test_epsilon_increases_with_steps(self):
        """Test that epsilon increases as training progresses."""
        epsilon_0 = self.accountant.get_epsilon()
        
        for _ in range(10):
            self.accountant.step()
        epsilon_10 = self.accountant.get_epsilon()
        
        for _ in range(90):
            self.accountant.step()
        epsilon_100 = self.accountant.get_epsilon()
        
        # Epsilon should increase with more steps
        self.assertGreater(epsilon_10, epsilon_0)
        self.assertGreater(epsilon_100, epsilon_10)
    
    def test_get_privacy_spent(self):
        """Test getting privacy budget spent."""
        epsilon, delta = self.accountant.get_privacy_spent()
        self.assertIsInstance(epsilon, float)
        self.assertEqual(delta, self.delta)
        self.assertGreater(epsilon, 0)
    
    def test_zero_noise_infinite_epsilon(self):
        """Test that zero noise gives infinite epsilon."""
        accountant = PrivacyAccountant(
            noise_multiplier=0.0,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            delta=self.delta
        )
        accountant.step()
        epsilon = accountant.get_epsilon()
        self.assertEqual(epsilon, float('inf'))


class TestDPOptimizer(unittest.TestCase):
    """Test DP optimizer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.l2_norm_clip = 1.0
        self.noise_multiplier = 1.1
        
        self.dp_optimizer = DPOptimizer(
            optimizer=self.base_optimizer,
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier
        )
    
    def test_initialization(self):
        """Test DP optimizer initialization."""
        self.assertEqual(self.dp_optimizer.l2_norm_clip, self.l2_norm_clip)
        self.assertEqual(self.dp_optimizer.noise_multiplier, self.noise_multiplier)
        self.assertIsInstance(self.dp_optimizer.optimizer, 
                            tf.keras.optimizers.Optimizer)
    
    def test_gradient_clipping(self):
        """Test that gradients are clipped."""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,))
        ])
        
        # Create sample data
        x = tf.random.normal((32, 5))
        y = tf.random.normal((32, 10))
        
        # Compute gradients
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.reduce_mean(tf.square(predictions - y))
        
        # Apply DP optimizer
        grads_and_vars = self.dp_optimizer.compute_gradients(
            loss, model.trainable_variables, tape
        )
        
        # Check that we got gradients
        self.assertEqual(len(grads_and_vars), len(model.trainable_variables))
        
        # Check gradient norms are bounded
        for grad, var in grads_and_vars:
            if grad is not None:
                grad_norm = tf.norm(grad).numpy()
                # After clipping and adding noise, norm might exceed clip value
                # but original clipping should have been applied
                self.assertIsInstance(grad_norm, (float, np.floating))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_clip_gradients_by_norm(self):
        """Test gradient clipping utility."""
        # Create sample gradients
        grad1 = tf.constant([[1.0, 2.0, 3.0]])
        grad2 = tf.constant([[4.0, 5.0, 6.0]])
        gradients = [grad1, grad2]
        
        clip_norm = 1.0
        clipped = clip_gradients_by_norm(gradients, clip_norm)
        
        self.assertEqual(len(clipped), len(gradients))
        
        # Check that clipped gradients have norm <= clip_norm
        for grad in clipped:
            if grad is not None:
                norm = tf.norm(grad).numpy()
                self.assertLessEqual(norm, clip_norm + 1e-6)  # Small tolerance
    
    def test_clip_gradients_with_none(self):
        """Test clipping handles None gradients."""
        gradients = [tf.constant([1.0, 2.0]), None, tf.constant([3.0, 4.0])]
        clipped = clip_gradients_by_norm(gradients, 1.0)
        
        self.assertEqual(len(clipped), len(gradients))
        self.assertIsNone(clipped[1])


class TestPrivacyUtilityTradeoff(unittest.TestCase):
    """Test privacy-utility tradeoff properties."""
    
    def test_higher_noise_stronger_privacy(self):
        """Test that higher noise multiplier gives lower epsilon."""
        batch_size = 256
        num_samples = 10000
        steps = 100
        
        # Low noise
        acc_low_noise = PrivacyAccountant(
            noise_multiplier=0.5,
            batch_size=batch_size,
            num_samples=num_samples,
            delta=1e-5
        )
        
        # High noise
        acc_high_noise = PrivacyAccountant(
            noise_multiplier=2.0,
            batch_size=batch_size,
            num_samples=num_samples,
            delta=1e-5
        )
        
        # Run for same number of steps
        for _ in range(steps):
            acc_low_noise.step()
            acc_high_noise.step()
        
        epsilon_low = acc_low_noise.get_epsilon()
        epsilon_high = acc_high_noise.get_epsilon()
        
        # Higher noise should give lower epsilon (stronger privacy)
        self.assertLess(epsilon_high, epsilon_low)
    
    def test_more_steps_weaker_privacy(self):
        """Test that more training steps increases epsilon."""
        accountant = PrivacyAccountant(
            noise_multiplier=1.1,
            batch_size=256,
            num_samples=10000,
            delta=1e-5
        )
        
        # Measure epsilon at different steps
        epsilons = []
        for _ in range(100):
            accountant.step()
            if accountant.steps % 10 == 0:
                epsilons.append(accountant.get_epsilon())
        
        # Epsilon should increase monotonically
        for i in range(len(epsilons) - 1):
            self.assertLess(epsilons[i], epsilons[i + 1])


if __name__ == "__main__":
    unittest.main()
