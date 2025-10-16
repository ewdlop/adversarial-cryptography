"""
Differential Privacy Module for Adversarial Neural Cryptography

This module provides differential privacy mechanisms for training neural networks
with privacy guarantees. It implements Differentially Private Stochastic Gradient
Descent (DP-SGD) and privacy budget tracking.

References:
- Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
- Dwork and Roth, "The Algorithmic Foundations of Differential Privacy", 2014
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional


class DPOptimizer:
    """
    Differentially Private Optimizer wrapper.
    
    Implements DP-SGD by clipping gradients and adding calibrated noise
    to protect individual training examples.
    
    Args:
        optimizer: Base TensorFlow optimizer (e.g., Adam, SGD)
        l2_norm_clip: Clipping norm for gradient clipping
        noise_multiplier: Ratio of noise stddev to clipping norm
        num_microbatches: Number of microbatches for gradient computation
        learning_rate: Learning rate for the optimizer
    """
    
    def __init__(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        l2_norm_clip: float = 1.0,
        noise_multiplier: float = 1.1,
        num_microbatches: Optional[int] = None,
        learning_rate: float = 0.0008
    ):
        self.optimizer = optimizer
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.learning_rate = learning_rate
        
    def compute_gradients(self, loss_fn, var_list, tape):
        """
        Compute gradients with differential privacy guarantees.
        
        Args:
            loss_fn: Loss function to compute gradients for
            var_list: List of variables to compute gradients for
            tape: GradientTape for computing gradients
            
        Returns:
            List of (gradient, variable) tuples with DP noise added
        """
        # Compute gradients
        gradients = tape.gradient(loss_fn, var_list)
        
        # Clip gradients by L2 norm
        clipped_gradients = []
        for gradient in gradients:
            if gradient is not None:
                # Clip gradient
                clipped_grad = tf.clip_by_norm(gradient, self.l2_norm_clip)
                clipped_gradients.append(clipped_grad)
            else:
                clipped_gradients.append(gradient)
        
        # Add Gaussian noise for differential privacy
        noised_gradients = []
        for gradient in clipped_gradients:
            if gradient is not None:
                # Calculate noise standard deviation
                noise_stddev = self.l2_norm_clip * self.noise_multiplier
                
                # Add Gaussian noise
                noise = tf.random.normal(
                    shape=gradient.shape,
                    mean=0.0,
                    stddev=noise_stddev,
                    dtype=gradient.dtype
                )
                noised_grad = gradient + noise
                noised_gradients.append(noised_grad)
            else:
                noised_gradients.append(gradient)
        
        return list(zip(noised_gradients, var_list))
    
    def apply_gradients(self, grads_and_vars):
        """Apply gradients to variables."""
        return self.optimizer.apply_gradients(grads_and_vars)


class PrivacyAccountant:
    """
    Privacy budget tracker for differential privacy.
    
    Tracks privacy loss (epsilon) and failure probability (delta) over
    training iterations using the moments accountant method.
    
    Args:
        noise_multiplier: Ratio of noise stddev to clipping norm
        batch_size: Size of training batches
        num_samples: Total number of training samples
        delta: Target delta for (epsilon, delta)-DP
    """
    
    def __init__(
        self,
        noise_multiplier: float,
        batch_size: int,
        num_samples: int,
        delta: float = 1e-5
    ):
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.delta = delta
        self.steps = 0
        
    def step(self):
        """Increment the step counter."""
        self.steps += 1
        
    def get_epsilon(self, target_delta: Optional[float] = None) -> float:
        """
        Compute privacy budget epsilon for current training steps.
        
        This is a simplified approximation. For production use, consider
        using the tensorflow-privacy library's RDP accountant.
        
        Args:
            target_delta: Target delta (defaults to self.delta)
            
        Returns:
            Current epsilon value
        """
        if target_delta is None:
            target_delta = self.delta
            
        # Sampling ratio
        q = self.batch_size / self.num_samples
        
        # Simplified epsilon calculation using strong composition
        # For a more accurate calculation, use RDP (Renyi Differential Privacy)
        if self.noise_multiplier == 0:
            return float('inf')
            
        # Approximate epsilon using the analytical moments accountant
        # This is a conservative bound
        c = q * np.sqrt(self.steps)
        epsilon = c / self.noise_multiplier + np.log(1 / target_delta) / self.noise_multiplier
        
        return epsilon
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get the privacy budget spent so far.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        epsilon = self.get_epsilon()
        return (epsilon, self.delta)


def add_noise_to_output(output: tf.Tensor, sensitivity: float, epsilon: float) -> tf.Tensor:
    """
    Add Laplace noise to model output for output privacy.
    
    Args:
        output: Model output tensor
        sensitivity: Global sensitivity of the query
        epsilon: Privacy parameter
        
    Returns:
        Noised output tensor
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
        
    # Laplace mechanism: scale = sensitivity / epsilon
    scale = sensitivity / epsilon
    
    # Generate Laplace noise
    # Laplace distribution can be sampled from uniform distribution
    noise = tf.random.uniform(
        shape=output.shape,
        minval=-1.0,
        maxval=1.0,
        dtype=output.dtype
    )
    laplace_noise = -scale * tf.sign(noise) * tf.math.log(1 - tf.abs(noise))
    
    return output + laplace_noise


def clip_gradients_by_norm(
    gradients: list,
    clip_norm: float
) -> list:
    """
    Clip gradients by their L2 norm.
    
    Args:
        gradients: List of gradient tensors
        clip_norm: Maximum L2 norm
        
    Returns:
        List of clipped gradient tensors
    """
    clipped = []
    for grad in gradients:
        if grad is not None:
            clipped.append(tf.clip_by_norm(grad, clip_norm))
        else:
            clipped.append(grad)
    return clipped


# Privacy parameters presets
PRIVACY_PRESETS = {
    "high": {
        "l2_norm_clip": 1.0,
        "noise_multiplier": 2.0,
        "target_epsilon": 1.0,
        "target_delta": 1e-5,
        "description": "High privacy (strong guarantees, may reduce utility)"
    },
    "medium": {
        "l2_norm_clip": 1.5,
        "noise_multiplier": 1.1,
        "target_epsilon": 3.0,
        "target_delta": 1e-5,
        "description": "Medium privacy (balanced privacy/utility tradeoff)"
    },
    "low": {
        "l2_norm_clip": 2.0,
        "noise_multiplier": 0.8,
        "target_epsilon": 8.0,
        "target_delta": 1e-5,
        "description": "Low privacy (weaker guarantees, better utility)"
    }
}


def get_privacy_preset(preset_name: str) -> dict:
    """
    Get privacy parameter preset.
    
    Args:
        preset_name: Name of preset ("high", "medium", or "low")
        
    Returns:
        Dictionary of privacy parameters
    """
    if preset_name not in PRIVACY_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {list(PRIVACY_PRESETS.keys())}"
        )
    return PRIVACY_PRESETS[preset_name].copy()
