#!/usr/bin/env python3
"""
Example: Differential Privacy in Adversarial Neural Cryptography

This script demonstrates how to use differential privacy with the adversarial
neural cryptography system. It shows the impact of different privacy levels
on model training and privacy budgets.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Concatenate, Reshape, Dense, Conv1D, Flatten
from keras.optimizers import Adam

# Import our differential privacy module
from differential_privacy import (
    DPOptimizer,
    PrivacyAccountant,
    get_privacy_preset,
    PRIVACY_PRESETS
)


def create_simple_network(input_dim, name="network"):
    """Create a simple neural network for demonstration."""
    inputs = Input(shape=(input_dim,), name=f"{name}_input")
    x = Dense(units=32, activation="relu", name=f"{name}_dense")(inputs)
    x = Reshape(target_shape=(32, 1), name=f"{name}_reshape")(x)
    x = Conv1D(filters=2, kernel_size=4, strides=1, padding="same", 
               activation="relu", name=f"{name}_conv")(x)
    x = Flatten(name=f"{name}_flatten")(x)
    outputs = Dense(units=input_dim, activation="tanh", name=f"{name}_output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model


def demonstrate_privacy_presets():
    """Demonstrate different privacy presets and their parameters."""
    print("=" * 70)
    print("DIFFERENTIAL PRIVACY PRESETS")
    print("=" * 70)
    
    for preset_name, params in PRIVACY_PRESETS.items():
        print(f"\n{preset_name.upper()} Privacy:")
        print(f"  Description: {params['description']}")
        print(f"  L2 Norm Clip: {params['l2_norm_clip']}")
        print(f"  Noise Multiplier: {params['noise_multiplier']}")
        print(f"  Target Epsilon: {params['target_epsilon']}")
        print(f"  Target Delta: {params['target_delta']}")


def demonstrate_privacy_accounting():
    """Demonstrate privacy budget tracking over training steps."""
    print("\n" + "=" * 70)
    print("PRIVACY BUDGET TRACKING")
    print("=" * 70)
    
    # Configuration
    batch_size = 256
    num_samples = 65536
    num_steps = 100
    
    preset = get_privacy_preset("medium")
    
    accountant = PrivacyAccountant(
        noise_multiplier=preset['noise_multiplier'],
        batch_size=batch_size,
        num_samples=num_samples,
        delta=preset['target_delta']
    )
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total samples: {num_samples}")
    print(f"  Noise multiplier: {preset['noise_multiplier']}")
    
    print(f"\nPrivacy budget over training steps:")
    print(f"{'Step':<8} {'Epsilon':<10} {'Status'}")
    print("-" * 40)
    
    milestones = [1, 10, 25, 50, 75, 100]
    for step in range(1, num_steps + 1):
        accountant.step()
        
        if step in milestones:
            epsilon, delta = accountant.get_privacy_spent()
            status = "✓ OK" if epsilon <= preset['target_epsilon'] else "⚠ EXCEEDED"
            print(f"{step:<8} {epsilon:<10.2f} {status}")
    
    final_epsilon, final_delta = accountant.get_privacy_spent()
    print("\n" + "-" * 40)
    print(f"Final privacy: ε={final_epsilon:.2f}, δ={final_delta:.2e}")
    print(f"Target: ε≤{preset['target_epsilon']}")


def demonstrate_dp_optimizer():
    """Demonstrate DP optimizer with gradient clipping and noise."""
    print("\n" + "=" * 70)
    print("DP-SGD GRADIENT PROCESSING")
    print("=" * 70)
    
    # Create a simple model
    model = create_simple_network(input_dim=16, name="demo")
    
    # Sample data
    x_sample = np.random.choice([-1, 1], size=(1, 16)).astype(np.float32)
    y_sample = np.random.choice([-1, 1], size=(1, 16)).astype(np.float32)
    
    # Regular optimizer
    print("\nRegular Optimizer (no DP):")
    regular_optimizer = Adam(learning_rate=0.001)
    
    with tf.GradientTape() as tape:
        predictions = model(x_sample, training=True)
        loss = tf.reduce_mean(tf.square(predictions - y_sample))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    grad_norms = [tf.norm(g).numpy() if g is not None else 0 for g in gradients]
    print(f"  Sample gradient norms: {grad_norms[:3]}")
    print(f"  Max gradient norm: {max(grad_norms):.4f}")
    
    # DP optimizer
    print("\nDP Optimizer (with clipping and noise):")
    preset = get_privacy_preset("medium")
    base_optimizer = Adam(learning_rate=0.001)
    dp_optimizer = DPOptimizer(
        optimizer=base_optimizer,
        l2_norm_clip=preset['l2_norm_clip'],
        noise_multiplier=preset['noise_multiplier']
    )
    
    with tf.GradientTape() as tape:
        predictions = model(x_sample, training=True)
        loss = tf.reduce_mean(tf.square(predictions - y_sample))
    
    # Compute gradients with DP
    grads_and_vars = dp_optimizer.compute_gradients(
        loss, model.trainable_variables, tape
    )
    
    dp_gradients = [g for g, _ in grads_and_vars]
    dp_grad_norms = [tf.norm(g).numpy() if g is not None else 0 for g in dp_gradients]
    print(f"  Sample DP gradient norms: {dp_grad_norms[:3]}")
    print(f"  Max DP gradient norm: {max(dp_grad_norms):.4f}")
    print(f"  Clipping threshold: {preset['l2_norm_clip']}")
    print(f"  Noise std dev: {preset['l2_norm_clip'] * preset['noise_multiplier']:.4f}")


def demonstrate_privacy_utility_tradeoff():
    """Show the privacy-utility tradeoff across different privacy levels."""
    print("\n" + "=" * 70)
    print("PRIVACY-UTILITY TRADEOFF")
    print("=" * 70)
    
    print("\nExpected behavior with different privacy levels:")
    print("-" * 70)
    
    scenarios = [
        ("No DP", None, "No privacy", "Best accuracy", "Baseline"),
        ("Low DP", "low", "ε ≤ 8.0", "~95-100% baseline", "Minimal impact"),
        ("Medium DP", "medium", "ε ≤ 3.0", "~92-98% baseline", "Good balance"),
        ("High DP", "high", "ε ≤ 1.0", "~85-95% baseline", "Strong privacy")
    ]
    
    print(f"{'Setting':<12} {'Privacy':<12} {'Expected Accuracy':<20} {'Notes'}")
    print("-" * 70)
    
    for setting, preset, privacy, accuracy, notes in scenarios:
        print(f"{setting:<12} {privacy:<12} {accuracy:<20} {notes}")
    
    print("\nKey insights:")
    print("  • Stronger privacy (lower ε) → More noise → Lower accuracy")
    print("  • Weaker privacy (higher ε) → Less noise → Higher accuracy")
    print("  • Larger datasets and models handle DP better")
    print("  • Privacy cost decreases with larger batch sizes")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("DIFFERENTIAL PRIVACY DEMONSTRATION")
    print("Adversarial Neural Cryptography with Privacy Guarantees")
    print("=" * 70)
    
    # Run demonstrations
    demonstrate_privacy_presets()
    demonstrate_privacy_accounting()
    demonstrate_dp_optimizer()
    demonstrate_privacy_utility_tradeoff()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Open symmetric_dp.ipynb to train with differential privacy")
    print("2. Try different privacy presets: 'high', 'medium', 'low'")
    print("3. Monitor privacy budget (epsilon) during training")
    print("4. Compare DP vs non-DP model performance")
    print("5. Read DIFFERENTIAL_PRIVACY.md for detailed documentation")
    
    print("\n" + "=" * 70)
    print("Example notebook configuration:")
    print("=" * 70)
    print("""
enable_dp = True
privacy_preset = "medium"  # Choose: "high", "medium", or "low"
epochs = 20

# Training will display:
# - Privacy spent: epsilon = X.XX, delta = X.XXe-XX
# - Privacy budget status
# - Model performance with DP guarantees
""")


if __name__ == "__main__":
    main()
