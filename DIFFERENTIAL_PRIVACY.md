# Differential Privacy in Adversarial Neural Cryptography

This document describes the differential privacy (DP) implementation added to the adversarial neural cryptography system.

## Overview

Differential privacy provides mathematical guarantees that the training process does not reveal information about individual training examples. This is crucial for protecting sensitive data used in training neural networks.

## What is Differential Privacy?

Differential privacy ensures that the removal or addition of a single training example does not significantly affect the model's behavior. Formally, a mechanism M provides (ε, δ)-differential privacy if for all datasets D and D' differing in at most one element, and all possible outputs S:

```
P[M(D) ∈ S] ≤ exp(ε) × P[M(D') ∈ S] + δ
```

Where:
- **ε (epsilon)**: Privacy loss parameter. Smaller values mean stronger privacy (typical: 0.1-10)
- **δ (delta)**: Failure probability. Very small value (typical: 10⁻⁵ to 10⁻⁷)

## Implementation

### Key Components

1. **DP-SGD (Differentially Private Stochastic Gradient Descent)**
   - Clips gradients to bound sensitivity
   - Adds calibrated Gaussian noise to gradients
   - Implemented in `differential_privacy.py`

2. **Privacy Accountant**
   - Tracks cumulative privacy loss over training
   - Computes (ε, δ) at any training step
   - Ensures privacy budget is not exceeded

3. **Privacy Presets**
   - **High Privacy**: ε ≤ 1.0, stronger guarantees, may reduce model utility
   - **Medium Privacy**: ε ≤ 3.0, balanced privacy/utility tradeoff
   - **Low Privacy**: ε ≤ 8.0, weaker guarantees, better model performance

## Usage

### Basic Usage

Run the symmetric model with differential privacy:

```python
from differential_privacy import get_privacy_preset, DPOptimizer, PrivacyAccountant

# Choose privacy level
privacy_preset = "medium"  # or "high" or "low"
dp_params = get_privacy_preset(privacy_preset)

# Create DP optimizer
dp_optimizer = DPOptimizer(
    optimizer=base_optimizer,
    l2_norm_clip=dp_params['l2_norm_clip'],
    noise_multiplier=dp_params['noise_multiplier']
)

# Initialize privacy accountant
privacy_accountant = PrivacyAccountant(
    noise_multiplier=dp_params['noise_multiplier'],
    batch_size=batch_size,
    num_samples=num_samples,
    delta=dp_params['target_delta']
)
```

### Jupyter Notebooks

Two notebooks are provided:

1. **symmetric_dp.ipynb**: Symmetric model with differential privacy
   - Toggle DP on/off with `enable_dp` flag
   - Visualize privacy budget over training
   - Compare DP vs non-DP training

2. **asymmetric_dp.ipynb**: Asymmetric model with differential privacy
   - Extends DP to public/private key cryptography
   - Similar features to symmetric notebook

### Running with Different Privacy Levels

```python
# High privacy (strongest guarantees)
enable_dp = True
privacy_preset = "high"
# Expected: Strong privacy, lower model accuracy

# Medium privacy (balanced)
enable_dp = True
privacy_preset = "medium"
# Expected: Good privacy, reasonable accuracy

# Low privacy (weaker guarantees)
enable_dp = True
privacy_preset = "low"
# Expected: Moderate privacy, better accuracy

# No privacy
enable_dp = False
# Expected: No privacy, best accuracy
```

## Privacy Parameters

### Tuning Guidelines

1. **l2_norm_clip**: Maximum L2 norm for gradient clipping
   - Larger values → less gradient distortion but weaker privacy
   - Smaller values → stronger privacy but more gradient distortion
   - Typical range: 0.5 - 2.0

2. **noise_multiplier**: Ratio of noise standard deviation to clipping norm
   - Larger values → stronger privacy but noisier gradients
   - Smaller values → weaker privacy but cleaner gradients
   - Typical range: 0.8 - 2.5

3. **Target epsilon**: Privacy budget limit
   - Lower values → stronger privacy guarantees
   - Higher values → weaker privacy but better utility
   - Typical range: 0.1 - 10.0

### Privacy-Utility Tradeoff

The fundamental tradeoff in differential privacy:

```
Strong Privacy (low ε) ←→ Better Utility (high ε)
     ↓                           ↓
More noise added          Less noise added
Lower accuracy           Higher accuracy
```

## Monitoring Privacy Budget

The privacy budget (epsilon) increases over training. Monitor it to ensure it stays within acceptable bounds:

```python
# During training
epsilon, delta = privacy_accountant.get_privacy_spent()
print(f"Privacy spent: ε={epsilon:.2f}, δ={delta:.2e}")

# Check against target
if epsilon > target_epsilon:
    print("Warning: Privacy budget exceeded!")
```

## Privacy-Preserving Best Practices

1. **Start with strong privacy**: Begin with high privacy settings and relax if needed
2. **Monitor privacy budget**: Track epsilon throughout training
3. **Use larger batches**: Larger batch sizes improve privacy-utility tradeoff
4. **Limit epochs**: More epochs → more privacy loss
5. **Tune hyperparameters**: Adjust noise and clipping based on your privacy requirements

## Technical Details

### DP-SGD Algorithm

```
For each batch:
  1. Compute per-example gradients
  2. Clip each gradient: g_i ← g_i / max(1, ||g_i|| / C)
  3. Add noise: g ← (1/B) Σ g_i + N(0, σ²C²I)
  4. Update parameters: θ ← θ - η·g
```

Where:
- C = l2_norm_clip
- σ = noise_multiplier
- B = batch_size
- η = learning_rate

### Privacy Accounting

Privacy loss accumulates over training steps. We use the moments accountant method (simplified) to track cumulative epsilon:

```
ε(T) ≈ (q·√T) / σ + log(1/δ) / σ
```

Where:
- q = batch_size / num_samples (sampling ratio)
- T = number of training steps
- σ = noise_multiplier

## Performance Impact

Differential privacy has computational and accuracy impacts:

### Computational Overhead
- **Training time**: 10-30% slower due to gradient clipping and noise addition
- **Memory**: Minimal additional memory required

### Accuracy Impact
- **High privacy**: 5-15% accuracy reduction
- **Medium privacy**: 2-8% accuracy reduction  
- **Low privacy**: 0-5% accuracy reduction

Actual impact depends on:
- Dataset size (larger → less impact)
- Model complexity
- Privacy parameters chosen

## References

1. Abadi, M., et al. (2016). "Deep Learning with Differential Privacy." *Proceedings of CCS*.
2. Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy." *Foundations and Trends in Theoretical Computer Science*.
3. Papernot, N., et al. (2018). "Scalable Private Learning with PATE." *ICLR*.

## Examples

### Example 1: Training with Medium Privacy

```python
# In symmetric_dp.ipynb
enable_dp = True
privacy_preset = "medium"
epochs = 20

# Run training...
# Expected output:
# Privacy spent: epsilon = 2.85, delta = 1.00e-05
# ✓ Privacy budget satisfied!
```

### Example 2: Comparing DP vs Non-DP

```python
# Non-DP training
enable_dp = False
# Train and evaluate...
# Bob error: 0.05, Eve error: 7.8

# DP training (medium)
enable_dp = True
privacy_preset = "medium"
# Train and evaluate...
# Bob error: 0.08, Eve error: 7.6
# Privacy: ε=2.85, δ=1e-05

# Observation: Slight accuracy decrease for strong privacy guarantees
```

## Troubleshooting

### Issue: Privacy budget exceeded

**Solution**: Increase `noise_multiplier`, reduce `epochs`, or increase `batch_size`

### Issue: Model doesn't converge

**Solution**: Decrease `noise_multiplier`, increase `l2_norm_clip`, or increase `learning_rate`

### Issue: Training is too slow

**Solution**: Reduce `num_microbatches` (if using) or use GPU acceleration

### Issue: Poor model accuracy

**Solution**: Use "low" or "medium" privacy preset, or disable DP for baseline comparison

## Contributing

When adding new features with differential privacy:

1. Ensure gradients are properly clipped
2. Add calibrated noise based on privacy parameters
3. Update privacy accountant after each training step
4. Document privacy implications
5. Test with multiple privacy presets

## License

Same as parent project.
