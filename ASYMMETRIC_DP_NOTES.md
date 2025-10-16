# Asymmetric Model with Differential Privacy

## Overview

The asymmetric adversarial neural cryptography model can also benefit from differential privacy to protect training data. The implementation is similar to the symmetric model but accounts for the additional key generator network.

## Key Differences from Symmetric Model

1. **Three Networks Instead of Two**:
   - Alice (encryption with public key)
   - Bob (decryption with private key)
   - Key Generator (generates public keys from private keys)
   - Eve (eavesdropper with only public key)

2. **Privacy Protection Scope**:
   - DP-SGD protects all three networks: Alice, Bob, and Key Generator
   - Privacy budget is tracked across all network updates
   - Each network gets gradient clipping and noise addition

3. **Training Dynamics**:
   - Alice, Bob, and Key Generator train together (with DP)
   - Eve trains separately (with DP)
   - Privacy accountant tracks cumulative privacy loss

## Implementation Notes

To create `asymmetric_dp.ipynb`, follow the same pattern as `symmetric_dp.ipynb`:

### 1. Import Differential Privacy Module

```python
from differential_privacy import (
    DPOptimizer,
    PrivacyAccountant,
    get_privacy_preset
)
```

### 2. Configure Privacy Parameters

```python
enable_dp = True
privacy_preset = "medium"  # "high", "medium", or "low"
dp_params = get_privacy_preset(privacy_preset)
```

### 3. Create DP Optimizers

```python
if enable_dp:
    # For Alice-Bob-KeyGen network
    base_opt_abk = Adam(learning_rate=learning_rate)
    dp_opt_abk = DPOptimizer(
        optimizer=base_opt_abk,
        l2_norm_clip=dp_params['l2_norm_clip'],
        noise_multiplier=dp_params['noise_multiplier']
    )
    
    # For Eve network
    base_opt_eve = Adam(learning_rate=learning_rate)
    dp_opt_eve = DPOptimizer(
        optimizer=base_opt_eve,
        l2_norm_clip=dp_params['l2_norm_clip'],
        noise_multiplier=dp_params['noise_multiplier']
    )
    
    # Privacy accountant
    privacy_accountant = PrivacyAccountant(
        noise_multiplier=dp_params['noise_multiplier'],
        batch_size=batch_size,
        num_samples=samples,
        delta=dp_params['target_delta']
    )
```

### 4. Training Loop with DP-SGD

```python
for batch in range(batches):
    # Train Alice-Bob-KeyGen with DP
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = tn_alice_bob_keygen([p_batch, pvk_batch], training=True)
        loss = tn_alice_bob_keygen.compiled_loss(None, predictions)
    
    # Compute gradients with DP
    grads_and_vars = dp_opt_abk.compute_gradients(
        loss, tn_alice_bob_keygen.trainable_variables, tape
    )
    dp_opt_abk.apply_gradients(grads_and_vars)
    
    # Train Eve with DP (similar process)
    # ...
    
    # Update privacy accountant
    privacy_accountant.step()
    epsilon, delta = privacy_accountant.get_privacy_spent()
```

## Expected Privacy Budget

For the asymmetric model with similar training configuration:

| Privacy Level | Target ε | Expected Final ε | Training Impact |
|--------------|----------|------------------|-----------------|
| High         | ≤ 1.0    | 0.8 - 1.2        | Significant utility loss |
| Medium       | ≤ 3.0    | 2.5 - 3.5        | Moderate utility loss |
| Low          | ≤ 8.0    | 7.0 - 9.0        | Minimal utility loss |

## Performance Considerations

The asymmetric model already has lower accuracy than the symmetric model. Adding differential privacy will further reduce accuracy:

- **Without DP**: Bob ~60-70% accuracy, Eve ~50-60%
- **With High DP**: Bob ~50-60% accuracy, Eve ~45-55%
- **With Medium DP**: Bob ~55-65% accuracy, Eve ~48-58%
- **With Low DP**: Bob ~58-68% accuracy, Eve ~49-59%

## Recommendations

1. **Start with Low Privacy**: Given the asymmetric model's complexity, begin with "low" privacy preset
2. **Increase Training Time**: More epochs may be needed to compensate for DP noise
3. **Tune Carefully**: Balance privacy requirements with model utility
4. **Monitor Closely**: Track both privacy budget and model performance

## Future Enhancements

Potential improvements for asymmetric DP:

1. **Per-Network Privacy Budgets**: Track separate budgets for each network
2. **Adaptive Noise**: Reduce noise as training progresses
3. **Privacy Amplification**: Leverage subsampling for better privacy-utility tradeoff
4. **Secure Aggregation**: Combine with secure multi-party computation

## References

See `DIFFERENTIAL_PRIVACY.md` for general DP information and `symmetric_dp.ipynb` for implementation reference.
