## Subset
- File: y_train_subset.h5, Total Length: 36.00 seconds, 576000 samples
- File: y_val_test_subset.h5, Total Length: 18.00 seconds, 288000 samples
- File: x_train_subset.h5, Total Length: 36.00 seconds, 576000 samples
- File: x_val_test_subset.h5, Total Length: 18.00 seconds, 288000 samples

====================================================================================

## Training loop parameters
```cond_dim = 0
kernel_size = 9
n_blocks = 5
dilation_growth = 10
n_channels = 32
n_iters = 50
length = 88800
lr = 0.001
```

## Training started...
```
Using: cuda
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 40:1.00e-04 -> 47:1.00e-05
Saved model to model_TCN_00.pt
 Loss: 1.328e+00 | : 100%|██████████| 50/50 [00:03<00:00, 16.51it/s]
Saved model to model_TCN_00.pt
```