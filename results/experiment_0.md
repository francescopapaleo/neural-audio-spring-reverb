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
```Using: cpu
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 40:1.00e-04 -> 47:1.00e-05
 Loss: 1.245e+00 | : 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [01:09<00:00,  1.38s/it]
Saved model to model_TCN_00.pt
```

## Evaluation started...
```Using CPU

Name: model_TCN_00.pt
MSE: 0.07450679689645767
ESR: 1.2496908903121948

### Evaluation by chunks...
Number of chunks: 10
Average MSE: 0.0074506796896457676
Average ESR: 0.12496908903121948
```

## Predicting on new data
```Using CPU
model_TCN_00.pt
Mean Squared Error: 0.07450679689645767
Error-to-Signal Ratio (mean): 1.2496908903121948
Error-to-Signal Ratio (sum): 1.2496908903121948

Error-to-Signal Ratio: 1.2496908903121948
Saving audio files

Plotting the results
```