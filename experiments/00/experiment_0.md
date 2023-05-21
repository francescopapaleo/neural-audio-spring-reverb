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
 Loss: 1.323e+00 | : 100%|██████████| 50/50 [00:02<00:00, 18.88it/s]
Saved model to model_TCN_00.pt
```

## Evaluation started...
Using device: cuda

Name: model_TCN_00.pt
MSE: 0.09637971967458725
ESR: 1.6165622472763062
model_TCN_00.pt
Error-to-Signal Ratio (mean): 1.6165622472763062
Error-to-Signal Ratio (sum): 1.6165622472763062

MAE: 0.1869836300611496

## Evaluation by chunks...
Number of chunks: 10
Average MSE: 0.009637971967458725
Average ESR: 0.1616562247276306


## Inference started...
Using device: cuda
model_TCN_00.pt
Mean Squared Error: 0.09637971967458725
Error-to-Signal Ratio (mean): 1.616562008857727
Error-to-Signal Ratio (sum): 1.6165621280670166

Error-to-Signal Ratio: 1.6165621280670166
Saving audio files

Plotting the results

Estimated RT60: 7.2034375

## Evaluation started...
Using device: cpu

Name: /Users/francescopapaleo/git-box/smc-spring-reverb/models/model_TCN_00.pt
Normalization: applying min-max scaling
MSE: 0.10632546991109848
L1: 0.2132655829191208
STFT: 1.9838253259658813
ESR: 1.5822157859802246

## Evaluation by chunks...
Number of chunks: 10
Average MSE: 0.010632546991109848
Average L1: 0.02132655829191208
Average STFT: 0.19838253259658814
Average ESR: 0.15822157859802247

