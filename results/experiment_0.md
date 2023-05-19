## Subset
File: y_train_subset.h5, Total Length: 36.00 seconds, 576000 samples
File: y_val_test_subset.h5, Total Length: 18.00 seconds, 288000 samples
File: x_train_subset.h5, Total Length: 36.00 seconds, 576000 samples
File: x_val_test_subset.h5, Total Length: 18.00 seconds, 288000 samples

====================================================================================

## Training loop parameters
cond_dim = 0
kernel_size = 9
n_blocks = 5
dilation_growth = 10
n_channels = 32
n_iters = 50
length = 88800
lr = 0.001

## Training...
Using CPU
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 40:1.00e-04 -> 47:1.00e-05
 Loss: 1.229e+00 | : 100%|===| 50/50 [00:44<00:00,  1.12it/s]
Saved model to model_TCN_00.pth
(jupyter) [fpapaleo@node023 smc-spring-reverb]$ /homedtic/fpapaleo/.conda/envs/jupyter/bin/python /homedtic/fpapaleo/smc-spring-reverb/eval.py

## Evaluation...
Using CPU

Name: model_TCN_00.pth
MSE: 0.08094695210456848
ESR: 1.3577107191085815

Average Evaluation
Number of chunks: 10
Average MSE: 0.008094695210456849
Average ESR: 0.13577107191085816



