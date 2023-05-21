## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 10, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 8:1.00e-04 -> 9:1.00e-05
 Loss at iteration 10: 2.603e+00 | : 100%|██████████████████████████████████████████████| 10/10 [00:01<00:00,  5.39it/s]



## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 100, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 80:1.00e-04 -> 95:1.00e-05
 Loss at iteration 100: 1.036e+00 | : 100%|███████████████████████████████████████████| 100/100 [00:05<00:00, 19.82it/s]


## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 1000, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 800:1.00e-04 -> 950:1.00e-05
 Loss at iteration 1000: 6.732e-01 | : 100%|████████████████████████████████████████| 1000/1000 [00:37<00:00, 26.95it/s]


## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 2000, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 1600:1.00e-04 -> 1900:1.00e-05
 Loss at iteration 2000: 6.468e-01 | : 100%|████████████████████████████████████████| 2000/2000 [01:14<00:00, 26.84it/s]


## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 2500, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 2000:1.00e-04 -> 2375:1.00e-05
 Loss at iteration 2500: 6.265e-01 | : 100%|████████████████████████████████████████| 2500/2500 [01:33<00:00, 26.65it/s]
Saved model to tcn_2500.pt


## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 3500, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 2800:1.00e-04 -> 3325:1.00e-05
 Loss at iteration 3500: 5.886e-01 | : 100%|████████████████████████████████████████| 3500/3500 [02:11<00:00, 26.65it/s]
Saved model to tcn_3500.pt

(envtorch) [fpapaleo@node023 smc-spring-reverb]$ python train.py
## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 4000, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 3200:1.00e-04 -> 3800:1.00e-05
 Loss at iteration 4000: 5.838e-01 | : 100%|████████████████████████████████████████| 4000/4000 [02:32<00:00, 26.20it/s]
Saved model to tcn_4000.pt

(envtorch) [fpapaleo@node023 smc-spring-reverb]$ python train.py
## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 5000, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 4000:1.00e-04 -> 4750:1.00e-05
 Loss at iteration 5000: 5.474e-01 | : 100%|████████████████████████████████████████| 5000/5000 [03:11<00:00, 26.12it/s]
Saved model to tcn_5000.pt


## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 6000, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 4800:1.00e-04 -> 5700:1.00e-05
 Loss at iteration 6000: 5.315e-01 | : 100%|████████████████████████████████████████| 6000/6000 [03:49<00:00, 26.17it/s]
Saved model to tcn_6000.pt


## Training started...
Using device: cuda
Shape before concatenation: (18, 32000, 1)
Shape before concatenation: (18, 32000, 1)
Shape of x torch.Size([1, 1, 576000])
Shape of y torch.Size([1, 1, 576000])
Shape of c: torch.Size([1, 1, 2])
Hyperparameters: {'cond_dim': 0, 'kernel_size': 9, 'n_blocks': 5, 'dilation_growth': 10, 'n_channels': 32, 'n_iters': 7000, 'length': 88800, 'lr': 0.001, 'batch_size': 1, 'c': 0.0}
Parameters: 31.494 k
Receptive field: 88889 samples or 5555.6 ms
Learning rate schedule: 1:1.00e-03 -> 5600:1.00e-04 -> 6650:1.00e-05
 Loss at iteration 7000: 5.343e-01 | : 100%|████████████████████████████████████████| 7000/7000 [04:25<00:00, 26.34it/s]
Saved model to tcn_7000.pt