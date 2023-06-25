hparams = {
        'model_type': 'TCN',
        'n_inputs': 1,
        'n_outputs': 1,
        'n_blocks': 10,
        'kernel_size': 15,
        'n_channels': 64,
        'dilation_growth': 2,
        'cond_dim': 0,
    }

hparams = {
    'model_type': 'WaveNet',
    'num_channels': 32,
    'dilation_depth': 10,
    'num_repeat': 3,
    'kernel_size': 15
    }