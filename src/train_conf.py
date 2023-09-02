from src.default_args import parse_args

"""
This file contains the configurations for training the models.
"""

args = parse_args()

train_conf = [
    {
    'conf_name': 'tcn-1800',
    'model_type': 'TCN',
    'n_blocks': 5,
    'kernel_size': 9,
    'num_channels': 32,
    'dilation': 10,
    'cond_dim': 0,
    'criterion': 'esr',
    'pre_emphasis': 0.95,
    'batch_size': 32,
    'lr': 5e-2,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'tcn-4000',
    'model_type': 'TCN',
    'n_blocks': 5,
    'kernel_size': 19,
    'num_channels': 32,
    'dilation': 10,
    'cond_dim': 0,
    'criterion': 'mrstft',
    'pre_emphasis': None,
    'batch_size': 8,
    'lr': 5e-3,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'tcn-6000',
    'model_type': 'TCN',
    'n_blocks': 5,
    'kernel_size': 27,
    'num_channels': 32,
    'dilation': 10,
    'cond_dim': 0,
    'criterion': 'esr',
    'pre_emphasis': 0.95,
    'batch_size': 8,
    'lr': 1e-1,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'gcn-250',
    'model_type': 'GCN',
    'num_blocks': 1,
    'num_layers': 4,
    'num_channels': 16,
    'kernel_size': 41,
    'dilation_depth': 6,
    'criterion': 'mrstft',
    'pre_emphasis': None,
    'batch_size': 8,
    'lr': 5e-3,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'gcn-2500',
    'model_type': 'GCN',
    'num_blocks': 1,
    'num_layers': 10,
    'num_channels': 16,
    'kernel_size': 5,
    'dilation_depth': 3,
    'criterion': 'mrstft',
    'pre_emphasis': None,
    'batch_size': 8,
    'lr': 5e-3,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'wavenet-10',
    'model_type': 'PedalNetWaveNet',
    'num_channels': 16,
    'dilation_depth': 10,
    'num_repeat': 1,
    'kernel_size': 3,
    'criterion': 'esr',
    'pre_emphasis': 0.95,
    'batch_size': 8,
    'lr': 5e-3,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'wavenet-18',
    'model_type': 'PedalNetWaveNet',
    'num_channels': 16,
    'dilation_depth': 9,
    'num_repeat': 2,
    'kernel_size': 3,
    'criterion': 'esr',
    'pre_emphasis': 0.95,
    'batch_size': 8,
    'lr': 5e-3,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'wavenet-24',
    'model_type': 'PedalNetWaveNet',
    'num_channels': 16,
    'dilation_depth': 8,
    'num_repeat': 3,
    'kernel_size': 3,
    'criterion': 'esr',
    'pre_emphasis': 0.95,
    'batch_size': 8,
    'lr': 5e-3,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'wavenet-1k5',
    'model_type': 'PedalNetWaveNet',
    'num_channels': 8,
    'dilation_depth': 10,
    'num_repeat': 5,
    'kernel_size': 5,
    'criterion': 'esr',
    'pre_emphasis': 0.95,
    'batch_size': 8,
    'lr': 5e-3,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'LSTM-96',
    'model_type': 'LSTM',
    'input_size': 1,
    'hidden_size': 96,
    'num_layers': 1,
    'output_size': 1,
    'criterion': 'mrstft',
    'batch_size': 8,
    'lr': 5e-3,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'lstm-cs-32',
    'model_type': 'LstmConvSkip',
    'input_size': 1,
    'hidden_size': 32,
    'num_layers': 2,
    'output_size': 1,
    'use_skip': True,
    'kernel_size': 3,
    'criterion': 'esr',
    'pre_emphasis': 0.95,
    'batch_size': 8,
    'lr': 5e-2,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    {
    'conf_name': 'lstm-cs-96-v28',
    'model_type': 'LstmConvSkip',
    'input_size': 1,
    'hidden_size': 96,
    'num_layers': 1,
    'output_size': 1,
    'use_skip': True,
    'kernel_size': 3,
    'criterion': 'mrstft',
    'batch_size': 8,
    'lr': 5e-3,
    'sample_rate': args.sample_rate,
    'bit_rate': args.bit_rate,
    },
    ]
