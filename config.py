# Folders
DATASET = '/homedtic/fpapaleo/smc-spring-reverb/dataset_subset'

# Training loop parameters
cond_dim = 2
kernel_size = 7 # Change this value to a smaller number
n_blocks = 5
dilation_growth = 8
n_channels = 16
n_iters = 3500
length = 160000
lr = 0.001

# Define sample rate
sample_rate = 16000

# Define input and output channels
input_channels = 1
output_channels = 1